from __future__ import annotations

import csv
import zipfile
from io import TextIOWrapper

import numpy as np
import pandas as pd

from .config import (
    CROPS,
    DATA_DIR,
    ELEMENTS,
    EXPECTED_MEMBERS,
    FIG_DATA_DIR,
    MAIN_MEMBER,
    RICE_CODES,
    TABLE_DIR,
    YEAR_COLS,
    YEARS,
    ZIP_PATH,
)
from .utils import get_logger, save_csv, write_json

LOGGER = get_logger()


LOOKUPS = {
    "elements": "Trade_DetailedTradeMatrix_E_Elements.csv",
    "flags": "Trade_DetailedTradeMatrix_E_Flags.csv",
    "items": "Trade_DetailedTradeMatrix_E_ItemCodes.csv",
    "partners": "Trade_DetailedTradeMatrix_E_PartnerCountries.csv",
    "reporters": "Trade_DetailedTradeMatrix_E_ReporterCountries.csv",
}


def _read_zip_csv(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as handle:
        return pd.read_csv(handle)


def inspect_archive() -> dict:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Required data archive not found: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        missing = sorted(set(EXPECTED_MEMBERS) - set(names))
        if missing:
            raise RuntimeError(f"Archive is missing expected members: {missing}")
        inventory = []
        for info in zf.infolist():
            inventory.append(
                {
                    "member": info.filename,
                    "uncompressed_bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "compression_ratio": info.compress_size / info.file_size if info.file_size else np.nan,
                }
            )
        with zf.open(MAIN_MEMBER) as handle:
            header = next(csv.reader(TextIOWrapper(handle, encoding="utf-8-sig", newline="")))
        lookups = {name: _read_zip_csv(zf, member) for name, member in LOOKUPS.items()}

    save_csv(pd.DataFrame(inventory), DATA_DIR / "archive_inventory.csv")
    save_csv(pd.DataFrame({"column": header}), DATA_DIR / "main_table_fields.csv")
    for name, frame in lookups.items():
        save_csv(frame, DATA_DIR / f"lookup_{name}.csv")

    year_cols = [c for c in header if c.startswith("Y") and c[1:].isdigit()]
    result = {
        # Manifests and audit files must be portable and must not expose a
        # workstation path or user profile.
        "archive": ZIP_PATH.name,
        "members": names,
        "columns": header,
        "years": [int(c[1:]) for c in year_cols],
        "expected_members_present": True,
    }
    write_json(result, DATA_DIR / "archive_inspection.json")
    return result


def _country_lists(zf: zipfile.ZipFile) -> tuple[set[int], set[int]]:
    reporters = _read_zip_csv(zf, LOOKUPS["reporters"])
    partners = _read_zip_csv(zf, LOOKUPS["partners"])
    reporter_code = "Reporter Country Code"
    partner_code = "Partner Country Code"

    # FAOSTAT geographic/regional aggregates in these lookup tables use FAO
    # codes >= 5000. This code-based rule is reproducible and avoids fuzzy names.
    reporter_keep = reporters[reporter_code].lt(5000)
    partner_keep = partners[partner_code].lt(5000)
    save_csv(reporters[reporter_keep].assign(status="kept_country"), DATA_DIR / "reporter_countries_kept.csv")
    save_csv(reporters[~reporter_keep].assign(status="removed_aggregate"), DATA_DIR / "reporter_entities_removed.csv")
    save_csv(partners[partner_keep].assign(status="kept_country_or_territory"), DATA_DIR / "partner_countries_kept.csv")
    save_csv(partners[~partner_keep].assign(status="removed_aggregate"), DATA_DIR / "partner_entities_removed.csv")
    return set(reporters.loc[reporter_keep, reporter_code].astype(int)), set(partners.loc[partner_keep, partner_code].astype(int))


def prepare_core_data(force: bool = False, chunksize: int = 40_000) -> dict:
    core_path = DATA_DIR / "core_four_crops_long.parquet"
    rice31_path = DATA_DIR / "rice_code31_long.parquet"
    if core_path.exists() and rice31_path.exists() and not force:
        LOGGER.info("Using existing filtered Parquet files")
        return read_prep_summary()

    inspect = inspect_archive()
    if inspect["years"] != YEARS:
        raise RuntimeError(f"Unexpected year coverage: {inspect['years']}")

    metadata_cols = [
        "Reporter Country Code",
        "Reporter Country Code (M49)",
        "Reporter Countries",
        "Partner Country Code",
        "Partner Country Code (M49)",
        "Partner Countries",
        "Item Code",
        "Item",
        "Element Code",
        "Element",
        "Unit",
    ]
    usecols = metadata_cols + YEAR_COLS
    selected_codes = set(CROPS) | set(RICE_CODES)
    selected = []
    scanned_rows = 0

    with zipfile.ZipFile(ZIP_PATH) as zf:
        reporter_codes, partner_codes = _country_lists(zf)
        items = _read_zip_csv(zf, LOOKUPS["items"])
        rice_names = items[items["Item Code"].isin(RICE_CODES)].sort_values("Item Code")
        save_csv(rice_names, DATA_DIR / "rice_item_codes_27_32.csv")
        with zf.open(MAIN_MEMBER) as handle:
            reader = pd.read_csv(
                handle,
                usecols=usecols,
                chunksize=chunksize,
                low_memory=False,
            )
            for idx, chunk in enumerate(reader, start=1):
                scanned_rows += len(chunk)
                mask = (
                    chunk["Item Code"].isin(selected_codes)
                    & chunk["Element Code"].isin(ELEMENTS)
                    & chunk["Unit"].eq("t")
                    & chunk["Reporter Country Code"].isin(reporter_codes)
                    & chunk["Partner Country Code"].isin(partner_codes)
                )
                if mask.any():
                    selected.append(chunk.loc[mask].copy())
                if idx % 50 == 0:
                    LOGGER.info("Scanned %s rows; retained %s selected wide rows", f"{scanned_rows:,}", f"{sum(map(len, selected)):,}")

    if not selected:
        raise RuntimeError("No rows matched the crop, element, unit, and country filters")
    wide = pd.concat(selected, ignore_index=True)
    del selected
    for col in YEAR_COLS:
        wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)

    grain = [
        "Reporter Country Code",
        "Reporter Country Code (M49)",
        "Reporter Countries",
        "Partner Country Code",
        "Partner Country Code (M49)",
        "Partner Countries",
        "Item Code",
        "Item",
        "Element Code",
        "Element",
        "Unit",
    ]
    duplicate_rows = int(wide.duplicated(grain, keep=False).sum())
    if duplicate_rows:
        LOGGER.warning("Aggregating %d duplicate-grain rows", duplicate_rows)
        wide = wide.groupby(grain, as_index=False, dropna=False)[YEAR_COLS].sum()

    rice_wide = wide[wide["Item Code"].isin(RICE_CODES)]
    rice_annual = (
        rice_wide.groupby(["Item Code", "Item", "Element Code", "Element"], as_index=False)[YEAR_COLS]
        .sum()
        .melt(id_vars=["Item Code", "Item", "Element Code", "Element"], value_vars=YEAR_COLS, var_name="year", value_name="quantity_tonnes")
    )
    rice_annual["year"] = rice_annual["year"].str[1:].astype(int)
    save_csv(rice_annual, FIG_DATA_DIR / "FigS2_rice_codes_annual.csv")

    long_id = grain
    core_wide = wide[wide["Item Code"].isin(CROPS)]
    core = core_wide.melt(id_vars=long_id, value_vars=YEAR_COLS, var_name="year", value_name="quantity_tonnes")
    core["year"] = core["year"].str[1:].astype("int16")
    core["crop"] = core["Item Code"].map(CROPS)
    core["direction"] = core["Element Code"].map(ELEMENTS)
    core["quantity_tonnes"] = core["quantity_tonnes"].astype(float)
    core.to_parquet(core_path, index=False, compression="zstd")

    rice31_wide = wide[wide["Item Code"].eq(31)]
    rice31 = rice31_wide.melt(id_vars=long_id, value_vars=YEAR_COLS, var_name="year", value_name="quantity_tonnes")
    rice31["year"] = rice31["year"].str[1:].astype("int16")
    rice31["crop"] = "Rice_milled_sensitivity"
    rice31["direction"] = rice31["Element Code"].map(ELEMENTS)
    rice31.to_parquet(rice31_path, index=False, compression="zstd")

    mirror = (
        core.groupby(["crop", "direction", "year"], as_index=False)["quantity_tonnes"]
        .sum()
        .pivot(index=["crop", "year"], columns="direction", values="quantity_tonnes")
        .reset_index()
    )
    mirror.columns.name = None
    mirror["export_import_ratio"] = mirror["Export"] / mirror["Import"].replace(0, np.nan)
    mirror["export_minus_import_tonnes"] = mirror["Export"] - mirror["Import"]
    save_csv(mirror, DATA_DIR / "annual_global_import_export_mirror.csv")
    save_csv(mirror, FIG_DATA_DIR / "FigS1_global_mirror.csv")

    summary = {
        "scanned_wide_rows": scanned_rows,
        "selected_wide_rows": len(wide),
        "core_long_rows": len(core),
        "duplicate_grain_rows": duplicate_rows,
        "years": YEARS,
        "crop_codes": CROPS,
        "reporter_countries_retained": int(core["Reporter Country Code"].nunique()),
        "partner_countries_retained": int(core["Partner Country Code"].nunique()),
        "bilateral_edges": int(core[["Reporter Country Code", "Partner Country Code", "Item Code", "Element Code"]].drop_duplicates().shape[0]),
        "main_source_member": MAIN_MEMBER,
        "country_filter": "FAOSTAT country/territory rows with FAO code < 5000; aggregate rows >= 5000 removed",
    }
    write_json(summary, DATA_DIR / "prep_summary.json")
    LOGGER.info("Saved %s core long rows to %s", f"{len(core):,}", core_path)
    return summary


def read_prep_summary() -> dict:
    import json

    path = DATA_DIR / "prep_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_table1(core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (crop, direction), group in core.groupby(["crop", "direction"], sort=False):
        edge_keys = ["Reporter Country Code", "Partner Country Code"]
        edge_year = group.groupby(edge_keys + ["year"], as_index=False)["quantity_tonnes"].sum()
        positive = edge_year["quantity_tonnes"].gt(0)
        rows.append(
            {
                "crop": crop,
                "direction": direction,
                "year_start": int(group["year"].min()),
                "year_end": int(group["year"].max()),
                "reporter_countries": int(group["Reporter Country Code"].nunique()),
                "partner_countries": int(group["Partner Country Code"].nunique()),
                "bilateral_edges": int(group[edge_keys].drop_duplicates().shape[0]),
                "positive_edge_years": int(positive.sum()),
                "edge_year_coverage_rate": float(positive.mean()),
                "total_trade_tonnes": float(group["quantity_tonnes"].sum()),
            }
        )
    result = pd.DataFrame(rows)
    save_csv(result, TABLE_DIR / "table1_data_summary.csv")
    return result
