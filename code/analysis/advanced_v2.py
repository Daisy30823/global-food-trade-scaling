from __future__ import annotations

import hashlib
import platform
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.covariance import LedoitWolf

from .analysis_runner import stable_seed
from .config import (
    BLOCK_LENGTH,
    BLOCK_LENGTH_SENSITIVITY,
    CROPS,
    DATA_DIR,
    ELEMENTS,
    FIG_DATA_DIR,
    MAIN_MEMBER,
    N_BOOT,
    ROOT,
    SEED,
    TABLE_DIR,
    YEAR_COLS,
    YEARS,
    ZIP_PATH,
)
from .statistics import TradePanel, detrend_residuals
from .utils import get_logger, save_csv, write_json

LOGGER = get_logger()
TINY = np.finfo(float).tiny


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash() -> str:
    files = [ROOT / "run_all.py", ROOT / "requirements.txt"]
    files += sorted((ROOT / "analysis").glob("*.py"))
    files += sorted((ROOT / "tests").glob("*.py"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_signature() -> dict:
    return {
        "raw_zip_sha256": sha256_file(ZIP_PATH),
        "code_sha256": code_hash(),
        "config_sha256": sha256_file(ROOT / "analysis" / "config.py"),
        "seed": SEED,
    }


def cache_is_valid(metadata_path: Path, signature: dict) -> bool:
    if not metadata_path.exists():
        return False
    try:
        cached = pd.read_json(metadata_path, typ="series").to_dict()
    except Exception:
        return False
    return all(cached.get(key) == value for key, value in signature.items())


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    denom = float(xc @ xc)
    return float(xc @ (y - y.mean()) / denom) if denom > 0 else np.nan


def _hc3(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    fit = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(x, float))).fit().get_robustcov_results(cov_type="HC3")
    ci = fit.conf_int()[1]
    return float(fit.params[1]), float(ci[0]), float(ci[1])


def _percentile(values: np.ndarray) -> tuple[float, float]:
    clean = np.asarray(values, float)
    clean = clean[np.isfinite(clean)]
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def _component_slopes(metrics: pd.DataFrame) -> dict[str, float]:
    x = np.log(metrics["mu"].to_numpy(float))
    values = {
        "alpha_V": np.log(metrics["variance"].to_numpy(float)),
        "alpha_D": np.log(metrics["D"].to_numpy(float)),
        "beta_phi": np.log(metrics["phi"].to_numpy(float)),
    }
    result = {name: _slope(x, y) for name, y in values.items()}
    error = abs(result["alpha_V"] - result["alpha_D"] - result["beta_phi"])
    if error >= 1e-8:
        raise AssertionError(f"alpha_V=alpha_D+beta_phi failed: {error}")
    result["identity_error"] = error
    return result


def _country_boot_components(metrics: pd.DataFrame, n_boot: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.log(metrics["mu"].to_numpy(float))
    ys = {
        "alpha_V": np.log(metrics["variance"].to_numpy(float)),
        "alpha_D": np.log(metrics["D"].to_numpy(float)),
        "beta_phi": np.log(metrics["phi"].to_numpy(float)),
    }
    draws = {key: np.empty(n_boot) for key in ys}
    for b in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        for key, y in ys.items():
            draws[key][b] = _slope(x[idx], y[idx])
    if np.nanmax(np.abs(draws["alpha_V"] - draws["alpha_D"] - draws["beta_phi"])) >= 1e-8:
        raise AssertionError("Bootstrap component identity failed")
    return draws


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, blocks)
    return np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]


def _resampled_country_metrics(panel: TradePanel, sample_idx: np.ndarray) -> tuple[float, float, float, float]:
    values = panel.values[:, sample_idx]
    mu = float(values.sum(axis=0).mean())
    residual = detrend_residuals(values, degree=1)
    total_residual = residual.sum(axis=0)
    variance = float(np.var(total_residual, ddof=1))
    diagonal = float(np.var(residual, axis=1, ddof=1).sum())
    phi = variance / diagonal if diagonal > 0 else np.nan
    return mu, variance, diagonal, phi


def block_bootstrap_components(
    panels: dict[tuple[str, str, int], TradePanel],
    metrics: pd.DataFrame,
    crop: str,
    direction: str,
    block_length: int,
    n_boot: int,
    two_stage: bool,
) -> dict[str, np.ndarray]:
    eligible = metrics[
        (metrics["crop"] == crop)
        & (metrics["direction"] == direction)
        & metrics["eligible_covariance"]
        & metrics["phi"].gt(0)
    ].sort_values("country_code")
    codes = eligible["country_code"].astype(int).to_numpy()
    rng = np.random.default_rng(stable_seed(f"block-{block_length}-{two_stage}-{crop}-{direction}"))
    draws = {name: np.full(n_boot, np.nan) for name in ["alpha_V", "alpha_D", "beta_phi"]}
    for b in range(n_boot):
        year_idx = moving_block_indices(len(YEARS), block_length, rng)
        rows = np.array([_resampled_country_metrics(panels[(crop, direction, int(code))], year_idx) for code in codes])
        valid = np.all(np.isfinite(rows), axis=1) & np.all(rows > 0, axis=1)
        rows = rows[valid]
        if len(rows) < 4:
            continue
        if two_stage:
            country_idx = rng.integers(0, len(rows), len(rows))
            rows = rows[country_idx]
        x = np.log(rows[:, 0])
        draws["alpha_V"][b] = _slope(x, np.log(rows[:, 1]))
        draws["alpha_D"][b] = _slope(x, np.log(rows[:, 2]))
        draws["beta_phi"][b] = _slope(x, np.log(rows[:, 3]))
    max_error = np.nanmax(np.abs(draws["alpha_V"] - draws["alpha_D"] - draws["beta_phi"]))
    if max_error >= 1e-8:
        raise AssertionError(f"Year bootstrap component identity failed: {max_error}")
    return draws


def run_exact_decomposition(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame, n_boot: int = N_BOOT
) -> dict:
    rows: list[dict] = []
    draw_frames: list[pd.DataFrame] = []
    for (crop, direction), group in full_metrics.groupby(["crop", "direction"], sort=False):
        eligible = group[group["eligible_covariance"] & group["phi"].gt(0)].copy().sort_values("country_code")
        point = _component_slopes(eligible)
        country = _country_boot_components(eligible, n_boot, stable_seed(f"component-country-{crop}-{direction}"))
        block_results: dict[tuple[int, bool], dict[str, np.ndarray]] = {}
        for length in (BLOCK_LENGTH, *BLOCK_LENGTH_SENSITIVITY):
            block_results[(length, False)] = block_bootstrap_components(
                panels, full_metrics, crop, direction, length, n_boot, False
            )
        block_results[(BLOCK_LENGTH, True)] = block_bootstrap_components(
            panels, full_metrics, crop, direction, BLOCK_LENGTH, n_boot, True
        )
        x = np.log(eligible["mu"].to_numpy(float))
        outcomes = {
            "alpha_V": np.log(eligible["variance"].to_numpy(float)),
            "alpha_D": np.log(eligible["D"].to_numpy(float)),
            "beta_phi": np.log(eligible["phi"].to_numpy(float)),
        }
        for component, y in outcomes.items():
            estimate, hc3_low, hc3_high = _hc3(x, y)
            country_low, country_high = _percentile(country[component])
            year_low, year_high = _percentile(block_results[(BLOCK_LENGTH, False)][component])
            two_low, two_high = _percentile(block_results[(BLOCK_LENGTH, True)][component])
            row = {
                "crop": crop,
                "direction": direction,
                "component": component,
                "estimate": estimate,
                "hc3_ci_low": hc3_low,
                "hc3_ci_high": hc3_high,
                "country_boot_ci_low": country_low,
                "country_boot_ci_high": country_high,
                "year_block_ci_low": year_low,
                "year_block_ci_high": year_high,
                "two_stage_ci_low": two_low,
                "two_stage_ci_high": two_high,
                "n_reporting_economies": len(eligible),
                "identity_error": point["identity_error"],
            }
            for length in BLOCK_LENGTH_SENSITIVITY:
                low, high = _percentile(block_results[(length, False)][component])
                row[f"year_block_L{length}_ci_low"] = low
                row[f"year_block_L{length}_ci_high"] = high
            rows.append(row)
            draw_frames.append(
                pd.DataFrame(
                    {
                        "crop": crop,
                        "direction": direction,
                        "component": component,
                        "draw": np.arange(n_boot),
                        "country_boot": country[component],
                        "year_block_L4": block_results[(BLOCK_LENGTH, False)][component],
                        "two_stage_L4": block_results[(BLOCK_LENGTH, True)][component],
                    }
                )
            )
    summary = pd.DataFrame(rows)
    draws = pd.concat(draw_frames, ignore_index=True)
    save_csv(summary, TABLE_DIR / "table_v2_exact_decomposition.csv")
    draws.to_parquet(DATA_DIR / "exact_decomposition_bootstrap_draws.parquet", index=False)
    return {"summary": summary, "draws": draws}


def run_paired_direction(full_metrics: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    alpha_rows: list[dict] = []
    phi_rows: list[dict] = []
    draw_rows: list[pd.DataFrame] = []
    membership_rows: list[dict] = []
    for crop, crop_group in full_metrics.groupby("crop", sort=False):
        imp = crop_group[(crop_group["direction"] == "Import") & crop_group["eligible_taylor"]].set_index("country_code")
        exp = crop_group[(crop_group["direction"] == "Export") & crop_group["eligible_taylor"]].set_index("country_code")
        common = np.array(sorted(set(imp.index) & set(exp.index)), dtype=int)
        imp_a, exp_a = imp.loc[common], exp.loc[common]
        membership_rows.extend({"crop": crop, "analysis": "alpha", "country_code": int(code), "in_import": True, "in_export": True} for code in common)
        point_imp = _slope(np.log(imp_a["mu"]), np.log(imp_a["variance"]))
        point_exp = _slope(np.log(exp_a["mu"]), np.log(exp_a["variance"]))
        rng = np.random.default_rng(stable_seed(f"paired-alpha-{crop}"))
        draws = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(common), len(common))
            draws[b] = _slope(np.log(imp_a["mu"].to_numpy()[idx]), np.log(imp_a["variance"].to_numpy()[idx])) - _slope(
                np.log(exp_a["mu"].to_numpy()[idx]), np.log(exp_a["variance"].to_numpy()[idx])
            )
        low, high = _percentile(draws)
        alpha_rows.append(
            {
                "crop": crop,
                "n_pairs": len(common),
                "alpha_import": point_imp,
                "alpha_export": point_exp,
                "estimate_import_minus_export": point_imp - point_exp,
                "ci_low": low,
                "ci_high": high,
            }
        )
        draw_rows.append(pd.DataFrame({"crop": crop, "draw": np.arange(n_boot), "alpha_import_minus_export": draws}))

        imp_p = crop_group[(crop_group["direction"] == "Import") & crop_group["eligible_covariance"] & crop_group["phi"].gt(0)].set_index("country_code")
        exp_p = crop_group[(crop_group["direction"] == "Export") & crop_group["eligible_covariance"] & crop_group["phi"].gt(0)].set_index("country_code")
        common_phi = np.array(sorted(set(imp_p.index) & set(exp_p.index)), dtype=int)
        membership_rows.extend({"crop": crop, "analysis": "phi", "country_code": int(code), "in_import": True, "in_export": True} for code in common_phi)
        diff = exp_p.loc[common_phi, "phi"].to_numpy(float) - imp_p.loc[common_phi, "phi"].to_numpy(float)
        rng = np.random.default_rng(stable_seed(f"paired-phi-{crop}"))
        med_draw = np.empty(n_boot)
        mean_draw = np.empty(n_boot)
        share_draw = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, len(diff), len(diff))
            sample = diff[idx]
            med_draw[b] = np.median(sample)
            mean_draw[b] = np.mean(sample)
            share_draw[b] = np.mean(sample > 0)
        med_low, med_high = _percentile(med_draw)
        mean_low, mean_high = _percentile(mean_draw)
        share_low, share_high = _percentile(share_draw)
        lower_tail = (np.sum(med_draw <= 0) + 1) / (n_boot + 1)
        upper_tail = (np.sum(med_draw >= 0) + 1) / (n_boot + 1)
        phi_rows.append(
            {
                "crop": crop,
                "n_pairs": len(common_phi),
                "median_difference": float(np.median(diff)),
                "median_ci_low": med_low,
                "median_ci_high": med_high,
                "mean_difference": float(np.mean(diff)),
                "mean_ci_low": mean_low,
                "mean_ci_high": mean_high,
                "share_export_gt_import": float(np.mean(diff > 0)),
                "share_ci_low": share_low,
                "share_ci_high": share_high,
                "empirical_p_median": float(min(1.0, 2 * min(lower_tail, upper_tail))),
            }
        )
    alpha = pd.DataFrame(alpha_rows)
    phi = pd.DataFrame(phi_rows)
    draws_df = pd.concat(draw_rows, ignore_index=True)
    save_csv(alpha, TABLE_DIR / "table_v2_paired_alpha.csv")
    save_csv(phi, TABLE_DIR / "table_v2_paired_phi.csv")
    save_csv(draws_df, DATA_DIR / "paired_alpha_bootstrap_draws.csv")
    save_csv(pd.DataFrame(membership_rows), DATA_DIR / "paired_country_membership.csv")
    return {"alpha": alpha, "phi": phi, "draws": draws_df}


def run_rolling_block_alpha(
    panels: dict[tuple[str, str, int], TradePanel], metric_cache: dict[str, pd.DataFrame], n_boot: int = N_BOOT
) -> pd.DataFrame:
    rows = []
    for window_id, metrics in metric_cache.items():
        if not window_id.startswith("roll20_"):
            continue
        start_year = int(metrics["start_year"].iloc[0])
        end_year = int(metrics["end_year"].iloc[0])
        base_idx = np.arange(YEARS.index(start_year), YEARS.index(end_year) + 1)
        for (crop, direction), group in metrics.groupby(["crop", "direction"], sort=False):
            eligible = group[group["eligible_taylor"]].sort_values("country_code")
            codes = eligible["country_code"].astype(int).to_numpy()
            totals = np.vstack([panels[(crop, direction, int(code))].values.sum(axis=0)[base_idx] for code in codes])
            rng = np.random.default_rng(stable_seed(f"rolling-block-{window_id}-{crop}-{direction}"))
            draws = np.empty(n_boot)
            for b in range(n_boot):
                sample_idx = moving_block_indices(len(base_idx), BLOCK_LENGTH, rng)
                sampled = totals[:, sample_idx]
                mu = sampled.mean(axis=1)
                residual = detrend_residuals(sampled, degree=1)
                variance = np.var(residual, axis=1, ddof=1)
                valid = (mu > 0) & (variance > 0)
                draws[b] = _slope(np.log(mu[valid]), np.log(variance[valid]))
            point = _slope(np.log(eligible["mu"]), np.log(eligible["variance"]))
            low, high = _percentile(draws)
            rows.append(
                {
                    "window_id": window_id,
                    "window_label": str(metrics["window_label"].iloc[0]),
                    "start_year": start_year,
                    "end_year": end_year,
                    "crop": crop,
                    "direction": direction,
                    "estimate": point,
                    "block_boot_ci_low": low,
                    "block_boot_ci_high": high,
                    "block_length": BLOCK_LENGTH,
                    "n_reporting_economies": len(eligible),
                    "n_draws": n_boot,
                }
            )
    result = pd.DataFrame(rows)
    save_csv(result, FIG_DATA_DIR / "Fig2_rolling20_block_bootstrap.csv")
    return result


def source_shock_minus_importer(export_panel: TradePanel, importer_code: int) -> np.ndarray:
    total = export_panel.values.sum(axis=0)
    match = np.flatnonzero(export_panel.partner_codes == importer_code)
    if len(match):
        total = total - export_panel.values[match].sum(axis=0)
    return total


def _correlation(residual: np.ndarray, method: str) -> np.ndarray:
    if method == "ledoit_wolf":
        cov = LedoitWolf().fit(residual.T).covariance_
        scale = np.sqrt(np.maximum(np.diag(cov), TINY))
        corr = cov / np.outer(scale, scale)
    elif method == "sample":
        corr = np.corrcoef(residual)
    else:
        raise ValueError(method)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1, 1)


def compute_source_diversification(
    panels: dict[tuple[str, str, int], TradePanel],
    full_metrics: pd.DataFrame,
    share_threshold: float,
    coverage_threshold: float,
    matrix_method: str,
) -> pd.DataFrame:
    eligible = set(
        map(tuple, full_metrics[(full_metrics["direction"] == "Import") & full_metrics["eligible_covariance"]][["crop", "country_code"]].to_numpy())
    )
    rows: list[dict] = []
    for (crop, direction, importer_code), import_panel in panels.items():
        if direction != "Import":
            continue
        totals = import_panel.values.sum(axis=1)
        grand = float(totals.sum())
        if grand <= 0:
            continue
        shares = totals / grand
        positive_years = np.sum(import_panel.values > 0, axis=1)
        base_selected = (shares >= share_threshold) & (positive_years >= 5)
        shocks, kept_shares, sources = [], [], []
        for source_code, share, selected in zip(import_panel.partner_codes, shares, base_selected):
            if not selected:
                continue
            export_panel = panels.get((crop, "Export", int(source_code)))
            if export_panel is None:
                continue
            shock = source_shock_minus_importer(export_panel, int(importer_code))
            residual = detrend_residuals(shock, degree=1)
            if np.var(residual, ddof=1) <= 0:
                continue
            shocks.append(residual)
            kept_shares.append(float(share))
            sources.append(int(source_code))
        coverage = float(np.sum(kept_shares))
        n_corr = nominal = gap = min_eigenvalue = np.nan
        if coverage >= coverage_threshold and len(shocks) >= 2:
            s = np.asarray(kept_shares) / coverage
            residuals = np.vstack(shocks)
            corr = _correlation(residuals, matrix_method)
            min_eigenvalue = float(np.linalg.eigvalsh(corr).min())
            denom = float(s @ corr @ s)
            if denom > 0:
                n_corr = 1 / denom
                nominal = 1 / float(s @ s)
                gap = float(np.log(nominal) - np.log(n_corr))
        rows.append(
            {
                "crop": crop,
                "country_code": int(importer_code),
                "reporting_economy": import_panel.country_name,
                "main_eligible": (crop, int(importer_code)) in eligible,
                "share_threshold": share_threshold,
                "coverage_threshold": coverage_threshold,
                "matrix_method": matrix_method,
                "selected_sources": len(shocks),
                "selected_source_codes": ";".join(map(str, sources)),
                "selected_coverage": coverage,
                "n_years": len(YEARS),
                "min_eigenvalue": min_eigenvalue,
                "positive_definite": bool(np.isfinite(min_eigenvalue) and min_eigenvalue > -1e-8),
                "N_nominal_matched": nominal,
                "N_corr_source": n_corr,
                "Gap_source": gap,
            }
        )
    return pd.DataFrame(rows)


def run_source_diversification(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame, n_boot: int = N_BOOT
) -> dict:
    frames = []
    for share in (0.005, 0.01, 0.02):
        for coverage in (0.70, 0.80, 0.90):
            for method in ("ledoit_wolf", "sample"):
                frames.append(compute_source_diversification(panels, full_metrics, share, coverage, method))
    all_results = pd.concat(frames, ignore_index=True)
    main = all_results[
        (all_results["share_threshold"] == 0.01)
        & (all_results["coverage_threshold"] == 0.80)
        & (all_results["matrix_method"] == "ledoit_wolf")
    ].copy()
    summary_rows = []
    for keys, group in all_results.groupby(["share_threshold", "coverage_threshold", "matrix_method"]):
        eligible = group[group["main_eligible"]].dropna(subset=["Gap_source"])
        all_computable = group.dropna(subset=["Gap_source"])
        rng = np.random.default_rng(stable_seed(f"div-{keys}"))
        values = eligible["Gap_source"].to_numpy(float)
        draws = np.array([np.median(rng.choice(values, len(values), replace=True)) for _ in range(n_boot)]) if len(values) else np.array([np.nan])
        low, high = _percentile(draws) if np.isfinite(draws).any() else (np.nan, np.nan)
        summary_rows.append(
            {
                "share_threshold": keys[0],
                "coverage_threshold": keys[1],
                "matrix_method": keys[2],
                "n_main_eligible": len(eligible),
                "n_all_computable": len(all_computable),
                "median_gap_main_eligible": float(np.median(values)) if len(values) else np.nan,
                "median_gap_ci_low": low,
                "median_gap_ci_high": high,
                "share_gap_positive_main_eligible": float(np.mean(values > 0)) if len(values) else np.nan,
                "median_gap_all_computable": float(all_computable["Gap_source"].median()) if len(all_computable) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    stable = bool(len(summary) == 18 and summary["median_gap_ci_low"].gt(0).all())
    all_results.to_parquet(DATA_DIR / "source_diversification_all_sensitivities.parquet", index=False)
    save_csv(main, FIG_DATA_DIR / "FigS_source_diversification_main.csv")
    save_csv(summary, TABLE_DIR / "table_v2_source_diversification_sensitivity.csv")
    write_json({"stable_signal": stable, "criterion": "lower 95% bootstrap CI for median Gap_source > 0 in all 18 prespecified configurations"}, DATA_DIR / "source_diversification_decision.json")
    return {"all": all_results, "main": main, "summary": summary, "stable": stable}


# This explicit source-label rule is deliberately conservative. It is a
# sensitivity sample, not a claim that the supplied FAOSTAT geography lookup is
# a legal sovereignty classification.
NON_SOVEREIGN_LABELS = {
    "American Samoa", "Anguilla", "Aruba", "Bermuda", "British Virgin Islands",
    "Cayman Islands", "China, Hong Kong SAR", "China, Macao SAR", "China, Taiwan Province of",
    "Cook Islands", "Falkland Islands (Malvinas)", "Faroe Islands", "French Guiana",
    "French Polynesia", "Gibraltar", "Greenland", "Guadeloupe", "Guam", "Isle of Man",
    "Martinique", "Montserrat", "New Caledonia", "Niue", "Northern Mariana Islands",
    "Puerto Rico", "Reunion", "Saint Helena", "Tokelau", "Turks and Caicos Islands",
    "United States Virgin Islands", "Wallis and Futuna Islands", "Western Sahara",
}


def run_sovereign_sensitivity(full_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_name, data in [
        ("all_reporting_economies_areas", full_metrics),
        ("restricted_sovereign_state_sensitivity", full_metrics[~full_metrics["country"].isin(NON_SOVEREIGN_LABELS)]),
    ]:
        for (crop, direction), group in data.groupby(["crop", "direction"], sort=False):
            taylor = group[group["eligible_taylor"] & group["mu"].gt(0) & group["variance"].gt(0)]
            covariance = group[group["eligible_covariance"] & group["phi"].gt(0)]
            rows.append(
                {
                    "sample": sample_name,
                    "crop": crop,
                    "direction": direction,
                    "alpha_V": _slope(np.log(taylor["mu"]), np.log(taylor["variance"])),
                    "beta_phi": _slope(np.log(covariance["mu"]), np.log(covariance["phi"])),
                    "median_phi": float(covariance["phi"].median()),
                    "n_taylor": len(taylor),
                    "n_covariance": len(covariance),
                }
            )
    result = pd.DataFrame(rows)
    save_csv(result, TABLE_DIR / "table_v2_sovereign_sensitivity.csv")
    return result


def build_reporting_activity(chunksize: int = 40_000) -> pd.DataFrame:
    output = DATA_DIR / "reporter_year_reporting_activity.parquet"
    grouped_chunks = []
    usecols = ["Reporter Country Code", "Reporter Countries", "Item Code", "Element Code", "Unit", *YEAR_COLS]
    with zipfile.ZipFile(ZIP_PATH) as archive, archive.open(MAIN_MEMBER) as handle:
        for chunk in pd.read_csv(handle, usecols=usecols, chunksize=chunksize, low_memory=False):
            keep = (
                chunk["Item Code"].isin(CROPS)
                & chunk["Element Code"].isin(ELEMENTS)
                & chunk["Unit"].eq("t")
                & chunk["Reporter Country Code"].lt(5000)
            )
            selected = chunk.loc[keep]
            if selected.empty:
                continue
            flags = selected[YEAR_COLS].notna()
            keys = selected[["Reporter Country Code", "Reporter Countries", "Item Code", "Element Code"]].reset_index(drop=True)
            grouped_chunks.append(pd.concat([keys, flags.reset_index(drop=True)], axis=1).groupby(list(keys.columns), as_index=False)[YEAR_COLS].max())
    wide = pd.concat(grouped_chunks, ignore_index=True).groupby(
        ["Reporter Country Code", "Reporter Countries", "Item Code", "Element Code"], as_index=False
    )[YEAR_COLS].max()
    long = wide.melt(
        id_vars=["Reporter Country Code", "Reporter Countries", "Item Code", "Element Code"],
        value_vars=YEAR_COLS,
        var_name="year",
        value_name="reporter_active",
    )
    long["year"] = long["year"].str[1:].astype(int)
    long["crop"] = long["Item Code"].map(CROPS)
    long["direction"] = long["Element Code"].map(ELEMENTS)
    long.to_parquet(output, index=False)
    save_csv(
        long.groupby(["crop", "direction", "year"], as_index=False)["reporter_active"].agg(active_reporting_economies="sum", observed_reporting_economies="size"),
        TABLE_DIR / "table_v2_reporting_activity.csv",
    )
    return long


def run_reporting_aware_sensitivity(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame, activity: pd.DataFrame
) -> dict:
    activity_map = {}
    for (crop, direction, code), group in activity.groupby(["crop", "direction", "Reporter Country Code"], sort=False):
        activity_map[(crop, direction, int(code))] = group.set_index("year")["reporter_active"].reindex(YEARS, fill_value=False).to_numpy(bool)
    rows = []
    for key, panel in panels.items():
        active = activity_map.get(key, np.zeros(len(YEARS), dtype=bool))
        idx = np.flatnonzero(active)
        if len(idx) < 20:
            continue
        values = panel.values[:, idx]
        residual = detrend_residuals(values, degree=1)
        total = values.sum(axis=0)
        mu = float(total.mean())
        variance = float(np.var(residual.sum(axis=0), ddof=1))
        diagonal = float(np.var(residual, axis=1, ddof=1).sum())
        phi = variance / diagonal if diagonal > 0 else np.nan
        rows.append(
            {
                "crop": panel.crop,
                "direction": panel.direction,
                "country_code": panel.country_code,
                "country": panel.country_name,
                "covered_years": len(idx),
                "noncoverage_years": len(YEARS) - len(idx),
                "mu": mu,
                "variance": variance,
                "D": diagonal,
                "phi": phi,
                "eligible_taylor": bool(mu > 0 and variance > 0 and np.sum(total > 0) >= 20),
                "eligible_covariance": bool(mu > 0 and variance > 0 and diagonal > 0 and phi > 0 and np.sum(total > 0) >= 20 and np.sum(np.any(values > 0, axis=1)) >= 2),
            }
        )
    aware = pd.DataFrame(rows)
    group_rows = []
    for spec, data in [("blank_as_zero", full_metrics), ("reporting_aware_complete_observations", aware)]:
        for (crop, direction), group in data.groupby(["crop", "direction"], sort=False):
            t = group[group["eligible_taylor"]]
            c = group[group["eligible_covariance"] & group["phi"].gt(0)]
            group_rows.append(
                {
                    "specification": spec,
                    "crop": crop,
                    "direction": direction,
                    "alpha_V": _slope(np.log(t["mu"]), np.log(t["variance"])),
                    "beta_phi": _slope(np.log(c["mu"]), np.log(c["phi"])),
                    "median_phi": float(c["phi"].median()),
                    "n_taylor": len(t),
                    "n_covariance": len(c),
                }
            )
    comparison = pd.DataFrame(group_rows)
    overlap = full_metrics.merge(aware, on=["crop", "direction", "country_code", "country"], suffixes=("_zero", "_aware"))
    overlap["abs_phi_change"] = (overlap["phi_aware"] - overlap["phi_zero"]).abs()
    affected = overlap.sort_values(["noncoverage_years", "abs_phi_change"], ascending=False).head(50)
    aware.to_parquet(DATA_DIR / "country_metrics_reporting_aware.parquet", index=False)
    save_csv(comparison, TABLE_DIR / "table_v2_reporting_aware_sensitivity.csv")
    save_csv(affected, TABLE_DIR / "table_v2_most_affected_reporting_economies.csv")
    write_json(
        {
            "blank_as_zero_audited": True,
            "reporting_aware_audited": True,
            "explicit_zero_is_reporting_activity": True,
            "noncoverage_definition": "all partner cells blank for reporter-crop-direction-year",
            "handling": "use only covered reporter-years; no interpolation",
        },
        DATA_DIR / "missingness_audit.json",
    )
    return {"metrics": aware, "comparison": comparison, "affected": affected}


def environment_manifest(signature: dict) -> dict:
    import matplotlib
    import pyarrow
    import scipy
    import sklearn
    import statsmodels

    return {
        **signature,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements": (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines(),
        "resolved_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def _block_permutation_variances(
    residuals: np.ndarray, n_iter: int, rng: np.random.Generator, block_length: int, batch_size: int = 25
) -> np.ndarray:
    p, t = residuals.shape
    n_full = t // block_length
    full_length = n_full * block_length
    blocks = np.arange(full_length).reshape(n_full, block_length)
    tail = np.arange(full_length, t)
    out = np.empty(n_iter)
    for start in range(0, n_iter, batch_size):
        stop = min(n_iter, start + batch_size)
        b = stop - start
        order = np.argsort(rng.random((b, p, n_full)), axis=2)
        indices = blocks[order].reshape(b, p, full_length)
        if len(tail):
            indices = np.concatenate([indices, np.broadcast_to(tail, (b, p, len(tail)))], axis=2)
        expanded = np.broadcast_to(residuals, (b, p, t))
        permuted = np.take_along_axis(expanded, indices, axis=2)
        out[start:stop] = np.var(permuted.sum(axis=1), axis=1, ddof=1)
    return out


def run_block_permutation_null(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame, n_iter: int = N_BOOT
) -> dict:
    summaries = []
    saved_draws = []
    for block_length in (3, 4, 5):
        for (crop, direction), observed_group in full_metrics.groupby(["crop", "direction"], sort=False):
            eligible = observed_group[observed_group["eligible_taylor"]].copy().sort_values("country_code")
            codes = eligible["country_code"].astype(int).to_numpy()
            null_v = np.full((n_iter, len(codes)), np.nan)
            d_values = np.full(len(codes), np.nan)
            rng = np.random.default_rng(stable_seed(f"block-null-{block_length}-{crop}-{direction}"))
            for col, code in enumerate(codes):
                residual = detrend_residuals(panels[(crop, direction, int(code))].values, degree=1)
                residual = residual[np.var(residual, axis=1, ddof=1) > 0]
                if residual.size:
                    d_values[col] = float(np.var(residual, axis=1, ddof=1).sum())
                    null_v[:, col] = _block_permutation_variances(residual, n_iter, rng, block_length)
            mu = eligible["mu"].to_numpy(float)
            valid = np.isfinite(d_values) & (d_values > 0) & (mu > 0)
            phi = null_v[:, valid] / d_values[valid]
            x = np.log(mu[valid])
            alpha = np.array([_slope(x, np.log(np.maximum(row, TINY))) for row in null_v[:, valid]])
            observed = {
                "median_phi": float(eligible.iloc[np.flatnonzero(valid)]["phi"].median()),
                "alpha": _slope(x, np.log(eligible.iloc[np.flatnonzero(valid)]["variance"])),
            }
            null_metrics = {"median_phi": np.nanmedian(phi, axis=1), "alpha": alpha}
            for metric, values in null_metrics.items():
                center = float(np.mean(values))
                obs = observed[metric]
                p_value = float((np.sum(np.abs(values - center) >= abs(obs - center)) + 1) / (n_iter + 1))
                summaries.append(
                    {
                        "null_model": f"block_permutation_L{block_length}",
                        "block_length": block_length,
                        "crop": crop,
                        "direction": direction,
                        "metric": metric,
                        "observed": obs,
                        "null_mean": center,
                        "null_median": float(np.median(values)),
                        "null_q025": float(np.quantile(values, 0.025)),
                        "null_q975": float(np.quantile(values, 0.975)),
                        "absolute_difference": float(obs - center),
                        "ratio": float(obs / center) if center != 0 else np.nan,
                        "empirical_p_value": p_value,
                        "n_iterations": n_iter,
                        "n_reporting_economies": int(valid.sum()),
                    }
                )
                if block_length == BLOCK_LENGTH:
                    saved_draws.append(pd.DataFrame({"crop": crop, "direction": direction, "metric": metric, "draw": np.arange(n_iter), "value": values}))
        LOGGER.info("Completed block-permutation null with block length %d", block_length)
    summary = pd.DataFrame(summaries)
    draws = pd.concat(saved_draws, ignore_index=True)
    save_csv(summary, TABLE_DIR / "table_v2_block_permutation_null.csv")
    draws.to_parquet(DATA_DIR / "block_permutation_null_L4_draws.parquet", index=False)
    return {"summary": summary, "draws": draws}
