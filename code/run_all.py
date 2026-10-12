from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from analysis.advanced_v2 import (
    build_reporting_activity,
    cache_is_valid,
    environment_manifest,
    run_block_permutation_null,
    run_exact_decomposition,
    run_paired_direction,
    run_reporting_aware_sensitivity,
    run_rolling_block_alpha,
    run_signature,
    run_source_diversification,
    run_sovereign_sensitivity,
)
from analysis.analysis_runner import (
    edge_mean_variance_data,
    run_null_model,
    run_null_window_robustness,
    run_rice_code31_sensitivity,
    run_taylor_and_models,
)
from analysis.config import DATA_DIR, N_BOOT, OUT, REPORT_DIR, ROOT, SEED, ensure_directories
from analysis.data_prep import build_table1, inspect_archive, prepare_core_data
from analysis.figures_v2 import create_all_figures_v2, write_figure5_pooling_audit
from analysis.reporting_v2 import write_reports_v2
from analysis.statistics import build_panels
from analysis.utils import get_logger, save_csv, write_json


def _aggregation_audit(core: pd.DataFrame, panels: dict) -> dict:
    grouped = core.groupby(["crop", "direction", "Reporter Country Code", "year"], as_index=False)["quantity_tonnes"].sum()
    check_rows = []
    for (crop, direction, code), panel in panels.items():
        for year, value in zip(range(1986, 2025), panel.values.sum(axis=0)):
            check_rows.append((crop, direction, code, year, float(value)))
    independent = pd.DataFrame(check_rows, columns=["crop", "direction", "Reporter Country Code", "year", "panel_total"])
    merged = grouped.merge(independent, on=["crop", "direction", "Reporter Country Code", "year"], how="outer")
    raw_total = merged["quantity_tonnes"].fillna(0).to_numpy(float)
    panel_total = merged["panel_total"].fillna(0).to_numpy(float)
    error = np.abs(raw_total - panel_total)
    relative_error = error / np.maximum(1.0, np.abs(raw_total))
    passed = bool(np.allclose(raw_total, panel_total, rtol=1e-12, atol=1e-8))
    result = {
        "rows_compared": len(merged),
        "max_absolute_error_tonnes": float(error.max()),
        "max_scaled_relative_error": float(relative_error.max()),
        "tolerance": {"relative": 1e-12, "absolute_tonnes": 1e-8},
        "passed": passed,
        "method": "raw filtered long-table groupby versus independent TradePanel summation; tolerance permits only floating-point summation-order error",
    }
    write_json(result, DATA_DIR / "raw_to_country_aggregation_audit.json")
    return result


def _portable_inventory() -> list[dict]:
    rows = []
    for base, label in [(OUT, "outputs_v2"), (REPORT_DIR, "report_v2")]:
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rows.append({"path": f"{label}/{path.relative_to(base).as_posix()}", "bytes": path.stat().st_size})
    return rows


def main() -> None:
    ensure_directories()
    logger = get_logger()
    np.random.seed(SEED)
    started = datetime.now(timezone.utc)
    signature = run_signature()
    cache_metadata = DATA_DIR / "cache_signature.json"
    cache_valid = cache_is_valid(cache_metadata, signature)
    logger.info("Starting v2 pipeline with seed %d and %d formal draws", SEED, N_BOOT)
    logger.info("Hash-validated cache available: %s", cache_valid)

    inspection = inspect_archive()
    prep = prepare_core_data(force=not cache_valid)
    write_json(signature, cache_metadata)
    core = pd.read_parquet(DATA_DIR / "core_four_crops_long.parquet")
    table1 = build_table1(core)
    panels = build_panels(core)
    aggregation_audit = _aggregation_audit(core, panels)
    if not aggregation_audit["passed"]:
        raise AssertionError("Independent raw-to-country aggregation audit failed")
    logger.info("Built %s reporting-economy crop-direction panels", f"{len(panels):,}")

    results = run_taylor_and_models(panels, n_boot=N_BOOT)
    rice31 = pd.read_parquet(DATA_DIR / "rice_code31_long.parquet")
    run_rice_code31_sensitivity(rice31, results["alpha"], n_boot=N_BOOT)
    decomposition = run_exact_decomposition(panels, results["full_metrics"], n_boot=N_BOOT)
    paired = run_paired_direction(results["full_metrics"], n_boot=N_BOOT)
    rolling = run_rolling_block_alpha(panels, results["metric_cache"], n_boot=N_BOOT)
    diversification = run_source_diversification(panels, results["full_metrics"], n_boot=N_BOOT)
    circular_null = run_null_model(panels, results["full_metrics"], n_iter=N_BOOT)
    all_window_null = run_null_window_robustness(
        panels, results["metric_cache"], circular_null["summary"], n_iter=N_BOOT, force=True
    )
    block_null = run_block_permutation_null(panels, results["full_metrics"], n_iter=N_BOOT)
    activity = build_reporting_activity()
    reporting = run_reporting_aware_sensitivity(panels, results["full_metrics"], activity)
    sovereign = run_sovereign_sensitivity(results["full_metrics"])
    edge_mean_variance_data(panels)
    write_figure5_pooling_audit()

    logger.info("Running mandatory v2 automated tests")
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    (DATA_DIR / "pytest_output.txt").write_text(test_result.stdout + test_result.stderr, encoding="utf-8")
    if test_result.returncode != 0:
        logger.error("Tests failed; formal figures and reports were not generated\n%s", test_result.stdout + test_result.stderr)
        raise SystemExit(test_result.returncode)
    match = re.search(r"(\d+) passed", test_result.stdout)
    tests_passed = int(match.group(1)) if match else 0
    logger.info("All %d automated tests passed", tests_passed)

    figures = create_all_figures_v2(
        results["full_fit"], results["full_metrics"], decomposition, rolling, paired,
        circular_null, all_window_null, diversification, reporting, sovereign,
    )
    summary = write_reports_v2(
        prep, results, decomposition, paired, rolling, diversification, circular_null,
        block_null, reporting, sovereign, figures, tests_passed,
    )
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "2.0",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "signature": signature,
        "environment": environment_manifest(signature),
        "source_archive": inspection["archive"],
        "source_members": inspection["members"],
        "prep_summary": prep,
        "aggregation_audit": aggregation_audit,
        "results_summary": summary,
        "output_inventory": _portable_inventory(),
    }
    write_json(manifest, DATA_DIR / "run_manifest.json")
    save_csv(pd.DataFrame(manifest["output_inventory"]), DATA_DIR / "output_inventory.csv")
    logger.info("v2 pipeline complete in %.1f seconds", manifest["duration_seconds"])


if __name__ == "__main__":
    main()
