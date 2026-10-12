from __future__ import annotations

import gc
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.config import OUT as V2, REPORT_DIR as V2_REPORT, ROOT, SEED, WORK_DIR, YEARS, ZIP_PATH
from analysis.figures_v2_1 import create_all_figures_v2_1
from analysis.reporting_v2_1 import write_reports_v2_1
from analysis.statistics import build_panels
from analysis.v2_1 import (
    DATA_DIR,
    FIGURE_DATA_DIR,
    LOG_DIR,
    N_BOOT,
    OUT,
    REPORT_DIR,
    TABLE_DIR,
    ensure_v21_directories,
    environment_manifest,
    run_country_level_null_adjusted,
    run_paired_alpha_residual_block,
    run_paired_phi_valid_inference,
    run_pairwise_covariance_audit,
    run_reporting_aware_paired,
    run_residual_block_decomposition,
    run_rolling20_residual_block,
    run_signature,
    save_csv,
    sha256_file,
)


def _safe_clean(path: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != WORK_DIR.resolve() or resolved.name not in {"outputs_v2_1", "report_v2_1"}:
        raise RuntimeError(f"Refusing to clean unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _logger() -> logging.Logger:
    logger = logging.getLogger("grain_trade_v2_1")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "pipeline.log", mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_v2_cache(signature: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        V2 / "data" / "core_four_crops_long.parquet",
        V2 / "data" / "country_metrics_full.parquet",
        V2 / "data" / "country_metrics_rolling20.parquet",
        V2 / "data" / "reporter_year_reporting_activity.parquet",
        V2 / "data" / "run_manifest.json",
        V2 / "tables" / "table2_alpha_estimates.csv",
        V2 / "tables" / "table6_null_model.csv",
        V2 / "figure_data" / "Fig1_full_period_country_data.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing validated v2 inputs: {missing}")
    old_manifest = json.loads((V2 / "data" / "run_manifest.json").read_text(encoding="utf-8-sig"))
    old_hash = old_manifest["signature"]["raw_zip_sha256"]
    if signature["raw_zip_sha256"] != old_hash:
        raise AssertionError("Raw archive hash does not match the validated v2 run")
    core = pd.read_parquet(required[0])
    full = pd.read_parquet(required[1])
    rolling = pd.read_parquet(required[2])
    activity = pd.read_parquet(required[3])
    grain = ["Reporter Country Code", "Partner Country Code", "Item Code", "Element Code", "year"]
    duplicate_grain_rows = int(core.duplicated(grain).sum())
    validation = {
        "passed": bool(
            len(core) == 2_045_199
            and duplicate_grain_rows == 0
            and sorted(core["year"].unique().tolist()) == YEARS
            and core["crop"].nunique() == 4
            and core["direction"].nunique() == 2
        ),
        "raw_zip": ZIP_PATH.name,
        "raw_zip_sha256": signature["raw_zip_sha256"],
        "matches_v2_manifest": signature["raw_zip_sha256"] == old_hash,
        "core_long_rows": len(core),
        "duplicate_grain_rows": duplicate_grain_rows,
        "years": [int(core["year"].min()), int(core["year"].max())],
        "crop_count": int(core["crop"].nunique()),
        "direction_count": int(core["direction"].nunique()),
        "full_metric_rows": len(full),
        "rolling_metric_rows": len(rolling),
        "reporting_activity_rows": len(activity),
        "input_policy": "validated v2 cache only; no new data or specification search",
    }
    if not validation["passed"]:
        raise AssertionError(f"v2 cache validation failed: {validation}")
    _write_json(validation, DATA_DIR / "cache_validation.json")
    return validation, core, full, rolling, activity


def _inventory() -> pd.DataFrame:
    rows = []
    for base, label in ((OUT, "outputs_v2_1"), (REPORT_DIR, "report_v2_1")):
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rows.append({"path": f"{label}/{path.relative_to(base).as_posix()}", "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def _file_hashes() -> pd.DataFrame:
    excluded = {
        OUT / "data" / "file_hashes_sha256.csv",
        OUT / "data" / "output_inventory.csv",
        OUT / "data" / "run_manifest.json",
        OUT / "logs" / "pipeline.log",
    }
    rows = []
    for base, label in ((OUT, "outputs_v2_1"), (REPORT_DIR, "report_v2_1")):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path not in excluded:
                rows.append({
                    "path": f"{label}/{path.relative_to(base).as_posix()}",
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                })
    return pd.DataFrame(rows)


def main() -> None:
    _safe_clean(OUT)
    _safe_clean(REPORT_DIR)
    ensure_v21_directories()
    shutil.copy2(V2 / "logs" / "pipeline.log", LOG_DIR / "pipeline_previous_failed_attempt.log")
    logger = _logger()
    np.random.seed(SEED)
    started = datetime.now(timezone.utc)
    logger.info("pipeline start")

    signature = run_signature()
    validation, core, full_metrics, rolling_metrics, activity = _validate_v2_cache(signature)
    logger.info("data/cache validation")

    logger.info("analysis")
    panels = build_panels(core)
    paired_phi = run_paired_phi_valid_inference(full_metrics)
    decomposition_result = run_residual_block_decomposition(panels, full_metrics)
    decomposition = decomposition_result["summary"]
    paired_alpha, _ = run_paired_alpha_residual_block(
        full_metrics, decomposition_result["cache"], decomposition_result["cubes"][4]
    )
    del decomposition_result
    gc.collect()
    rolling = run_rolling20_residual_block(panels, rolling_metrics)
    reporting = run_reporting_aware_paired(panels, activity, full_metrics)
    null_adjusted = run_country_level_null_adjusted(panels, full_metrics, n_iter=N_BOOT)
    covariance_audit = run_pairwise_covariance_audit(panels, full_metrics)

    # Preserve the already-run prespecified v2 specifications as transparent v2.1 inputs.
    for source, target in (
        (V2 / "tables" / "table_v2_source_diversification_sensitivity.csv", TABLE_DIR / "source_diversification_18_prespecified_specs.csv"),
        (V2 / "tables" / "table_v2_reporting_aware_sensitivity.csv", TABLE_DIR / "reporting_entity_coverage_sensitivity.csv"),
    ):
        shutil.copy2(source, target)

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (DATA_DIR / "pytest_output.txt").write_text(test_result.stdout + test_result.stderr, encoding="utf-8")
    if test_result.returncode:
        raise SystemExit(test_result.returncode)
    match = re.search(r"(\d+) passed", test_result.stdout)
    tests_passed = int(match.group(1)) if match else 0
    logger.info("all tests passed")

    alpha = pd.read_csv(V2 / "tables" / "table2_alpha_estimates.csv")
    full_fit = pd.read_csv(V2 / "figure_data" / "Fig1_full_period_country_data.csv")
    circular = pd.read_csv(V2 / "tables" / "table6_null_model.csv")
    figures = create_all_figures_v2_1(
        full_fit,
        alpha,
        full_metrics,
        rolling,
        paired_alpha,
        paired_phi,
        decomposition,
        circular,
        null_adjusted["paired"],
        reporting["summary"],
    )
    logger.info("figures generated")

    alpha_main = alpha[
        (alpha["window_id"] == "full_1986_2024")
        & (alpha["sample_type"] == "A_dynamic")
        & (alpha["trend"] == "linear")
        & (alpha["method"] == "OLS_HC3")
    ].copy()
    results_summary = write_reports_v2_1(
        alpha_main,
        paired_phi["summary"],
        decomposition,
        paired_alpha,
        reporting["summary"],
        null_adjusted["paired"],
        covariance_audit,
        figures,
        tests_passed,
    )
    logger.info("reports generated")

    finished = datetime.now(timezone.utc)
    save_csv(_file_hashes(), DATA_DIR / "file_hashes_sha256.csv")
    manifest = {
        "schema_version": "2.1",
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "signature": signature,
        "environment": environment_manifest(signature),
        "cache_validation": validation,
        "analysis_contract": {
            "data": "supplied archive through hash-validated v2 cache only",
            "years": YEARS,
            "formal_draws": N_BOOT,
            "residual_block_lengths": [3, 4, 5],
            "main_residual_block_length": 4,
            "new_specification_search": False,
        },
        "tests_passed": tests_passed,
        "results_summary": results_summary,
        "hash_policy": "SHA256 covers all immutable artifacts except the self-referential hash, inventory and manifest files and the mutable current pipeline log.",
    }
    _write_json(manifest, DATA_DIR / "run_manifest.json")
    save_csv(_inventory(), DATA_DIR / "output_inventory.csv")
    logger.info("manifest generated")
    logger.info("v2.1 pipeline complete")
    for handler in logger.handlers:
        handler.flush()
    # Refresh after the final two required log records. Iteration makes the
    # inventory's own byte count converge without adding any post-completion log.
    for _ in range(4):
        save_csv(_inventory(), DATA_DIR / "output_inventory.csv")


if __name__ == "__main__":
    main()
