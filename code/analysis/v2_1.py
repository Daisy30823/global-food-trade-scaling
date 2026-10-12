from __future__ import annotations

import hashlib
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .analysis_runner import _null_variances, stable_seed
from .config import CROPS, ROOT, SEED, WORK_DIR, YEARS, ZIP_PATH
from .statistics import TradePanel, detrend_residuals


OUT = WORK_DIR / "outputs_v2_1"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
FIGURE_DATA_DIR = OUT / "figure_data"
FIGURE_MAIN_DIR = OUT / "figures" / "main"
FIGURE_SUPP_DIR = OUT / "figures" / "supplement"
LOG_DIR = OUT / "logs"
REPORT_DIR = WORK_DIR / "report_v2_1"
N_BOOT = 2_000
BLOCK_LENGTHS = (3, 4, 5)
TINY = np.finfo(float).tiny


def ensure_v21_directories() -> None:
    for path in (DATA_DIR, TABLE_DIR, FIGURE_DATA_DIR, FIGURE_MAIN_DIR, FIGURE_SUPP_DIR, LOG_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_hash() -> str:
    paths = [ROOT / "run_all_v2_1.py", ROOT / "requirements.txt"]
    paths += sorted((ROOT / "analysis").glob("*.py"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_signature() -> dict:
    return {
        "raw_zip_sha256": sha256_file(ZIP_PATH),
        "code_sha256": code_hash(),
        "seed": SEED,
        "formal_draws": N_BOOT,
        "main_block_length": 4,
        "sensitivity_block_lengths": [3, 5],
    }


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return np.nan
    xc = x - x.mean()
    denom = float(xc @ xc)
    return float(xc @ (y - y.mean()) / denom) if denom > 0 else np.nan


def _percentile(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return tuple(map(float, np.quantile(values, [0.025, 0.975]))) if len(values) else (np.nan, np.nan)


def _interval_summary(values: np.ndarray, point: float) -> dict:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    q025, q975 = _percentile(values)
    return {
        "estimate": float(point),
        "percentile_ci_low": q025,
        "percentile_ci_high": q975,
        "basic_ci_low": float(2 * point - q975),
        "basic_ci_high": float(2 * point - q025),
        "bootstrap_bias": float(values.mean() - point),
        "bootstrap_median": float(np.median(values)),
        "effective_draws": int(len(values)),
    }


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, int(np.ceil(n / block_length)))
    return np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]


def common_block_indices(n: int, block_length: int, n_boot: int = N_BOOT) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(f"v21-residual-global-L{block_length}-T{n}"))
    return np.vstack([moving_block_indices(n, block_length, rng) for _ in range(n_boot)])


def residual_maker(n: int) -> np.ndarray:
    t = np.arange(n, dtype=float)
    x = np.column_stack([np.ones(n), t])
    return np.eye(n) - x @ np.linalg.inv(x.T @ x) @ x.T


def block_weight_matrices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_boot, n = indices.shape
    maker = residual_maker(n)
    selector = np.eye(n)[indices]
    weights = np.einsum("qai,ac,qcj->qij", selector, maker, selector, optimize=True)
    counts = np.zeros((n_boot, n), dtype=float)
    for b in range(n_boot):
        counts[b] = np.bincount(indices[b], minlength=n)
    return weights, counts


@dataclass
class ResidualPanelCache:
    keys: list[tuple[str, str, int]]
    country_names: list[str]
    mu: np.ndarray
    total_residual: np.ndarray
    total_outer: np.ndarray
    edge_gram: np.ndarray


def build_residual_cache(
    panels: dict[tuple[str, str, int], TradePanel], year_positions: np.ndarray | None = None
) -> ResidualPanelCache:
    idx = np.arange(len(YEARS)) if year_positions is None else np.asarray(year_positions, int)
    keys = sorted(panels)
    mu, total_residual, total_outer, edge_gram, names = [], [], [], [], []
    for key in keys:
        panel = panels[key]
        values = panel.values[:, idx]
        residual = detrend_residuals(values, degree=1)
        total = residual.sum(axis=0)
        mu.append(float(values.sum(axis=0).mean()))
        total_residual.append(total)
        total_outer.append(np.outer(total, total).ravel())
        edge_gram.append((residual.T @ residual).ravel())
        names.append(panel.country_name)
    return ResidualPanelCache(
        keys=keys,
        country_names=names,
        mu=np.asarray(mu),
        total_residual=np.asarray(total_residual),
        total_outer=np.asarray(total_outer),
        edge_gram=np.asarray(edge_gram),
    )


def residual_block_metric_cube(cache: ResidualPanelCache, block_length: int, n_boot: int = N_BOOT) -> dict:
    n = cache.total_residual.shape[1]
    indices = common_block_indices(n, block_length, n_boot)
    weights, counts = block_weight_matrices(indices)
    flat_weights = weights.reshape(n_boot, n * n)
    mu = cache.mu[:, None] + cache.total_residual @ counts.T / n
    variance = cache.total_outer @ flat_weights.T / (n - 1)
    diagonal = cache.edge_gram @ flat_weights.T / (n - 1)
    phi = np.divide(variance, diagonal, out=np.full_like(variance, np.nan), where=diagonal != 0)
    return {
        "indices": indices,
        "mu": mu,
        "variance": variance,
        "D": diagonal,
        "phi": phi,
    }


def residual_block_mu_variance_cube(
    cache: ResidualPanelCache, block_length: int, n_boot: int = N_BOOT
) -> dict:
    """Compute only the quantities needed by rolling Taylor fits.

    Skipping the diagonal edge-variance multiplication substantially reduces the
    cost of the 20 overlapping rolling windows without changing the resampling
    algorithm or its common global year indices.
    """
    n = cache.total_residual.shape[1]
    indices = common_block_indices(n, block_length, n_boot)
    weights, counts = block_weight_matrices(indices)
    flat_weights = weights.reshape(n_boot, n * n)
    return {
        "indices": indices,
        "mu": cache.mu[:, None] + cache.total_residual @ counts.T / n,
        "variance": cache.total_outer @ flat_weights.T / (n - 1),
    }


def _row_lookup(cache: ResidualPanelCache) -> dict[tuple[str, str, int], int]:
    return {key: i for i, key in enumerate(cache.keys)}


def _component_points(group: pd.DataFrame) -> dict[str, float]:
    x = np.log(group["mu"].to_numpy(float))
    points = {
        "alpha_V": _slope(x, np.log(group["variance"])),
        "alpha_D": _slope(x, np.log(group["D"])),
        "beta_phi": _slope(x, np.log(group["phi"])),
    }
    error = abs(points["alpha_V"] - points["alpha_D"] - points["beta_phi"])
    if error >= 1e-8:
        raise AssertionError(f"Observed alpha identity failed: {error}")
    return points


def _component_draws(
    cube: dict,
    rows: np.ndarray,
    country_bootstrap: bool,
    seed_label: str,
) -> dict[str, np.ndarray]:
    n_boot = cube["mu"].shape[1]
    rng = np.random.default_rng(stable_seed(seed_label))
    result = {name: np.full(n_boot, np.nan) for name in ("alpha_V", "alpha_D", "beta_phi")}
    for b in range(n_boot):
        use = rows
        if country_bootstrap:
            use = rows[rng.integers(0, len(rows), len(rows))]
        mu = cube["mu"][use, b]
        v = cube["variance"][use, b]
        d = cube["D"][use, b]
        phi = cube["phi"][use, b]
        valid = (mu > 0) & (v > 0) & (d > 0) & (phi > 0)
        if valid.sum() < 4:
            continue
        x = np.log(mu[valid])
        result["alpha_V"][b] = _slope(x, np.log(v[valid]))
        result["alpha_D"][b] = _slope(x, np.log(d[valid]))
        result["beta_phi"][b] = _slope(x, np.log(phi[valid]))
    error = np.abs(result["alpha_V"] - result["alpha_D"] - result["beta_phi"])
    if np.nanmax(error) >= 1e-8:
        raise AssertionError(f"Residual-block alpha identity failed: {np.nanmax(error)}")
    return result


def run_residual_block_decomposition(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame
) -> dict:
    cache = build_residual_cache(panels)
    lookup = _row_lookup(cache)
    cubes = {length: residual_block_metric_cube(cache, length) for length in BLOCK_LENGTHS}
    summaries, draw_frames = [], []
    for (crop, direction), raw_group in full_metrics.groupby(["crop", "direction"], sort=False):
        group = raw_group[raw_group["eligible_covariance"] & raw_group["phi"].gt(0)].sort_values("country_code")
        rows = np.asarray([lookup[(crop, direction, int(code))] for code in group["country_code"]], int)
        points = _component_points(group)
        for length in BLOCK_LENGTHS:
            year_draws = _component_draws(cubes[length], rows, False, f"v21-year-L{length}-{crop}-{direction}")
            schemes = [("residual_year_block", year_draws)]
            if length == 4:
                schemes.append(("residual_two_stage", _component_draws(cubes[length], rows, True, f"v21-two-stage-{crop}-{direction}")))
            for scheme, component_draws in schemes:
                for component, draws in component_draws.items():
                    summaries.append({
                        "crop": crop,
                        "direction": direction,
                        "component": component,
                        "scheme": scheme,
                        "block_length": length,
                        "n_reporting_economies": len(group),
                        **_interval_summary(draws, points[component]),
                    })
                    draw_frames.append(pd.DataFrame({
                        "crop": crop,
                        "direction": direction,
                        "component": component,
                        "scheme": scheme,
                        "block_length": length,
                        "draw": np.arange(N_BOOT),
                        "value": draws,
                    }))
    summary = pd.DataFrame(summaries)
    draws = pd.concat(draw_frames, ignore_index=True)
    save_csv(summary, TABLE_DIR / "exact_decomposition_residual_block.csv")
    draws.to_parquet(DATA_DIR / "residual_block_bootstrap_draws.parquet", index=False)
    return {"summary": summary, "draws": draws, "cache": cache, "cubes": cubes}


def run_paired_alpha_residual_block(
    full_metrics: pd.DataFrame, cache: ResidualPanelCache, cube: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = _row_lookup(cache)
    summaries, draw_frames = [], []
    for crop, crop_group in full_metrics.groupby("crop", sort=False):
        imp = crop_group[(crop_group["direction"] == "Import") & crop_group["eligible_taylor"]].set_index("country_code")
        exp = crop_group[(crop_group["direction"] == "Export") & crop_group["eligible_taylor"]].set_index("country_code")
        common = np.asarray(sorted(set(imp.index) & set(exp.index)), int)
        imp_rows = np.asarray([lookup[(crop, "Import", int(code))] for code in common])
        exp_rows = np.asarray([lookup[(crop, "Export", int(code))] for code in common])
        point_imp = _slope(np.log(imp.loc[common, "mu"]), np.log(imp.loc[common, "variance"]))
        point_exp = _slope(np.log(exp.loc[common, "mu"]), np.log(exp.loc[common, "variance"]))
        point = point_imp - point_exp
        rng = np.random.default_rng(stable_seed(f"v21-paired-alpha-two-stage-{crop}"))
        imp_draw = np.full(N_BOOT, np.nan)
        exp_draw = np.full(N_BOOT, np.nan)
        for b in range(N_BOOT):
            country_idx = rng.integers(0, len(common), len(common))
            ri, re = imp_rows[country_idx], exp_rows[country_idx]
            mi, vi = cube["mu"][ri, b], cube["variance"][ri, b]
            me, ve = cube["mu"][re, b], cube["variance"][re, b]
            valid_i = (mi > 0) & (vi > 0)
            valid_e = (me > 0) & (ve > 0)
            imp_draw[b] = _slope(np.log(mi[valid_i]), np.log(vi[valid_i]))
            exp_draw[b] = _slope(np.log(me[valid_e]), np.log(ve[valid_e]))
        difference = imp_draw - exp_draw
        summaries.append({
            "crop": crop,
            "n_common_reporting_economies": len(common),
            "alpha_import": point_imp,
            "alpha_export": point_exp,
            "estimand": "alpha_import_minus_export",
            "block_length": 4,
            **_interval_summary(difference, point),
        })
        draw_frames.append(pd.DataFrame({
            "crop": crop,
            "draw": np.arange(N_BOOT),
            "alpha_import": imp_draw,
            "alpha_export": exp_draw,
            "alpha_import_minus_export": difference,
        }))
    summary = pd.DataFrame(summaries)
    draws = pd.concat(draw_frames, ignore_index=True)
    save_csv(summary, TABLE_DIR / "paired_alpha_residual_block.csv")
    draws.to_parquet(DATA_DIR / "paired_alpha_residual_block_draws.parquet", index=False)
    return summary, draws


def _paired_phi_one(crop: str, metrics: pd.DataFrame, specification: str) -> tuple[dict, pd.DataFrame]:
    imp = metrics[(metrics["crop"] == crop) & (metrics["direction"] == "Import") & metrics["eligible_covariance"] & metrics["phi"].gt(0)].set_index("country_code")
    exp = metrics[(metrics["crop"] == crop) & (metrics["direction"] == "Export") & metrics["eligible_covariance"] & metrics["phi"].gt(0)].set_index("country_code")
    common = np.asarray(sorted(set(imp.index) & set(exp.index)), int)
    phi_i = imp.loc[common, "phi"].to_numpy(float)
    phi_e = exp.loc[common, "phi"].to_numpy(float)
    difference = phi_e - phi_i
    log_ratio = np.log(phi_e / phi_i)
    rng = np.random.default_rng(stable_seed(f"v21-paired-phi-{specification}-{crop}"))
    draws = []
    for b in range(N_BOOT):
        idx = rng.integers(0, len(common), len(common))
        d, r = difference[idx], log_ratio[idx]
        draws.append({
            "specification": specification,
            "crop": crop,
            "draw": b,
            "median_difference": float(np.median(d)),
            "mean_difference": float(np.mean(d)),
            "median_log_ratio": float(np.median(r)),
            "typical_phi_ratio": float(np.exp(np.median(r))),
            "share_export_gt_import": float(np.mean(d > 0)),
        })
    draw_frame = pd.DataFrame(draws)
    sign = stats.binomtest(int(np.sum(difference > 0)), int(np.sum(difference != 0)), 0.5, alternative="two-sided")
    try:
        wilcoxon_p = float(stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = np.nan
    row = {
        "specification": specification,
        "crop": crop,
        "n_pairs": len(common),
        "median_difference": float(np.median(difference)),
        "mean_difference": float(np.mean(difference)),
        "median_log_ratio": float(np.median(log_ratio)),
        "typical_phi_ratio": float(np.exp(np.median(log_ratio))),
        "share_export_gt_import": float(np.mean(difference > 0)),
        "exact_sign_test_p": float(sign.pvalue),
        "wilcoxon_signed_rank_p": wilcoxon_p,
    }
    for metric in ("median_difference", "mean_difference", "median_log_ratio", "typical_phi_ratio", "share_export_gt_import"):
        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = _percentile(draw_frame[metric].to_numpy())
    country_pairs = pd.DataFrame({
        "specification": specification,
        "crop": crop,
        "country_code": common,
        "country": imp.loc[common, "country"].to_numpy() if "country" in imp else common.astype(str),
        "phi_import": phi_i,
        "phi_export": phi_e,
        "difference": difference,
        "log_ratio": log_ratio,
    })
    return row, draw_frame, country_pairs


def run_paired_phi_valid_inference(full_metrics: pd.DataFrame) -> dict:
    rows, draws, pairs = [], [], []
    for crop in full_metrics["crop"].drop_duplicates():
        row, draw, country = _paired_phi_one(crop, full_metrics, "blank_as_zero")
        rows.append(row); draws.append(draw); pairs.append(country)
    summary = pd.DataFrame(rows)
    draw_frame = pd.concat(draws, ignore_index=True)
    pair_frame = pd.concat(pairs, ignore_index=True)
    save_csv(summary, TABLE_DIR / "paired_phi_valid_inference.csv")
    draw_frame.to_parquet(DATA_DIR / "paired_phi_bootstrap_draws.parquet", index=False)
    save_csv(pair_frame, FIGURE_DATA_DIR / "paired_phi_country_pairs.csv")
    return {"summary": summary, "draws": draw_frame, "pairs": pair_frame}


def compute_reporting_aware_metrics(
    panels: dict[tuple[str, str, int], TradePanel], activity: pd.DataFrame
) -> pd.DataFrame:
    activity_map = {}
    for (crop, direction, code), group in activity.groupby(["crop", "direction", "Reporter Country Code"], sort=False):
        activity_map[(crop, direction, int(code))] = group.set_index("year")["reporter_active"].reindex(YEARS, fill_value=False).to_numpy(bool)
    rows = []
    for key, panel in panels.items():
        idx = np.flatnonzero(activity_map.get(key, np.zeros(len(YEARS), bool)))
        if len(idx) < 20:
            continue
        values = panel.values[:, idx]
        residual = detrend_residuals(values, degree=1)
        total = values.sum(axis=0)
        mu = float(total.mean())
        variance = float(np.var(residual.sum(axis=0), ddof=1))
        diagonal = float(np.var(residual, axis=1, ddof=1).sum())
        phi = variance / diagonal if diagonal > 0 else np.nan
        positive_years = int(np.sum(total > 0))
        partners = int(np.sum(np.any(values > 0, axis=1)))
        rows.append({
            "crop": panel.crop, "direction": panel.direction, "country_code": panel.country_code,
            "country": panel.country_name, "covered_years": len(idx), "noncoverage_years": len(YEARS) - len(idx),
            "mu": mu, "variance": variance, "D": diagonal, "phi": phi,
            "eligible_taylor": bool(mu > 0 and variance > 0 and positive_years >= 20),
            "eligible_covariance": bool(mu > 0 and variance > 0 and diagonal > 0 and phi > 0 and positive_years >= 20 and partners >= 2),
        })
    return pd.DataFrame(rows)


def run_reporting_aware_paired(
    panels: dict[tuple[str, str, int], TradePanel], activity: pd.DataFrame, blank_metrics: pd.DataFrame
) -> dict:
    aware = compute_reporting_aware_metrics(panels, activity)
    aware.to_parquet(DATA_DIR / "country_metrics_reporting_aware.parquet", index=False)
    rows, draws, pairs = [], [], []
    for specification, frame in (("blank_as_zero", blank_metrics), ("reporting_aware", aware)):
        for crop in blank_metrics["crop"].drop_duplicates():
            row, draw, country = _paired_phi_one(crop, frame, specification)
            rows.append(row); draws.append(draw); pairs.append(country)
    summary = pd.DataFrame(rows)
    save_csv(summary, TABLE_DIR / "paired_phi_reporting_aware.csv")
    pd.concat(draws, ignore_index=True).to_parquet(DATA_DIR / "paired_phi_reporting_aware_bootstrap_draws.parquet", index=False)
    save_csv(pd.concat(pairs, ignore_index=True), FIGURE_DATA_DIR / "paired_phi_reporting_aware_pairs.csv")
    return {"summary": summary, "metrics": aware}


def run_country_level_null_adjusted(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame, n_iter: int = N_BOOT
) -> dict:
    observed_lookup = full_metrics.set_index(["crop", "direction", "country_code"])
    rows = []
    for key, panel in panels.items():
        if key not in observed_lookup.index:
            continue
        observed = observed_lookup.loc[key]
        if not bool(observed["eligible_covariance"]) or not float(observed["phi"]) > 0:
            continue
        residual = detrend_residuals(panel.values, degree=1)
        residual = residual[np.var(residual, axis=1, ddof=1) > 0]
        if residual.shape[0] < 2:
            continue
        diagonal = float(np.var(residual, axis=1, ddof=1).sum())
        rng = np.random.default_rng(stable_seed(f"v21-country-null-{key}"))
        null_phi = _null_variances(residual, n_iter, rng, batch_size=100) / diagonal
        null_median = float(np.median(null_phi))
        rows.append({
            "crop": key[0], "direction": key[1], "country_code": key[2], "country": panel.country_name,
            "phi_observed": float(observed["phi"]),
            "phi_null_mean": float(np.mean(null_phi)), "phi_null_median": null_median,
            "phi_null_q025": float(np.quantile(null_phi, 0.025)), "phi_null_q975": float(np.quantile(null_phi, 0.975)),
            "delta_sync": float(np.log(float(observed["phi"]) / null_median)), "null_iterations": n_iter,
        })
    country = pd.DataFrame(rows)
    save_csv(country, TABLE_DIR / "country_level_null_adjusted_phi.csv")
    paired_rows, paired_draws = [], []
    for crop, group in country.groupby("crop", sort=False):
        imp = group[group["direction"] == "Import"].set_index("country_code")
        exp = group[group["direction"] == "Export"].set_index("country_code")
        common = np.asarray(sorted(set(imp.index) & set(exp.index)), int)
        difference = exp.loc[common, "delta_sync"].to_numpy() - imp.loc[common, "delta_sync"].to_numpy()
        rng = np.random.default_rng(stable_seed(f"v21-paired-null-adjusted-{crop}"))
        draws = np.asarray([np.median(difference[rng.integers(0, len(common), len(common))]) for _ in range(N_BOOT)])
        low, high = _percentile(draws)
        sign = stats.binomtest(int(np.sum(difference > 0)), int(np.sum(difference != 0)), 0.5, alternative="two-sided")
        try:
            wp = float(stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided").pvalue)
        except ValueError:
            wp = np.nan
        paired_rows.append({
            "crop": crop, "n_pairs": len(common), "paired_median_export_minus_import": float(np.median(difference)),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high, "exact_sign_test_p": float(sign.pvalue),
            "wilcoxon_signed_rank_p": wp,
            "interpretation": "directional contemporaneous synchronization difference beyond marginal edge time structure",
        })
        paired_draws.append(pd.DataFrame({"crop": crop, "draw": np.arange(N_BOOT), "paired_median": draws}))
    paired = pd.DataFrame(paired_rows)
    save_csv(paired, TABLE_DIR / "paired_null_adjusted_direction.csv")
    pd.concat(paired_draws, ignore_index=True).to_parquet(DATA_DIR / "paired_null_adjusted_bootstrap_draws.parquet", index=False)
    return {"country": country, "paired": paired}


def run_pairwise_covariance_audit(
    panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    selected = []
    for _, group in full_metrics[full_metrics["eligible_covariance"]].groupby(["crop", "direction"], sort=False):
        take = min(20, len(group))
        selected.append(group.iloc[rng.choice(len(group), take, replace=False)])
    sample = pd.concat(selected, ignore_index=True)
    rows = []
    for row in sample.itertuples(index=False):
        key = (row.crop, row.direction, int(row.country_code))
        residual = detrend_residuals(panels[key].values, degree=1)
        covariance = np.cov(residual, ddof=1)
        if np.ndim(covariance) < 2:
            continue
        pairwise = float(2 * np.triu(covariance, 1).sum())
        via_identity = float(row.variance - row.D)
        scale = max(abs(pairwise), abs(via_identity), 1.0)
        rows.append({
            "crop": row.crop, "direction": row.direction, "country_code": int(row.country_code), "country": row.country,
            "partner_count": residual.shape[0], "C_pairwise": pairwise, "C_v_minus_d": via_identity,
            "absolute_error": abs(pairwise - via_identity), "relative_error": abs(pairwise - via_identity) / scale,
            "ddof": 1,
        })
    audit = pd.DataFrame(rows)
    if audit["relative_error"].max() >= 1e-8:
        raise AssertionError(f"Pairwise covariance audit failed: {audit['relative_error'].max()}")
    save_csv(audit, DATA_DIR / "pairwise_covariance_audit.csv")
    return audit


def run_rolling20_residual_block(
    panels: dict[tuple[str, str, int], TradePanel], rolling_metrics: pd.DataFrame
) -> pd.DataFrame:
    results = []
    for (start, end), window_metrics in rolling_metrics.groupby(["start_year", "end_year"], sort=True):
        positions = np.arange(YEARS.index(int(start)), YEARS.index(int(end)) + 1)
        eligible_window = window_metrics[window_metrics["eligible_taylor"]]
        needed = {
            (row.crop, row.direction, int(row.country_code))
            for row in eligible_window.itertuples(index=False)
        }
        window_panels = {key: panels[key] for key in needed}
        cache = build_residual_cache(window_panels, positions)
        cube = residual_block_mu_variance_cube(cache, 4)
        lookup = _row_lookup(cache)
        for (crop, direction), group in window_metrics.groupby(["crop", "direction"], sort=False):
            eligible = group[group["eligible_taylor"]].sort_values("country_code")
            rows = np.asarray([lookup[(crop, direction, int(code))] for code in eligible["country_code"]])
            point = _slope(np.log(eligible["mu"]), np.log(eligible["variance"]))
            draws = np.full(N_BOOT, np.nan)
            for b in range(N_BOOT):
                mu, variance = cube["mu"][rows, b], cube["variance"][rows, b]
                valid = (mu > 0) & (variance > 0)
                draws[b] = _slope(np.log(mu[valid]), np.log(variance[valid]))
            results.append({
                "window_id": f"roll20_{int(start)}_{int(end)}", "start_year": int(start), "end_year": int(end),
                "crop": crop, "direction": direction, "estimate": point, "block_length": 4,
                **_interval_summary(draws, point), "descriptive_only": True,
            })
    result = pd.DataFrame(results)
    save_csv(result, FIGURE_DATA_DIR / "rolling20_residual_block_bootstrap.csv")
    return result


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
        "resolved_versions": {
            "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
        },
    }
