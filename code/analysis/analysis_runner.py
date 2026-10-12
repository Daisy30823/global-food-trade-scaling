from __future__ import annotations

import zlib
from itertools import product

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import DATA_DIR, FIG_DATA_DIR, SEED, TABLE_DIR, YEARS
from .statistics import (
    TradePanel,
    build_panels,
    compute_window_metrics,
    detrend_residuals,
    fit_model_comparison,
    fit_scaling,
    m2_alpha_eff,
)
from .utils import get_logger, save_csv, write_json

LOGGER = get_logger()


def stable_seed(label: str) -> int:
    return int((SEED + zlib.crc32(label.encode("utf-8"))) % (2**32 - 1))


def window_definitions() -> list[dict]:
    windows = [{"window_id": "full_1986_2024", "label": "1986-2024", "type": "full", "start": 1986, "end": 2024}]
    windows += [
        {"window_id": f"roll20_{start}_{start + 19}", "label": f"{start}-{start + 19}", "type": "rolling20", "start": start, "end": start + 19}
        for start in range(1986, 2006)
    ]
    windows += [
        {"window_id": "nonoverlap_1986_2004", "label": "1986-2004", "type": "nonoverlap", "start": 1986, "end": 2004},
        {"window_id": "nonoverlap_2005_2024", "label": "2005-2024", "type": "nonoverlap", "start": 2005, "end": 2024},
    ]
    windows += [
        {"window_id": f"roll15_{start}_{start + 14}", "label": f"{start}-{start + 14}", "type": "rolling15", "start": start, "end": start + 14}
        for start in range(1986, 2011)
    ]
    windows += [
        {"window_id": f"roll25_{start}_{start + 24}", "label": f"{start}-{start + 24}", "type": "rolling25", "start": start, "end": start + 24}
        for start in range(1986, 2001)
    ]
    return windows


def _fit_rows(group: pd.DataFrame, context: dict, n_boot: int = 1000) -> tuple[list[dict], pd.DataFrame, np.ndarray]:
    result, country_fit, boot = fit_scaling(group, n_boot=n_boot, seed=stable_seed(str(context)))
    common = {**context, **result}
    rows = [
        {**common, "method": "OLS_HC3", "estimate": result["alpha"]},
        {**context, "method": "Huber", "estimate": result["huber_alpha"], "n_countries": result["n_countries"]},
        {**context, "method": "TheilSen", "estimate": result["theil_sen_alpha"], "n_countries": result["n_countries"]},
    ]
    return rows, country_fit, boot


def run_taylor_and_models(panels: dict[tuple[str, str, int], TradePanel], n_boot: int = 1000) -> dict:
    windows = window_definitions()
    metric_cache: dict[str, pd.DataFrame] = {}
    for i, window in enumerate(windows, start=1):
        LOGGER.info("Computing country metrics for %s (%d/%d)", window["label"], i, len(windows))
        metric_cache[window["window_id"]] = compute_window_metrics(panels, window["start"], window["end"], degree=1).assign(
            window_id=window["window_id"], window_label=window["label"], window_type=window["type"]
        )

    rolling_labels = [w["window_id"] for w in windows if w["type"] == "rolling20"]
    fixed: dict[tuple[str, str], set[int]] = {}
    full_metrics = metric_cache["full_1986_2024"]
    combos = full_metrics[["crop", "direction"]].drop_duplicates().itertuples(index=False, name=None)
    for crop, direction in combos:
        eligible_sets = []
        for label in rolling_labels:
            m = metric_cache[label]
            ok = m[(m["crop"] == crop) & (m["direction"] == direction) & m["eligible_taylor"]]
            eligible_sets.append(set(ok["country_code"].astype(int)))
        fixed[(crop, direction)] = set.intersection(*eligible_sets) if eligible_sets else set()

    all_rows = []
    fitted_full = []
    boot_full = []
    rolling_metrics_out = []
    for window in windows:
        metrics = metric_cache[window["window_id"]]
        if window["type"] == "rolling20":
            rolling_metrics_out.append(metrics)
        for (crop, direction), group in metrics.groupby(["crop", "direction"], sort=False):
            context = {
                "crop": crop,
                "direction": direction,
                "window": window["label"],
                "window_id": window["window_id"],
                "window_label": window["label"],
                "window_type": window["type"],
                "start_year": window["start"],
                "end_year": window["end"],
                "sample_type": "A_dynamic",
                "trend": "linear",
                "transform": "tonnes",
            }
            try:
                rows, fitted, boot = _fit_rows(group, context, n_boot=n_boot)
                all_rows.extend(rows)
                if window["type"] == "full":
                    fitted_full.append(fitted.assign(crop=crop, direction=direction, sample_type="A_dynamic"))
                    boot_full.append(pd.DataFrame({"draw": np.arange(len(boot)), "alpha": boot, "crop": crop, "direction": direction, "sample_type": "A_dynamic"}))
            except ValueError:
                LOGGER.warning("Insufficient dynamic sample for %s %s %s", crop, direction, window["label"])

            if window["type"] in {"full", "rolling20"}:
                codes = fixed.get((crop, direction), set())
                fixed_group = group[group["country_code"].isin(codes)]
                context_b = {**context, "sample_type": "B_fixed"}
                try:
                    rows, fitted, boot = _fit_rows(fixed_group, context_b, n_boot=n_boot)
                    all_rows.extend(rows)
                    if window["type"] == "full":
                        fitted_full.append(fitted.assign(crop=crop, direction=direction, sample_type="B_fixed"))
                        boot_full.append(pd.DataFrame({"draw": np.arange(len(boot)), "alpha": boot, "crop": crop, "direction": direction, "sample_type": "B_fixed"}))
                except ValueError:
                    LOGGER.warning("Insufficient fixed sample for %s %s %s", crop, direction, window["label"])

    alpha = pd.DataFrame(all_rows)

    # Taylor specification curve remains on the tonne-scale estimand only.
    spec_rows = []
    for trend_name, degree in [("raw", None), ("linear", 1), ("quadratic", 2)]:
        metrics = compute_window_metrics(panels, 1986, 2024, degree=degree)
        for (crop, direction), group in metrics.groupby(["crop", "direction"], sort=False):
            context = {
                "crop": crop,
                "direction": direction,
                "window": "1986-2024",
                "window_id": "full_1986_2024",
                "window_label": "1986-2024",
                "window_type": "full",
                "start_year": 1986,
                "end_year": 2024,
                "sample_type": "A_dynamic",
                "trend": trend_name,
                "transform": "tonnes",
            }
            rows, _, _ = _fit_rows(group, context, n_boot=n_boot)
            spec_rows.extend(rows)
    specifications = pd.concat([alpha, pd.DataFrame(spec_rows)], ignore_index=True, sort=False).drop_duplicates(
        ["crop", "direction", "window_id", "sample_type", "trend", "transform", "method"], keep="last"
    )
    # A separate descriptive volatility table keeps log-flow variance outside
    # the Taylor exponent ranking and summary.
    log_metrics = compute_window_metrics(panels, 1986, 2024, degree=1, transform="log1p")
    log_metrics.to_parquet(DATA_DIR / "descriptive_log1p_volatility.parquet", index=False)
    save_csv(specifications, TABLE_DIR / "table2_alpha_estimates.csv")
    save_csv(specifications, FIG_DATA_DIR / "FigS12_specification_curve.csv")
    save_csv(alpha, DATA_DIR / "alpha_window_estimates.csv")
    full_fit = pd.concat(fitted_full, ignore_index=True)
    save_csv(full_fit, FIG_DATA_DIR / "Fig1_full_period_country_data.csv")
    save_csv(full_fit, FIG_DATA_DIR / "Fig3_cv_identity_country_data.csv")
    boot_full_df = pd.concat(boot_full, ignore_index=True)
    save_csv(boot_full_df, DATA_DIR / "full_period_alpha_bootstrap_draws.csv")

    # Full-period M1 versus M2 comparison.
    model_rows = []
    model_country_rows = []
    for (crop, direction), group in full_metrics.groupby(["crop", "direction"], sort=False):
        for sample_type, codes in [("A_dynamic", None), ("B_fixed", fixed.get((crop, direction), set()))]:
            use = group if codes is None else group[group["country_code"].isin(codes)]
            try:
                result, fitted = fit_model_comparison(use)
            except ValueError:
                continue
            result.update({"crop": crop, "direction": direction, "sample_type": sample_type, "window": "1986-2024"})
            model_rows.append(result)
            fitted["crop"] = crop
            fitted["direction"] = direction
            fitted["sample_type"] = sample_type
            fitted["M2_alpha_eff"] = m2_alpha_eff(fitted["mu"], result["M2_a"], result["M2_b"])
            model_country_rows.append(fitted)
    model_comparison = pd.DataFrame(model_rows)
    save_csv(model_comparison, TABLE_DIR / "table3_model_comparison.csv")
    save_csv(pd.concat(model_country_rows, ignore_index=True), FIG_DATA_DIR / "Fig1_model_fits.csv")

    # Import-export alpha differences from paired bootstrap draws by crop/sample.
    diff_rows = []
    for (crop, sample_type), group in boot_full_df.groupby(["crop", "sample_type"]):
        wide = group.pivot(index="draw", columns="direction", values="alpha").dropna()
        if {"Import", "Export"}.issubset(wide.columns):
            diff = wide["Import"] - wide["Export"]
            obs = specifications[
                (specifications["crop"] == crop)
                & (specifications["sample_type"] == sample_type)
                & (specifications["window"] == "1986-2024")
                & (specifications["trend"] == "linear")
                & (specifications["transform"] == "tonnes")
                & (specifications["method"] == "OLS_HC3")
            ].set_index("direction")["estimate"]
            diff_rows.append(
                {
                    "crop": crop,
                    "sample_type": sample_type,
                    "estimate": float(obs["Import"] - obs["Export"]),
                    "ci_low": float(diff.quantile(0.025)),
                    "ci_high": float(diff.quantile(0.975)),
                }
            )
    alpha_diff = pd.DataFrame(diff_rows)
    save_csv(alpha_diff, FIG_DATA_DIR / "Fig2_alpha_import_export_difference.csv")

    full_metrics.to_parquet(DATA_DIR / "country_metrics_full.parquet", index=False)
    pd.concat(rolling_metrics_out, ignore_index=True).to_parquet(DATA_DIR / "country_metrics_rolling20.parquet", index=False)
    save_csv(pd.DataFrame([{"crop": k[0], "direction": k[1], "country_code": c} for k, codes in fixed.items() for c in sorted(codes)]), DATA_DIR / "fixed_sample_countries.csv")

    table4 = distribution_table(full_metrics[full_metrics["eligible_covariance"]], ["D", "C", "phi", "C_over_D", "CovarianceShare"])
    save_csv(table4, TABLE_DIR / "table4_covariance_decomposition.csv")
    save_csv(full_metrics[full_metrics["eligible_covariance"]], FIG_DATA_DIR / "Fig4_covariance_country_data.csv")

    return {
        "alpha": specifications,
        "full_metrics": full_metrics,
        "model_comparison": model_comparison,
        "fixed": fixed,
        "metric_cache": metric_cache,
        "full_fit": full_fit,
    }


def distribution_table(data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for (crop, direction), group in data.groupby(["crop", "direction"], sort=False):
        for variable in variables:
            values = group[variable].dropna().to_numpy(float)
            if not len(values):
                continue
            rows.append(
                {
                    "crop": crop,
                    "direction": direction,
                    "variable": variable,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "p10": float(np.quantile(values, 0.10)),
                    "p25": float(np.quantile(values, 0.25)),
                    "median": float(np.median(values)),
                    "p75": float(np.quantile(values, 0.75)),
                    "p90": float(np.quantile(values, 0.90)),
                }
            )
    return pd.DataFrame(rows)


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(cov), np.finfo(float).tiny))
    corr = cov / np.outer(scale, scale)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def compute_diversification(
    panels: dict[tuple[str, str, int], TradePanel],
    start: int,
    end: int,
    share_threshold: float = 0.01,
    coverage_threshold: float = 0.80,
    matrix_method: str = "ledoit_wolf",
) -> pd.DataFrame:
    idx = np.array([YEARS.index(y) for y in range(start, end + 1)])
    rows = []
    for panel in panels.values():
        if panel.direction != "Import":
            continue
        values = panel.values[:, idx]
        totals = values.sum(axis=1)
        grand = float(totals.sum())
        if grand <= 0:
            continue
        shares = totals / grand
        positive = shares > 0
        s_all = shares[positive]
        n_nominal = 1.0 / float(np.sum(s_all * s_all))
        edge_resid = detrend_residuals(values, degree=1)
        edge_var = np.var(edge_resid, axis=1, ddof=1)
        positive_years = np.sum(values > 0, axis=1)
        selected = (shares >= share_threshold) & (positive_years >= 5) & (edge_var > 0)
        coverage = float(shares[selected].sum())
        n_corr = np.nan
        n_nominal_matched = np.nan
        gap = np.nan
        selected_count = int(selected.sum())
        if coverage >= coverage_threshold and selected_count >= 2:
            s = shares[selected]
            s = s / s.sum()
            residual = edge_resid[selected].T
            if matrix_method == "ledoit_wolf":
                cov = LedoitWolf().fit(residual).covariance_
                corr = _correlation_from_covariance(cov)
            elif matrix_method == "sample":
                corr = np.corrcoef(residual, rowvar=False)
            else:
                raise ValueError(matrix_method)
            denom = float(s @ corr @ s)
            if denom > 0:
                n_corr = 1.0 / denom
                n_nominal_matched = 1.0 / float(np.sum(s * s))
                gap = float(np.log(n_nominal_matched) - np.log(n_corr))
        ordered = np.sort(shares)[::-1]
        rows.append(
            {
                "crop": panel.crop,
                "country_code": panel.country_code,
                "country": panel.country_name,
                "start_year": start,
                "end_year": end,
                "window": f"{start}-{end}",
                "share_threshold": share_threshold,
                "coverage_threshold": coverage_threshold,
                "matrix_method": matrix_method,
                "partner_count": int(positive.sum()),
                "HHI": float(np.sum(s_all * s_all)),
                "N_nominal": n_nominal,
                "max_source_share": float(ordered[0]),
                "top3_source_share": float(ordered[:3].sum()),
                "selected_sources": selected_count,
                "selected_coverage": coverage,
                "N_nominal_matched": n_nominal_matched,
                "N_corr": n_corr,
                "Gap": gap,
            }
        )
    return pd.DataFrame(rows)


def run_diversification(panels: dict[tuple[str, str, int], TradePanel], full_metrics: pd.DataFrame) -> dict:
    main_frames = [compute_diversification(panels, 1986, 2024)]
    for start in range(1986, 2006):
        main_frames.append(compute_diversification(panels, start, start + 19))
    main = pd.concat(main_frames, ignore_index=True)
    full = main[main["window"] == "1986-2024"].copy()
    full = full.merge(
        full_metrics[(full_metrics["direction"] == "Import")][["crop", "country_code", "phi", "C_over_D", "eligible_covariance"]],
        on=["crop", "country_code"],
        how="left",
    )
    save_csv(full, FIG_DATA_DIR / "Fig5_diversification_full.csv")
    save_csv(main, FIG_DATA_DIR / "Fig5_diversification_rolling20.csv")

    sensitivity = []
    for share, coverage, method in product([0.005, 0.01, 0.02], [0.70, 0.80, 0.90], ["ledoit_wolf", "sample"]):
        sensitivity.append(compute_diversification(panels, 1986, 2024, share, coverage, method))
    sensitivity_df = pd.concat(sensitivity, ignore_index=True)
    save_csv(sensitivity_df, FIG_DATA_DIR / "FigS10_S11_diversification_sensitivity.csv")

    table5 = distribution_table(
        full.assign(direction="Import").dropna(subset=["N_corr"]),
        ["N_nominal", "N_nominal_matched", "N_corr", "Gap", "HHI", "max_source_share", "top3_source_share"],
    )
    save_csv(table5, TABLE_DIR / "table5_diversification.csv")
    return {"main": main, "full": full, "sensitivity": sensitivity_df}


def circular_shift_rows(residuals: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    shifts = np.asarray(shifts, dtype=int)
    return np.stack([np.roll(row, int(shift)) for row, shift in zip(residuals, shifts)], axis=0)


def _null_variances(residuals: np.ndarray, n_iter: int, rng: np.random.Generator, batch_size: int = 100) -> np.ndarray:
    p, t = residuals.shape
    out = np.empty(n_iter, dtype=float)
    base_t = np.arange(t, dtype=int)[None, None, :]
    for start in range(0, n_iter, batch_size):
        stop = min(start + batch_size, n_iter)
        b = stop - start
        shifts = rng.integers(0, t, size=(b, p))
        indices = (base_t - shifts[:, :, None]) % t
        expanded = np.broadcast_to(residuals[None, :, :], (b, p, t))
        shifted = np.take_along_axis(expanded, indices, axis=2)
        totals = shifted.sum(axis=1)
        out[start:stop] = np.var(totals, axis=1, ddof=1)
    return out


def run_null_model(
    panels: dict[tuple[str, str, int], TradePanel],
    full_metrics: pd.DataFrame,
    n_iter: int = 1000,
) -> dict:
    summary_rows = []
    draws = []
    idx = np.arange(len(YEARS))
    for (crop, direction), observed_group in full_metrics.groupby(["crop", "direction"], sort=False):
        eligible = observed_group[observed_group["eligible_taylor"]].copy().sort_values("country_code")
        country_codes = eligible["country_code"].astype(int).to_list()
        n_country = len(country_codes)
        null_v = np.full((n_iter, n_country), np.nan)
        d_values = np.full(n_country, np.nan)
        mu_values = eligible["mu"].to_numpy(float)
        rng = np.random.default_rng(stable_seed(f"null-{crop}-{direction}"))
        for col, country_code in enumerate(country_codes):
            panel = panels[(crop, direction, country_code)]
            residual = detrend_residuals(panel.values[:, idx], degree=1)
            nonzero_var = np.var(residual, axis=1, ddof=1) > 0
            residual = residual[nonzero_var]
            if residual.size == 0:
                continue
            d_values[col] = float(np.var(residual, axis=1, ddof=1).sum())
            null_v[:, col] = _null_variances(residual, n_iter, rng)

        valid = np.isfinite(d_values) & (d_values > 0) & (mu_values > 0)
        null_phi = null_v[:, valid] / d_values[valid][None, :]
        x = np.log(mu_values[valid])
        null_alpha = np.empty(n_iter)
        for b in range(n_iter):
            y = np.log(np.maximum(null_v[b, valid], np.finfo(float).tiny))
            xc = x - x.mean()
            null_alpha[b] = np.dot(xc, y - y.mean()) / np.dot(xc, xc)
        null_metrics = {
            "median_phi": np.nanmedian(null_phi, axis=1),
            "alpha": null_alpha,
        }
        obs_valid = eligible[valid]
        obs_scaling, _, _ = fit_scaling(obs_valid, n_boot=1000, seed=stable_seed(f"obs-{crop}-{direction}"))
        observed = {
            "median_phi": float(obs_valid["phi"].median()),
            "alpha": float(obs_scaling["alpha"]),
        }
        for metric, values in null_metrics.items():
            center = float(np.mean(values))
            obs = observed[metric]
            p = (1 + np.sum(np.abs(values - center) >= abs(obs - center))) / (len(values) + 1)
            sd = float(np.std(values, ddof=1))
            summary_rows.append(
                {
                    "crop": crop,
                    "direction": direction,
                    "metric": metric,
                    "observed": obs,
                    "null_mean": center,
                    "null_median": float(np.median(values)),
                    "null_q025": float(np.quantile(values, 0.025)),
                    "null_q975": float(np.quantile(values, 0.975)),
                    "empirical_p_value": float(p),
                    "z_score": float((obs - center) / sd) if sd > 0 else np.nan,
                    "n_iterations": n_iter,
                    "n_countries": int(valid.sum()),
                }
            )
            draws.append(pd.DataFrame({"crop": crop, "direction": direction, "metric": metric, "iteration": np.arange(n_iter), "value": values}))
        LOGGER.info("Completed null model for %s %s", crop, direction)
    summary = pd.DataFrame(summary_rows)
    draws_df = pd.concat(draws, ignore_index=True)
    save_csv(summary, TABLE_DIR / "table6_null_model.csv")
    save_csv(draws_df, FIG_DATA_DIR / "Fig6_null_distributions.csv")
    save_csv(summary, FIG_DATA_DIR / "Fig6_null_summary.csv")
    return {"summary": summary, "draws": draws_df}


def run_null_window_robustness(
    panels: dict[tuple[str, str, int], TradePanel],
    metric_cache: dict[str, pd.DataFrame],
    full_summary: pd.DataFrame,
    n_iter: int = 1000,
    force: bool = False,
) -> pd.DataFrame:
    """Run the circular-shift null for every unique analysis window.

    Full draws used in Figure 6 are retained separately; for the remaining
    windows this function stores the complete summary statistics to keep the
    output compact.
    """
    output = DATA_DIR / "null_model_all_windows.csv"
    if output.exists() and not force:
        cached = pd.read_csv(output)
        if len(cached) and int(cached["n_iterations"].min()) >= n_iter:
            LOGGER.info("Using cached all-window circular-shift summaries")
            return cached

    frames = [full_summary.assign(window="1986-2024", window_id="full_1986_2024", start_year=1986, end_year=2024)]
    window_ids = [window_id for window_id in metric_cache if window_id != "full_1986_2024"]
    for window_no, window_id in enumerate(window_ids, start=1):
        metrics = metric_cache[window_id]
        label = str(metrics["window_label"].iloc[0])
        start = int(metrics["start_year"].iloc[0])
        end = int(metrics["end_year"].iloc[0])
        idx = np.array([YEARS.index(y) for y in range(start, end + 1)])
        rows = []
        for (crop, direction), observed_group in metrics.groupby(["crop", "direction"], sort=False):
            eligible = observed_group[observed_group["eligible_taylor"]].copy().sort_values("country_code")
            country_codes = eligible["country_code"].astype(int).to_list()
            null_v = np.full((n_iter, len(country_codes)), np.nan)
            d_values = np.full(len(country_codes), np.nan)
            mu_values = eligible["mu"].to_numpy(float)
            rng = np.random.default_rng(stable_seed(f"null-window-{label}-{crop}-{direction}"))
            for col, country_code in enumerate(country_codes):
                panel = panels[(crop, direction, country_code)]
                residual = detrend_residuals(panel.values[:, idx], degree=1)
                residual = residual[np.var(residual, axis=1, ddof=1) > 0]
                if residual.size:
                    d_values[col] = float(np.var(residual, axis=1, ddof=1).sum())
                    null_v[:, col] = _null_variances(residual, n_iter, rng)
            valid = np.isfinite(d_values) & (d_values > 0) & (mu_values > 0)
            if valid.sum() < 4:
                continue
            null_phi = null_v[:, valid] / d_values[valid][None, :]
            x = np.log(mu_values[valid])
            xc = x - x.mean()
            denom = np.dot(xc, xc)
            null_alpha = np.array(
                [np.dot(xc, np.log(np.maximum(v, np.finfo(float).tiny)) - np.log(np.maximum(v, np.finfo(float).tiny)).mean()) / denom for v in null_v[:, valid]]
            )
            obs_valid = eligible.iloc[np.flatnonzero(valid)]
            obs_scaling, _, _ = fit_scaling(obs_valid, n_boot=20, seed=stable_seed(f"obs-window-{label}-{crop}-{direction}"))
            null_metrics = {
                "median_phi": np.nanmedian(null_phi, axis=1),
                "alpha": null_alpha,
            }
            observed = {
                "median_phi": float(obs_valid["phi"].median()),
                "alpha": float(obs_scaling["alpha"]),
            }
            for metric, values in null_metrics.items():
                center = float(np.mean(values))
                obs = observed[metric]
                sd = float(np.std(values, ddof=1))
                p = (1 + np.sum(np.abs(values - center) >= abs(obs - center))) / (len(values) + 1)
                rows.append(
                    {
                        "crop": crop,
                        "direction": direction,
                        "metric": metric,
                        "observed": obs,
                        "null_mean": center,
                        "null_median": float(np.median(values)),
                        "null_q025": float(np.quantile(values, 0.025)),
                        "null_q975": float(np.quantile(values, 0.975)),
                        "empirical_p_value": float(p),
                        "z_score": float((obs - center) / sd) if sd > 0 else np.nan,
                        "n_iterations": n_iter,
                        "n_countries": int(valid.sum()),
                        "window": label,
                        "window_id": window_id,
                        "start_year": start,
                        "end_year": end,
                    }
                )
        frames.append(pd.DataFrame(rows))
        LOGGER.info("Completed all-window null robustness for %s (%d/%d)", window_id, window_no, len(window_ids))
    result = pd.concat(frames, ignore_index=True)
    save_csv(result, output)
    return result


def run_rice_code31_sensitivity(rice31: pd.DataFrame, main_alpha: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rice31 = rice31.copy()
    rice31["crop"] = "Rice"
    panels = build_panels(rice31)
    metrics = compute_window_metrics(panels, 1986, 2024, degree=1)
    rows = []
    for direction, group in metrics.groupby("direction"):
        fit, _, _ = fit_scaling(group, n_boot=n_boot, seed=stable_seed(f"rice31-{direction}"))
        rows.append({"item_code": 31, "item_scope": "Rice, milled", "direction": direction, **fit})
    code31 = pd.DataFrame(rows)
    code30 = main_alpha[
        (main_alpha["crop"] == "Rice")
        & (main_alpha["window"] == "1986-2024")
        & (main_alpha["sample_type"] == "A_dynamic")
        & (main_alpha["trend"] == "linear")
        & (main_alpha["transform"] == "tonnes")
        & (main_alpha["method"] == "OLS_HC3")
    ][["direction", "estimate", "ci_low", "ci_high", "delta", "delta_ci_low", "delta_ci_high", "n_countries"]].rename(columns={"estimate": "alpha"})
    code30.insert(0, "item_scope", "Rice, paddy (rice milled equivalent)")
    code30.insert(0, "item_code", 30)
    result = pd.concat([code30, code31], ignore_index=True, sort=False)
    save_csv(result, DATA_DIR / "rice_code30_vs31_taylor_sensitivity.csv")
    save_csv(result, FIG_DATA_DIR / "FigS2_rice_code30_vs31_taylor_sensitivity.csv")
    return result


def covariance_direction_bootstrap(full_metrics: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(stable_seed("phi-direction-bootstrap"))
    rows = []
    for crop, group in full_metrics[full_metrics["eligible_covariance"]].groupby("crop"):
        imp = group[group["direction"] == "Import"]["phi"].dropna().to_numpy(float)
        exp = group[group["direction"] == "Export"]["phi"].dropna().to_numpy(float)
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            diffs[b] = np.median(rng.choice(exp, size=len(exp), replace=True)) - np.median(rng.choice(imp, size=len(imp), replace=True))
        rows.append(
            {
                "crop": crop,
                "estimate": float(np.median(exp) - np.median(imp)),
                "ci_low": float(np.quantile(diffs, 0.025)),
                "ci_high": float(np.quantile(diffs, 0.975)),
                "p_two_sided": float(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))),
                "n_import": len(imp),
                "n_export": len(exp),
            }
        )
    result = pd.DataFrame(rows)
    save_csv(result, FIG_DATA_DIR / "Fig4_phi_direction_bootstrap.csv")
    return result


def edge_mean_variance_data(panels: dict[tuple[str, str, int], TradePanel]) -> pd.DataFrame:
    rows = []
    for panel in panels.values():
        residual = detrend_residuals(panel.values, degree=1)
        means = panel.values.mean(axis=1)
        variances = np.var(residual, axis=1, ddof=1)
        for code, name, mean, variance in zip(panel.partner_codes, panel.partner_names, means, variances):
            if mean > 0 and variance > 0:
                rows.append(
                    {
                        "crop": panel.crop,
                        "direction": panel.direction,
                        "reporter_code": panel.country_code,
                        "reporter": panel.country_name,
                        "partner_code": int(code),
                        "partner": name,
                        "mean": float(mean),
                        "variance": float(variance),
                    }
                )
    result = pd.DataFrame(rows)
    save_csv(result, FIG_DATA_DIR / "FigS9_edge_mean_variance.csv")
    return result
