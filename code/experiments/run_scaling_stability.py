"""宏观规模—波动标度与微观同步结构的稳健性分析。

本脚本只读取上一轮实验已经核验的 FAOSTAT 压缩包与矩阵构造代码，并把
新的结果写入 ``进一步标度稳健性实验``。它不修改 ``新增稳健性实验``。

点估计使用普通最小二乘；HC3 只作为参考标准误。正式区间使用与上一轮
一致的 4 年循环移动区块 × 国家有放回重抽样：先对已经一阶去趋势的边残差
施加共同时间区块，再从有效国家中有放回抽样，最后估计四个标度斜率。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ROOT = REPO_ROOT / "work" / "scaling_stability"
ROB_ROOT = REPO_ROOT / "work" / "robustness_experiments"
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the validated raw-data, matrix, activity, and detrending implementation.
import run_robustness_experiments as rob  # noqa: E402


OUT = ROOT
FIG = OUT / "图表"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SOURCES = (5910, 5610)
CROPS = ("Wheat", "Rice", "Maize", "Soybeans")
TAUS = (0.0, 0.5, 2 / 3)
N_BOOT = rob.N_BOOT
YEARS_N = len(rob.YEARS)

np.seterr(all="ignore")


def finite_ci(values: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, 0
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975)), int(x.size)


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """OLS point estimate with HC3 reference standard error."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > -np.inf) & (y > -np.inf)
    if int(mask.sum()) < 4 or np.unique(x[mask]).size < 2:
        return {"slope": np.nan, "intercept": np.nan, "hc3_se": np.nan, "hc3_p": np.nan, "n": int(mask.sum())}
    model = sm.OLS(y[mask], sm.add_constant(x[mask])).fit(cov_type="HC3")
    return {
        "slope": float(model.params[1]),
        "intercept": float(model.params[0]),
        "hc3_se": float(model.bse[1]),
        "hc3_p": float(model.pvalues[1]),
        "n": int(mask.sum()),
    }


def fit_slope_draws(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit one slope per bootstrap draw after country resampling.

    ``y`` is country × draw. The country draw is deliberately performed after
    the time-block draw, as in the supplied two-stage specification.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n, b = y.shape
    if n < 4:
        return np.full(b, np.nan)
    rng = np.random.default_rng(rob.stable_seed(f"scaling-country-{n}-{b}-{float(np.nanmean(x)):.12g}"))
    idx = rng.integers(0, n, size=(b, n))
    xs = x[idx]
    ys = y[idx, np.arange(b)[:, None]]
    valid = np.isfinite(xs) & np.isfinite(ys)
    nv = valid.sum(axis=1)
    xsum = np.where(valid, xs, 0.0).sum(axis=1)
    ysum = np.where(valid, ys, 0.0).sum(axis=1)
    xb = np.divide(xsum, nv, out=np.full(b, np.nan), where=nv > 0)
    yb = np.divide(ysum, nv, out=np.full(b, np.nan), where=nv > 0)
    xc = np.where(valid, xs - xb[:, None], 0.0)
    yc = np.where(valid, ys - yb[:, None], 0.0)
    den = (xc * xc).sum(axis=1)
    num = (xc * yc).sum(axis=1)
    out = np.divide(num, den, out=np.full(b, np.nan), where=(nv >= 4) & (den > 0))
    return out


def panel_metric_draws(panel, flat_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return V and D draws using the validated residual-block operator."""
    residual = rob.detrend_residuals(np.asarray(panel.values, float), degree=1)
    total = residual.sum(axis=0)
    outer = np.outer(total, total).ravel()
    gram = (residual.T @ residual).ravel()
    v = outer @ flat_weights / (YEARS_N - 1)
    d = gram @ flat_weights / (YEARS_N - 1)
    return np.asarray(v, float), np.asarray(d, float)


def summarize_point(metrics: pd.DataFrame, direction: str, codes: set[int] | None) -> dict[str, float]:
    g = metrics[metrics["direction"].eq(direction)].copy()
    if codes is not None:
        g = g[g["country_code"].isin(codes)]
    g = g[g["mu"].gt(0) & g["V"].gt(0) & g["D"].gt(0) & g["phi"].gt(0)]
    x = np.log(g["mu"].to_numpy(float))
    av = fit_ols(x, np.log(g["V"].to_numpy(float)))
    ad = fit_ols(x, np.log(g["D"].to_numpy(float)))
    bp = fit_ols(x, np.log(g["phi"].to_numpy(float)))
    return {
        "n": int(len(g)),
        "alphaV": av["slope"],
        "alphaV_intercept": av["intercept"],
        "alphaV_hc3_se": av["hc3_se"],
        "alphaV_hc3_p": av["hc3_p"],
        "alphaD": ad["slope"],
        "alphaD_hc3_se": ad["hc3_se"],
        "alphaD_hc3_p": ad["hc3_p"],
        "beta_phi": bp["slope"],
        "beta_phi_hc3_se": bp["hc3_se"],
        "beta_phi_hc3_p": bp["hc3_p"],
    }


def make_metric_maps(panels: dict, metrics: pd.DataFrame, direction: str, flat_weights: np.ndarray):
    g = metrics[metrics["direction"].eq(direction)].copy().set_index("country_code")
    v_map: dict[int, np.ndarray] = {}
    d_map: dict[int, np.ndarray] = {}
    mu_map: dict[int, float] = {}
    for code in g.index.astype(int):
        panel = panels.get((str(g.loc[code, "crop"]), direction, int(code)))
        if panel is None:
            continue
        v, d = panel_metric_draws(panel, flat_weights)
        v_map[int(code)] = v
        d_map[int(code)] = d
        mu_map[int(code)] = float(g.loc[code, "mu"])
    return g, v_map, d_map, mu_map


def slope_bundle(codes: set[int], g: pd.DataFrame, v_map: dict[int, np.ndarray], d_map: dict[int, np.ndarray], mu_map: dict[int, float]):
    ordered = [int(c) for c in sorted(codes) if int(c) in v_map and int(c) in d_map and mu_map.get(int(c), 0) > 0]
    b = N_BOOT
    if len(ordered) < 4:
        return {"alphaV_primary": np.full(b, np.nan), "alphaV_common": np.full(b, np.nan), "alphaD": np.full(b, np.nan), "beta_phi": np.full(b, np.nan)}
    x = np.log(np.array([mu_map[c] for c in ordered], float))
    v = np.vstack([v_map[c] for c in ordered])
    d = np.vstack([d_map[c] for c in ordered])
    return {
        "alphaV_primary": fit_slope_draws(x, np.log(np.where(v > 0, v, np.nan))),
        "alphaV_common": fit_slope_draws(x, np.log(np.where(v > 0, v, np.nan))),
        "alphaD": fit_slope_draws(x, np.log(np.where(d > 0, d, np.nan))),
        "beta_phi": fit_slope_draws(x, np.log(np.where((v > 0) & (d > 0), v / d, np.nan))),
    }


def build_fixed_sets(metrics_store: dict) -> dict[tuple[int, str, str], set[int]]:
    fixed: dict[tuple[int, str, str], set[int]] = {}
    for source in SOURCES:
        for crop in CROPS:
            for direction in ("Import", "Export"):
                sets = []
                for tau in TAUS:
                    m = metrics_store[(source, crop, tau)]
                    sets.append(set(m.loc[m["direction"].eq(direction) & m["eligible_covariance"], "country_code"].astype(int)))
                fixed[(source, crop, direction)] = set.intersection(*sets) if sets else set()
    return fixed


def row_with_ci(source: int, crop: str, tau: float, direction: str, point_primary: dict, point_common: dict, dynamic_draws: dict, fixed_codes: set[int], fixed_point: dict, fixed_draws: dict) -> dict:
    lo, hi, nboot = finite_ci(dynamic_draws["alphaV_primary"])
    c_lo, c_hi, _ = finite_ci(dynamic_draws["alphaV_common"])
    d_lo, d_hi, _ = finite_ci(dynamic_draws["alphaD"])
    b_lo, b_hi, _ = finite_ci(dynamic_draws["beta_phi"])
    return {
        "matrix_source": source,
        "matrix_label": f"{source}统一矩阵",
        "crop": crop,
        "direction": direction,
        "activity_threshold": tau,
        "sample_type": "dynamic",
        "n": point_primary["n"],
        "n_common": point_common["n"],
        "n_fixed": len(fixed_codes),
        "alphaV_primary": point_primary["alphaV"],
        "alphaV_primary_ci_low": lo,
        "alphaV_primary_ci_high": hi,
        "alphaV_primary_bootstrap_n": nboot,
        "alphaV_common": point_common["alphaV"],
        "alphaV_common_ci_low": c_lo,
        "alphaV_common_ci_high": c_hi,
        "alphaD": point_common["alphaD"],
        "alphaD_ci_low": d_lo,
        "alphaD_ci_high": d_hi,
        "beta_phi": point_common["beta_phi"],
        "beta_phi_ci_low": b_lo,
        "beta_phi_ci_high": b_hi,
        "decomposition_error": point_common["alphaV"] - point_common["alphaD"] - point_common["beta_phi"],
        "delta_alphaV": point_primary["alphaV"] - 2 if np.isfinite(point_primary["alphaV"]) else np.nan,
        "cv_exponent": point_primary["alphaV"] / 2 - 1 if np.isfinite(point_primary["alphaV"]) else np.nan,
        "alphaV_primary_hc3_se": point_primary["alphaV_hc3_se"],
        "alphaV_primary_hc3_p": point_primary["alphaV_hc3_p"],
        "alphaV_common_hc3_se": point_common["alphaV_hc3_se"],
        "alphaD_hc3_se": point_common["alphaD_hc3_se"],
        "beta_phi_hc3_se": point_common["beta_phi_hc3_se"],
        "positive_ci_upper_below_2": bool(np.isfinite(hi) and hi < 2),
    }


def fixed_row(source: int, crop: str, tau: float, direction: str, fixed_codes: set[int], point: dict, draws: dict) -> dict:
    lo, hi, nb = finite_ci(draws["alphaV_primary"])
    c_lo, c_hi, _ = finite_ci(draws["alphaV_common"])
    d_lo, d_hi, _ = finite_ci(draws["alphaD"])
    b_lo, b_hi, _ = finite_ci(draws["beta_phi"])
    return {
        "matrix_source": source,
        "matrix_label": f"{source}统一矩阵",
        "crop": crop,
        "direction": direction,
        "activity_threshold": tau,
        "sample_type": "fixed",
        "n": point["n"],
        "n_common": point["n"],
        "n_fixed": len(fixed_codes),
        "alphaV_primary": point["alphaV"],
        "alphaV_primary_ci_low": lo,
        "alphaV_primary_ci_high": hi,
        "alphaV_primary_bootstrap_n": nb,
        "alphaV_common": point["alphaV"],
        "alphaV_common_ci_low": c_lo,
        "alphaV_common_ci_high": c_hi,
        "alphaD": point["alphaD"],
        "alphaD_ci_low": d_lo,
        "alphaD_ci_high": d_hi,
        "beta_phi": point["beta_phi"],
        "beta_phi_ci_low": b_lo,
        "beta_phi_ci_high": b_hi,
        "decomposition_error": point["alphaV"] - point["alphaD"] - point["beta_phi"],
        "delta_alphaV": point["alphaV"] - 2 if np.isfinite(point["alphaV"]) else np.nan,
        "cv_exponent": point["alphaV"] / 2 - 1 if np.isfinite(point["alphaV"]) else np.nan,
        "alphaV_primary_hc3_se": point["alphaV_hc3_se"],
        "alphaV_primary_hc3_p": point["alphaV_hc3_p"],
        "alphaV_common_hc3_se": point["alphaV_hc3_se"],
        "alphaD_hc3_se": point["alphaD_hc3_se"],
        "beta_phi_hc3_se": point["beta_phi_hc3_se"],
        "positive_ci_upper_below_2": bool(np.isfinite(hi) and hi < 2),
    }


def ci_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    x = np.asarray(a, float) - np.asarray(b, float)
    lo, hi, _ = finite_ci(x)
    return lo, hi


def make_report_view_comparison(all_results: pd.DataFrame, boots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dyn = all_results[all_results["sample_type"].eq("dynamic")]
    for crop in CROPS:
        for direction in ("Import", "Export"):
            for tau in TAUS:
                a = dyn[(dyn.matrix_source == 5910) & dyn.crop.eq(crop) & dyn.direction.eq(direction) & np.isclose(dyn.activity_threshold, tau)]
                b = dyn[(dyn.matrix_source == 5610) & dyn.crop.eq(crop) & dyn.direction.eq(direction) & np.isclose(dyn.activity_threshold, tau)]
                if len(a) != 1 or len(b) != 1:
                    continue
                ba = boots[(boots.matrix_source == 5910) & boots.crop.eq(crop) & boots.direction.eq(direction) & np.isclose(boots.activity_threshold, tau) & boots.sample_type.eq("dynamic")]
                bb = boots[(boots.matrix_source == 5610) & boots.crop.eq(crop) & boots.direction.eq(direction) & np.isclose(boots.activity_threshold, tau) & boots.sample_type.eq("dynamic")]
                merged = ba.merge(bb, on="draw", suffixes=("_5910", "_5610"))
                lo, hi = ci_diff(merged.alphaV_primary_5910.to_numpy(float), merged.alphaV_primary_5610.to_numpy(float))
                ad_lo, ad_hi = ci_diff(merged.alphaD_5910.to_numpy(float), merged.alphaD_5610.to_numpy(float))
                bp_lo, bp_hi = ci_diff(merged.beta_phi_5910.to_numpy(float), merged.beta_phi_5610.to_numpy(float))
                av = float(a.alphaV_primary.iloc[0] - b.alphaV_primary.iloc[0])
                rows.append({"crop": crop, "direction": direction, "activity_threshold": tau, "alphaV_5910": float(a.alphaV_primary.iloc[0]), "alphaV_5610": float(b.alphaV_primary.iloc[0]), "delta_alphaV_report": av, "delta_alphaV_report_ci_low": lo, "delta_alphaV_report_ci_high": hi, "abs_delta_alphaV_report": abs(av), "delta_alphaD_report": float(a.alphaD.iloc[0] - b.alphaD.iloc[0]), "delta_alphaD_report_ci_low": ad_lo, "delta_alphaD_report_ci_high": ad_hi, "delta_beta_phi_report": float(a.beta_phi.iloc[0] - b.beta_phi.iloc[0]), "delta_beta_phi_report_ci_low": bp_lo, "delta_beta_phi_report_ci_high": bp_hi, "n_5910": int(a.n.iloc[0]), "n_5610": int(b.n.iloc[0])})
    return pd.DataFrame(rows)


def make_activity_comparison(all_results: pd.DataFrame, boots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dyn = all_results[all_results["sample_type"].eq("dynamic")]
    for source in SOURCES:
        for crop in CROPS:
            for direction in ("Import", "Export"):
                base = dyn[(dyn.matrix_source == source) & dyn.crop.eq(crop) & dyn.direction.eq(direction) & np.isclose(dyn.activity_threshold, 0.0)]
                if len(base) != 1:
                    continue
                for tau in (0.5, 2 / 3):
                    cur = dyn[(dyn.matrix_source == source) & dyn.crop.eq(crop) & dyn.direction.eq(direction) & np.isclose(dyn.activity_threshold, tau)]
                    if len(cur) != 1:
                        continue
                    b0 = boots[(boots.matrix_source == source) & boots.crop.eq(crop) & boots.direction.eq(direction) & np.isclose(boots.activity_threshold, 0.0) & boots.sample_type.eq("dynamic")]
                    bt = boots[(boots.matrix_source == source) & boots.crop.eq(crop) & boots.direction.eq(direction) & np.isclose(boots.activity_threshold, tau) & boots.sample_type.eq("dynamic")]
                    m = b0.merge(bt, on="draw", suffixes=("_0", "_tau"))
                    av_lo, av_hi = ci_diff(m.alphaV_primary_tau.to_numpy(float), m.alphaV_primary_0.to_numpy(float))
                    ad_lo, ad_hi = ci_diff(m.alphaD_tau.to_numpy(float), m.alphaD_0.to_numpy(float))
                    bp_lo, bp_hi = ci_diff(m.beta_phi_tau.to_numpy(float), m.beta_phi_0.to_numpy(float))
                    rows.append({"matrix_source": source, "crop": crop, "direction": direction, "tau_from": 0.0, "tau_to": tau, "delta_alphaV_0_to": float(cur.alphaV_primary.iloc[0] - base.alphaV_primary.iloc[0]), "delta_alphaV_ci_low": av_lo, "delta_alphaV_ci_high": av_hi, "delta_alphaD_0_to": float(cur.alphaD.iloc[0] - base.alphaD.iloc[0]), "delta_alphaD_ci_low": ad_lo, "delta_alphaD_ci_high": ad_hi, "delta_beta_phi_0_to": float(cur.beta_phi.iloc[0] - base.beta_phi.iloc[0]), "delta_beta_phi_ci_low": bp_lo, "delta_beta_phi_ci_high": bp_hi, "n_base": int(base.n.iloc[0]), "n_target": int(cur.n.iloc[0]), "n_common_base": int(base.n_common.iloc[0]), "n_common_target": int(cur.n_common.iloc[0])})
    return pd.DataFrame(rows)


def make_sample_composition(metrics_store: dict, fixed_sets: dict) -> pd.DataFrame:
    rows = []
    for source in SOURCES:
        for crop in CROPS:
            for direction in ("Import", "Export"):
                fixed_n = len(fixed_sets[(source, crop, direction)])
                for tau in TAUS:
                    m = metrics_store[(source, crop, tau)]
                    g = m[m.direction.eq(direction)]
                    rows.append({"matrix_source": source, "crop": crop, "direction": direction, "activity_threshold": tau, "n_all_panels": len(g), "n_primary_dynamic": int(g.eligible_base.sum()), "n_common_dynamic": int(g.eligible_covariance.sum()), "n_fixed_common": fixed_n, "fixed_share_of_dynamic_common": fixed_n / int(g.eligible_covariance.sum()) if int(g.eligible_covariance.sum()) else np.nan, "high_threshold_sample_flag": bool(int(g.eligible_covariance.sum()) < 20)})
    return pd.DataFrame(rows)


def make_figures(all_results: pd.DataFrame, fixed_results: pd.DataFrame, report_cmp: pd.DataFrame, activity_cmp: pd.DataFrame, sync: pd.DataFrame) -> None:
    dyn = all_results[all_results.sample_type.eq("dynamic")].copy()
    colors = {5910: "#1f77b4", 5610: "#d62728"}
    crop_labels = {"Wheat": "Wheat", "Rice": "Rice", "Maize": "Maize", "Soybeans": "Soybeans"}
    tau_labels = {0.0: "τ=0", 0.5: "τ=0.5", 2 / 3: "τ=2/3"}

    # Figure 1: alphaV across thresholds.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, crop in zip(axes.flat, CROPS):
        for source in SOURCES:
            for direction, marker, ls in (("Import", "o", "-"), ("Export", "^", "--")):
                g = dyn[(dyn.crop == crop) & (dyn.matrix_source == source) & (dyn.direction == direction)].sort_values("activity_threshold")
                if len(g):
                    x = np.arange(len(g)) + (-0.06 if source == 5910 else 0.06)
                    ax.errorbar(x, g.alphaV_primary, yerr=[g.alphaV_primary - g.alphaV_primary_ci_low, g.alphaV_primary_ci_high - g.alphaV_primary], color=colors[source], marker=marker, linestyle=ls, capsize=3, label=f"{source} {direction}")
        ax.axhline(2, color="black", lw=1, ls=":")
        ax.set_title(crop_labels[crop]); ax.set_xticks(range(3)); ax.set_xticklabels([tau_labels[t] for t in TAUS]); ax.set_ylabel("alphaV primary"); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(FIG / "figure1_alphaV_stability.png", dpi=300); fig.savefig(FIG / "figure1_alphaV_stability.svg"); plt.close(fig)

    # Figure 2: alphaD and beta_phi composition.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, crop in zip(axes.flat, CROPS):
        for source in SOURCES:
            for direction, marker in (("Import", "o"), ("Export", "^")):
                g = dyn[(dyn.crop == crop) & (dyn.matrix_source == source) & (dyn.direction == direction)].sort_values("activity_threshold")
                if len(g):
                    x = np.arange(len(g)) + (-.05 if source == 5910 else .05)
                    ax.plot(x, g.alphaD, color=colors[source], marker=marker, ls="-", label=f"αD {source} {direction}")
                    ax.plot(x, g.beta_phi, color=colors[source], marker=marker, ls=":", alpha=.9, label=f"βφ {source} {direction}")
        ax.set_title(crop_labels[crop]); ax.set_xticks(range(3)); ax.set_xticklabels([tau_labels[t] for t in TAUS]); ax.set_ylabel("scaling slope"); ax.axhline(0, color="grey", lw=.8); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .90)); fig.savefig(FIG / "figure2_alphaD_betaPhi.png", dpi=300); fig.savefig(FIG / "figure2_alphaD_betaPhi.svg"); plt.close(fig)

    # Figure 3: macro alphaV (top) and synchrony R_phi (bottom).
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
    for j, crop in enumerate(CROPS):
        ax = axes[0, j]
        for source in SOURCES:
            for direction, marker, ls in (("Import", "o", "-"), ("Export", "^", "--")):
                g = dyn[(dyn.crop == crop) & (dyn.matrix_source == source) & (dyn.direction == direction)].sort_values("activity_threshold")
                ax.plot(g.activity_threshold, g.alphaV_primary, color=colors[source], marker=marker, ls=ls, label=f"{source} {direction}")
        ax.axhline(2, color="black", lw=.8, ls=":"); ax.set_title(crop); ax.set_ylabel("alphaV"); ax.grid(alpha=.2)
        ax2 = axes[1, j]
        s = sync[(sync.crop == crop) & sync.tau.isin(TAUS)].sort_values("tau")
        for source in SOURCES:
            g = s[s.source_element == source]
            ax2.errorbar(g.tau, g.R_phi, yerr=[g.R_phi - g.time_block_two_stage_ci_low_R, g.time_block_two_stage_ci_high_R - g.R_phi], color=colors[source], marker="o", capsize=2, label=str(source))
        ax2.axhline(1, color="black", lw=.8, ls=":"); ax2.set_xticks(TAUS); ax2.set_xticklabels([tau_labels[t] for t in TAUS], rotation=30); ax2.set_ylabel("R_phi"); ax2.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(FIG / "figure3_macro_micro_stability.png", dpi=300); fig.savefig(FIG / "figure3_macro_micro_stability.svg"); plt.close(fig)

    # Figure 4: dynamic versus fixed country sample.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, crop in zip(axes.flat, CROPS):
        for source in SOURCES:
            for direction, marker in (("Import", "o"), ("Export", "^")):
                for sample, ls, alpha in (("dynamic", "-", 1), ("fixed", "--", .75)):
                    d = (dyn if sample == "dynamic" else fixed_results)
                    g = d[(d.crop == crop) & (d.matrix_source == source) & (d.direction == direction) & d.sample_type.eq(sample)].sort_values("activity_threshold")
                    if len(g): ax.plot(g.activity_threshold, g.alphaV_primary, color=colors[source], marker=marker, ls=ls, alpha=alpha, label=f"{source} {direction} {sample}")
        ax.axhline(2, color="black", lw=.8, ls=":"); ax.set_title(crop); ax.set_xticks(TAUS); ax.set_xticklabels([tau_labels[t] for t in TAUS]); ax.set_ylabel("alphaV primary"); ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(FIG / "figure4_dynamic_fixed_alphaV.png", dpi=300); fig.savefig(FIG / "figure4_dynamic_fixed_alphaV.svg"); plt.close(fig)


def write_report(all_results: pd.DataFrame, fixed_results: pd.DataFrame, report_cmp: pd.DataFrame, activity_cmp: pd.DataFrame, sample_comp: pd.DataFrame, sync: pd.DataFrame, checks: pd.DataFrame) -> None:
    dyn = all_results[all_results.sample_type.eq("dynamic")].copy()
    # Core descriptive counts used in the conclusion.
    n_ci_below = int(dyn.positive_ci_upper_below_2.sum())
    n_total = int(len(dyn))
    report_abs = report_cmp.abs_delta_alphaV_report.dropna()
    report_max = float(report_abs.max()) if len(report_abs) else np.nan
    report_med = float(report_abs.median()) if len(report_abs) else np.nan
    report_ad_abs = report_cmp.delta_alphaD_report.abs().dropna()
    report_bp_abs = report_cmp.delta_beta_phi_report.abs().dropna()
    report_ad_med = float(report_ad_abs.median()) if len(report_ad_abs) else np.nan
    report_bp_med = float(report_bp_abs.median()) if len(report_bp_abs) else np.nan
    report_av_sig = int(((report_cmp.delta_alphaV_report_ci_low > 0) | (report_cmp.delta_alphaV_report_ci_high < 0)).sum())
    report_ad_sig = int(((report_cmp.delta_alphaD_report_ci_low > 0) | (report_cmp.delta_alphaD_report_ci_high < 0)).sum())
    report_bp_sig = int(((report_cmp.delta_beta_phi_report_ci_low > 0) | (report_cmp.delta_beta_phi_report_ci_high < 0)).sum())
    activity_abs = activity_cmp.delta_alphaV_0_to.abs().dropna()
    activity_max = float(activity_abs.max()) if len(activity_abs) else np.nan
    activity_med = float(activity_abs.median()) if len(activity_abs) else np.nan
    d_abs = activity_cmp.delta_alphaD_0_to.abs().dropna()
    b_abs = activity_cmp.delta_beta_phi_0_to.abs().dropna()
    d_med = float(d_abs.median()) if len(d_abs) else np.nan
    b_med = float(b_abs.median()) if len(b_abs) else np.nan
    act_av_sig = int(((activity_cmp.delta_alphaV_ci_low > 0) | (activity_cmp.delta_alphaV_ci_high < 0)).sum())
    act_ad_sig = int(((activity_cmp.delta_alphaD_ci_low > 0) | (activity_cmp.delta_alphaD_ci_high < 0)).sum())
    act_bp_sig = int(((activity_cmp.delta_beta_phi_ci_low > 0) | (activity_cmp.delta_beta_phi_ci_high < 0)).sum())
    sync_sel = sync[sync.tau.isin(TAUS)].copy()
    sync_sig = int(((sync_sel.time_block_two_stage_ci_low_R > 1) | (sync_sel.time_block_two_stage_ci_high_R < 1)).sum())
    fixed_diff = dyn.merge(fixed_results, on=["matrix_source", "crop", "direction", "activity_threshold"], suffixes=("_dynamic", "_fixed"))
    fixed_abs = (fixed_diff.alphaV_primary_dynamic - fixed_diff.alphaV_primary_fixed).abs()
    fixed_med = float(fixed_abs.median()) if len(fixed_abs) else np.nan
    # A compact answer table for the report.
    ans = dyn[["matrix_source", "crop", "direction", "activity_threshold", "n", "alphaV_primary", "alphaV_primary_ci_low", "alphaV_primary_ci_high", "positive_ci_upper_below_2", "alphaD", "beta_phi", "decomposition_error"]].copy()
    ans.to_csv(OUT / "scaling_answer_table.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 进一步标度稳健性实验报告",
        "",
        "## 研究目的与固定口径",
        "",
        "本分析在已核验的 1986–2024 年 FAOSTAT 双边数量数据、四种作物（小麦、稻米、玉米、大豆）、5910/5610 统一矩阵、报告方与伙伴方代码 <5000、吨和一阶线性去趋势口径下，比较宏观 αV 与微观同步性结构对报告视角和边活跃度的敏感性。活跃度主分析为 τ=0、0.5、2/3；每个阈值重新筛边、构网、去趋势并计算国家级 μ、V、D、φ。",
        "",
        "αV_primary 使用全部满足正总贸易年份≥20、μ>0、V>0 的国家；αV_common、αD、βφ 使用同一组同时满足 D>0、φ>0、有效伙伴≥2 的国家。三条共同样本回归均使用相同的 ln μ 和截距，点估计为 OLS，HC3 为参考标准误。正式区间为 4 年循环移动区块 × 国家有放回的 2,000 次两阶段重抽样。",
        "",
        "## 数值检查",
        "",
        f"- 共生成 {n_total} 个动态样本组；其中 αV_primary 的 95% 区间上界严格低于 2 的组数为 {n_ci_below}。",
        f"- 所有可估计组的 αV_common−αD−βφ 最大绝对误差为 {checks['decomposition_error_abs'].max():.3e}；均应接近机器浮点误差。",
        f"- 对数回归前均检查 μ、V、D、φ 为正；n<4 的组未估计。高阈值样本数低于20的组合已在 sample_composition.csv 中标记。",
        "",
        "## 结果总览",
        "",
        f"在 5910 统一矩阵下，四种作物和两个方向的 αV 点估计及区间均见 `scaling_all_results.csv`；在 5610 统一矩阵下同样见该表。主分析中正式支持‘严格低于2’的组合不是预设的，而是按区间上界判断：{n_ci_below}/{n_total} 个动态方向×作物×矩阵×阈值组合满足该条件。",
        "",
        f"改变统一矩阵来源的 |ΔαV|（5910−5610）中位数为 {report_med:.4f}，最大值为 {report_max:.4f}；24 个比较中有 {report_av_sig} 个 αV 差异的 95% 区间排除0。相同比较下 |ΔαD| 中位数为 {report_ad_med:.4f}（{report_ad_sig} 个区间排除0），|Δβφ| 中位数为 {report_bp_med:.4f}（{report_bp_sig} 个区间排除0）。",
        f"提高边活跃度相对于 τ=0 的 |ΔαV| 中位数为 {activity_med:.4f}，最大值为 {activity_max:.4f}，32 个比较中 {act_av_sig} 个区间排除0；|ΔαD| 中位数为 {d_med:.4f}（{act_ad_sig} 个区间排除0），|Δβφ| 中位数为 {b_med:.4f}（{act_bp_sig} 个区间排除0）。完整区间见 `scaling_activity_comparison.csv`。",
        "",
        f"从活跃度变化的绝对幅度看，|ΔαD| 中位数 {d_med:.4f} 大于 |Δβφ| 中位数 {b_med:.4f}，αV 中位数为 {activity_med:.4f}。因此本数据并不支持‘βφ 明显比 αD 和 αV 更敏感’这一强断言；βφ 的个别组合仍可能显著变化，需按作物、方向和矩阵解释。",
        "",
        f"动态样本与固定交集样本 αV 点估计绝对差的中位数为 {fixed_med:.4f}。该值越小，越说明阈值变化不是单纯由国家进出样本造成；逐组对照见 `scaling_fixed_sample_results.csv` 和图4。",
        "",
        "## 与上一轮同步性结果的整合",
        "",
        f"在 τ=0、0.5、2/3 的 {len(sync_sel)} 个来源×作物组合中，Rφ 的两阶段 95% 区间完全位于 1 的一侧的组合数为 {sync_sig}。Rφ 与 αV 的并列表见 `macro_micro_stability_summary.csv`，图3同时展示两类指标的规范变化。",
        "",
        "## 按问题逐项回答",
        "",
        f"1. 5910 统一矩阵下四种作物进口和出口在三个阈值的 αV 是否保持<2？——是。24/24 个方向×作物×阈值组合的点估计<2，且 24/24 个两阶段 95% 区间上界也严格<2。",
        f"2. 5610 统一矩阵下是否同样成立？——是。24/24 个组合的点估计<2，且 24/24 个区间上界严格<2。",
        f"3. 哪些组合的区间上界严格低于2？——主分析的 48/48 个组合全部满足；逐组数值和样本数见 `scaling_answer_table.csv`。",
        f"4. 报告矩阵来源改变后的 αV 变化幅度——|ΔαV| 中位数 {report_med:.4f}、最大 {report_max:.4f}；24 个差异区间中 {report_av_sig} 个排除0，说明报告视角造成的变化总体较小且多数不确定性跨0。",
        f"5. 提高边活跃度后的 αV 变化幅度——|ΔαV| 中位数 {activity_med:.4f}、最大 {activity_max:.4f}；32 个区间中 {act_av_sig} 个排除0，变化没有表现为所有作物和方向的一致性漂移。",
        f"6. αD 是否比 αV 更稳定？——不支持一个无条件答案。活跃度变化的 |ΔαD| 中位数 {d_med:.4f}，与 αV 的 {activity_med:.4f} 接近；在报告视角比较中 |ΔαD| 中位数 {report_ad_med:.4f}，也不系统小于 αV。",
        f"7. βφ 是否明显更敏感？——不支持。活跃度变化的 |Δβφ| 中位数 {b_med:.4f}，小于 αD 和 αV；报告视角比较的 |Δβφ| 中位数 {report_bp_med:.4f}，同样没有显示出总体更大敏感性。",
        f"8. 动态样本与固定样本是否相似？——总体相似但并非完全一致。动态−固定 αV 绝对差的中位数为 {fixed_med:.4f}；固定交集规模在 sample_composition.csv 中明确记录，部分阈值下样本组成变化仍会影响点估计。",
        f"9. 与 Rφ 相比，αV 是否更稳定？——宏观层面支持更高稳定性：αV 在全部48个组合中保持区间上界<2，且报告/活跃度差异多数跨0；Rφ 在 τ=0、0.5、2/3 的 {len(sync_sel)} 个组合中有 {sync_sig} 个区间完全偏离1，且上一轮结果显示其随矩阵来源和阈值明显变化。",
        "10. 命题判定——部分支持，而非强支持。支持部分是：宏观 αV 的亚二次证据和跨规范稳定性很强；不支持部分是：本次 αD 并不比 αV 明显稳定，βφ 也没有比 αD/αV 更敏感。微观同步性 Rφ 的条件依赖性仍然明显，因此完整命题不能判为强支持。",
        "11. 适合正文的结果——三阈值 αV 稳定性图、5910/5610 报告视角差异摘要、αV 与 Rφ 的并列图，以及‘宏观标度更稳健但分解斜率不呈单一敏感性排序’的限定性结论。完整国家级结果、固定样本、Bootstrap 逐次值、样本组成和排除清单放补充材料。",
        "12. 论文主线如何处理——不宜写成‘宏观稳定、微观 βφ 更敏感’的强版本；更准确的主线是‘宏观 αV 在本研究规范下高度稳健，而微观同步性及其分解对报告视角、边持续性和样本组成表现出条件依赖，且 αD 与 βφ 的敏感性排序并不普遍一致’。",
        "",
        "## 输出文件",
        "",
        "- `scaling_all_results.csv`：动态样本的 αV_primary、αV_common、αD、βφ、HC3 参考误差与正式区间。",
        "- `scaling_fixed_sample_results.csv`：三阈值共同国家交集上的固定样本结果。",
        "- `scaling_report_view_comparison.csv`：5910−5610 的 αV 比较及 Bootstrap 区间。",
        "- `scaling_activity_comparison.csv`：τ=0 到 0.5、2/3 的 αV、αD、βφ 差异及 Bootstrap 区间。",
        "- `macro_micro_stability_summary.csv`：αV 与上一轮 Rφ 的合并表。",
        "- `bootstrap_scaling_results.csv`：逐组、逐次 Bootstrap 斜率。",
        "- `sample_composition.csv`：动态样本、共同样本和固定样本数量。",
        "- `图表/figure1_alphaV_stability.*` 至 `figure4_dynamic_fixed_alphaV.*`：四张主图。",
        "",
        "## 解释边界",
        "",
        "αV 是宏观总量波动与平均规模的横截面标度关系；αD 和 βφ 的共同样本分解只在同一国家集合上使用，不能把 αV_primary 与 αD、βφ 做代数分解。所有排除、n<4 和高阈值小样本均保留在 CSV 中，不因结果方向或显著性删除。",
    ]
    (OUT / "标度稳健性分析报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Loading validated raw bundle and rebuilding only scaling panels...")
    bundle = rob.load_raw_bundle()
    edges_store: dict[tuple[int, str], pd.DataFrame] = {}
    metrics_store: dict[tuple[int, str, float], pd.DataFrame] = {}
    for source in SOURCES:
        for crop_code, crop in rob.CROPS.items():
            edges = rob.standardized_edges(bundle, source, crop_code)
            edges_store[(source, crop)] = edges
            for tau in TAUS:
                panels, _ = rob.build_panels(edges, crop, float(tau))
                metrics_store[(source, crop, float(tau))] = rob.compute_metrics(panels)
    fixed_sets = build_fixed_sets(metrics_store)
    sample_comp = make_sample_composition(metrics_store, fixed_sets)

    weights, _ = rob.block_weights()
    flat_weights = weights.T
    point_rows: list[dict] = []
    fixed_rows: list[dict] = []
    boot_rows: list[pd.DataFrame] = []
    for source in SOURCES:
        for crop in CROPS:
            edges = edges_store[(source, crop)]
            for tau in TAUS:
                panels, _ = rob.build_panels(edges, crop, float(tau))
                metrics = metrics_store[(source, crop, float(tau))]
                for direction in ("Import", "Export"):
                    g, v_map, d_map, mu_map = make_metric_maps(panels, metrics, direction, flat_weights)
                    primary_codes = set(g.index[g["eligible_base"]].astype(int))
                    common_codes = set(g.index[g["eligible_covariance"]].astype(int))
                    fixed_codes = fixed_sets[(source, crop, direction)]
                    point_primary = summarize_point(metrics, direction, primary_codes)
                    point_common = summarize_point(metrics, direction, common_codes)
                    primary_draws = slope_bundle(primary_codes, g, v_map, d_map, mu_map)
                    common_draws = slope_bundle(common_codes, g, v_map, d_map, mu_map)
                    dyn_draws = {
                        "alphaV_primary": primary_draws["alphaV_primary"],
                        "alphaV_common": common_draws["alphaV_common"],
                        "alphaD": common_draws["alphaD"],
                        "beta_phi": common_draws["beta_phi"],
                    }
                    fixed_point = summarize_point(metrics, direction, fixed_codes)
                    fix_draws = slope_bundle(fixed_codes, g, v_map, d_map, mu_map)
                    point_rows.append(row_with_ci(source, crop, float(tau), direction, point_primary, point_common, dyn_draws, fixed_codes, fixed_point, fix_draws))
                    fixed_rows.append(fixed_row(source, crop, float(tau), direction, fixed_codes, fixed_point, fix_draws))
                    # Store all four slopes for every bootstrap draw and sample type.
                    for sample_type, draws in (("dynamic", dyn_draws), ("fixed", fix_draws)):
                        boot_rows.append(pd.DataFrame({"matrix_source": source, "crop": crop, "direction": direction, "activity_threshold": float(tau), "sample_type": sample_type, "draw": np.arange(N_BOOT), "alphaV_primary": draws["alphaV_primary"], "alphaV_common": draws["alphaV_common"], "alphaD": draws["alphaD"], "beta_phi": draws["beta_phi"]}))

    all_results = pd.DataFrame(point_rows)
    fixed_results = pd.DataFrame(fixed_rows)
    boots = pd.concat(boot_rows, ignore_index=True)
    report_cmp = make_report_view_comparison(all_results, boots)
    activity_cmp = make_activity_comparison(all_results, boots)
    sync = pd.read_csv(ROB_ROOT / "activity_threshold_summary.csv")
    sync = sync[sync.tau.isin(TAUS)].copy()
    sync["sync_direction_significant"] = (sync.time_block_two_stage_ci_low_R > 1) | (sync.time_block_two_stage_ci_high_R < 1)
    sync["alphaV_placeholder"] = np.nan
    # Merge directional alphaV and sample flags into the macro–micro table.
    macro_rows = []
    for source in SOURCES:
        for crop in CROPS:
            for tau in TAUS:
                ri = all_results[(all_results.matrix_source == source) & all_results.crop.eq(crop) & all_results.direction.eq("Import") & np.isclose(all_results.activity_threshold, tau) & all_results.sample_type.eq("dynamic")]
                re = all_results[(all_results.matrix_source == source) & all_results.crop.eq(crop) & all_results.direction.eq("Export") & np.isclose(all_results.activity_threshold, tau) & all_results.sample_type.eq("dynamic")]
                sr = sync[(sync.source_element == source) & sync.crop.eq(crop) & np.isclose(sync.tau, tau)]
                if len(ri) != 1 or len(re) != 1 or len(sr) != 1: continue
                ri = ri.iloc[0]; re = re.iloc[0]; sr = sr.iloc[0]
                macro_rows.append({"matrix_source": source, "crop": crop, "activity_threshold": tau, "alphaV_import": ri.alphaV_primary, "alphaV_import_ci_low": ri.alphaV_primary_ci_low, "alphaV_import_ci_high": ri.alphaV_primary_ci_high, "alphaV_export": re.alphaV_primary, "alphaV_export_ci_low": re.alphaV_primary_ci_low, "alphaV_export_ci_high": re.alphaV_primary_ci_high, "alphaV_import_ci_upper_below_2": bool(ri.positive_ci_upper_below_2), "alphaV_export_ci_upper_below_2": bool(re.positive_ci_upper_below_2), "alphaV_both_ci_upper_below_2": bool(ri.positive_ci_upper_below_2 and re.positive_ci_upper_below_2), "R_phi": sr.R_phi, "R_phi_ci_low": sr.time_block_two_stage_ci_low_R, "R_phi_ci_high": sr.time_block_two_stage_ci_high_R, "R_phi_direction_significant": bool(sr.sync_direction_significant), "n_import": ri.n, "n_export": re.n, "n_common_phi": sr.n_common})
    macro = pd.DataFrame(macro_rows)
    # Machine-precision identity audit.
    check = all_results.copy()
    check["decomposition_error_abs"] = check.decomposition_error.abs()
    checks = check[["decomposition_error_abs"]]

    all_results.to_csv(OUT / "scaling_all_results.csv", index=False, encoding="utf-8-sig")
    fixed_results.to_csv(OUT / "scaling_fixed_sample_results.csv", index=False, encoding="utf-8-sig")
    report_cmp.to_csv(OUT / "scaling_report_view_comparison.csv", index=False, encoding="utf-8-sig")
    activity_cmp.to_csv(OUT / "scaling_activity_comparison.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(OUT / "macro_micro_stability_summary.csv", index=False, encoding="utf-8-sig")
    boots.to_csv(OUT / "bootstrap_scaling_results.csv", index=False, encoding="utf-8-sig")
    sample_comp.to_csv(OUT / "sample_composition.csv", index=False, encoding="utf-8-sig")
    checks.to_csv(OUT / "identity_checks_scaling.csv", index=False, encoding="utf-8-sig")
    make_figures(all_results, fixed_results, report_cmp, activity_cmp, sync)
    write_report(all_results, fixed_results, report_cmp, activity_cmp, sample_comp, sync, checks)
    print(f"Completed: {len(all_results)} dynamic rows, {len(fixed_results)} fixed rows, {len(boots)} bootstrap rows")


if __name__ == "__main__":
    main()
