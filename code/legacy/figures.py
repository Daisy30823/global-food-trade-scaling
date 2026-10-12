from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis_runner import stable_seed
from .config import DATA_DIR, FIG_DATA_DIR, FIG_MAIN_DIR, FIG_SUPP_DIR
from .statistics import m2_alpha_eff
from .utils import get_logger, save_csv

LOGGER = get_logger()
CROP_ORDER = ["Wheat", "Rice", "Maize", "Soybeans"]
DIRECTION_ORDER = ["Import", "Export"]
COLORS = {"Import": "#2A6FBB", "Export": "#D07A22", "A_dynamic": "#2A6FBB", "B_fixed": "#B04A6A"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bootstrap_band(data: pd.DataFrame, xgrid: np.ndarray, n_boot: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    x = data["log_mu"].to_numpy(float)
    y = data["log_variance"].to_numpy(float)
    rng = np.random.default_rng(stable_seed(f"band-{data.iloc[0]['crop']}-{data.iloc[0]['direction']}"))
    predictions = np.empty((n_boot, len(xgrid)))
    for b in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        xb, yb = x[idx], y[idx]
        xc = xb - xb.mean()
        slope = np.dot(xc, yb - yb.mean()) / np.dot(xc, xc)
        intercept = yb.mean() - slope * xb.mean()
        predictions[b] = intercept + slope * xgrid
    return np.quantile(predictions, 0.025, axis=0), np.quantile(predictions, 0.975, axis=0)


def figure1(alpha: pd.DataFrame, full_fit: pd.DataFrame, model_comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=False, sharey=False)
    band_rows = []
    model_data = pd.read_csv(FIG_DATA_DIR / "Fig1_model_fits.csv")
    for r, direction in enumerate(DIRECTION_ORDER):
        for c, crop in enumerate(CROP_ORDER):
            ax = axes[r, c]
            data = full_fit[(full_fit["crop"] == crop) & (full_fit["direction"] == direction) & (full_fit["sample_type"] == "A_dynamic")]
            fit_data = model_data[(model_data["crop"] == crop) & (model_data["direction"] == direction) & (model_data["sample_type"] == "A_dynamic")]
            comp = model_comparison[(model_comparison["crop"] == crop) & (model_comparison["direction"] == direction) & (model_comparison["sample_type"] == "A_dynamic")].iloc[0]
            est = alpha[(alpha["crop"] == crop) & (alpha["direction"] == direction) & (alpha["sample_type"] == "A_dynamic") & (alpha["window"] == "1986-2024") & (alpha["trend"] == "linear") & (alpha["transform"] == "tonnes") & (alpha["method"] == "OLS_HC3")].iloc[0]
            x = data["log_mu"].to_numpy() / np.log(10)
            y = data["log_variance"].to_numpy() / np.log(10)
            ax.scatter(x, y, s=15, alpha=0.6, color=COLORS[direction], edgecolor="white", linewidth=0.3)
            fit_log_mu = np.log(fit_data["mu"].to_numpy(float))
            order = np.argsort(fit_log_mu)
            xf = fit_log_mu[order] / np.log(10)
            ax.plot(xf, fit_data["M1_fitted_log_variance"].to_numpy()[order] / np.log(10), color="#222222", lw=1.5, label="M1 power law")
            ax.plot(xf, fit_data["M2_fitted_log_variance"].to_numpy()[order] / np.log(10), color="#8A5A00", lw=1.5, ls="--", label="M2 linear–quadratic")
            xgrid_ln = np.linspace(data["log_mu"].min(), data["log_mu"].max(), 100)
            lo, hi = _bootstrap_band(data, xgrid_ln)
            ax.fill_between(xgrid_ln / np.log(10), lo / np.log(10), hi / np.log(10), color="#666666", alpha=0.12, lw=0)
            for xv, lov, hiv in zip(xgrid_ln, lo, hi):
                band_rows.append({"crop": crop, "direction": direction, "log_mu": xv, "low": lov, "high": hiv})
            labels = data.nlargest(3, "cooks_distance")
            for _, row in labels.iterrows():
                ax.annotate(row["country"], (row["log_mu"] / np.log(10), row["log_variance"] / np.log(10)), xytext=(3, 3), textcoords="offset points", fontsize=6)
            ax.set_title(crop)
            ax.text(
                0.03,
                0.97,
                f"{direction}\nα={est['estimate']:.2f} [{est['ci_low']:.2f}, {est['ci_high']:.2f}]\n"
                f"n={int(est['n_countries'])}; CV-RMSE M1={comp['M1_cv_rmse']:.2f}, M2={comp['M2_cv_rmse']:.2f}\nBetter: {comp['better_model_cv_rmse']}",
                transform=ax.transAxes,
                va="top",
                fontsize=7,
                bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none"},
            )
            ax.set_xlabel(r"$\log_{10}(\mu_i)$")
            ax.set_ylabel(r"$\log_{10}(V_i)$")
            if r == 0 and c == 3:
                ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Full-period Taylor scaling, 1986–2024", fontsize=14, y=1.01)
    fig.tight_layout()
    save_csv(pd.DataFrame(band_rows), FIG_DATA_DIR / "Fig1_bootstrap_bands.csv")
    save_figure(fig, FIG_MAIN_DIR, "Fig1_full_period_taylor_scaling")


def figure2(alpha: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    main = alpha[(alpha["method"] == "OLS_HC3") & (alpha["trend"] == "linear") & (alpha["transform"] == "tonnes")]
    full = main[(main["window"] == "1986-2024") & (main["sample_type"] == "A_dynamic")].copy()
    full["label"] = full["crop"] + " " + full["direction"]
    full = full.sort_values(["crop", "direction"], key=lambda s: s.map({**{c: i for i, c in enumerate(CROP_ORDER)}, **{"Import": 0, "Export": 1}}))
    y = np.arange(len(full))
    axes[0, 0].errorbar(full["estimate"], y, xerr=[full["estimate"] - full["ci_low"], full["ci_high"] - full["estimate"]], fmt="o", color="#333333", ecolor="#777777")
    axes[0, 0].set_yticks(y, full["label"])
    axes[0, 0].axvline(1, color="#999999", ls=":")
    axes[0, 0].axvline(2, color="#555555", ls="--")
    axes[0, 0].set_title("A. Full-period alpha (dynamic sample)")
    axes[0, 0].set_xlabel("Taylor exponent α")

    rolling = main[main["window_type"] == "rolling20"].copy()
    for ax, direction, title in [(axes[0, 1], "Import", "B. Import: 20-year rolling windows"), (axes[1, 0], "Export", "C. Export: 20-year rolling windows")]:
        subset = rolling[rolling["direction"] == direction]
        for crop in CROP_ORDER:
            for sample, ls in [("A_dynamic", "-"), ("B_fixed", "--")]:
                d = subset[(subset["crop"] == crop) & (subset["sample_type"] == sample)].sort_values("end_year")
                if len(d):
                    ax.plot(d["end_year"], d["estimate"], marker="o", ms=2.5, lw=1.1, ls=ls, label=f"{crop} {sample[0]}")
        ax.axhline(1, color="#999999", ls=":", lw=0.8)
        ax.axhline(2, color="#555555", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("Window end year")
        ax.set_ylabel("α")
    axes[0, 1].legend(ncol=2, fontsize=6, frameon=False)

    diff = pd.read_csv(FIG_DATA_DIR / "Fig2_alpha_import_export_difference.csv")
    diff = diff[diff["sample_type"] == "A_dynamic"].set_index("crop").reindex(CROP_ORDER).reset_index()
    x = np.arange(len(diff))
    axes[1, 1].errorbar(x, diff["estimate"], yerr=[diff["estimate"] - diff["ci_low"], diff["ci_high"] - diff["estimate"]], fmt="o", color="#6B3FA0")
    axes[1, 1].axhline(0, color="#555555", lw=0.9)
    axes[1, 1].set_xticks(x, diff["crop"], rotation=20)
    axes[1, 1].set_title("D. α import − α export")
    axes[1, 1].set_ylabel("Difference with bootstrap 95% CI")
    fig.suptitle("Taylor exponent estimates and temporal evolution", fontsize=14)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig2_alpha_evolution")


def figure3(alpha: pd.DataFrame, full_fit: pd.DataFrame, model_comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, crop in zip(axes.flat[:4], CROP_ORDER):
        for direction in DIRECTION_ORDER:
            d = full_fit[(full_fit["crop"] == crop) & (full_fit["direction"] == direction) & (full_fit["sample_type"] == "A_dynamic")]
            ax.scatter(d["log_mu"] / np.log(10), d["log_cv2"] / np.log(10), s=14, alpha=0.55, color=COLORS[direction], label=direction)
            x = d["log_mu"].to_numpy()
            y = d["log_cv2"].to_numpy()
            slope = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
            intercept = y.mean() - slope * x.mean()
            grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(grid / np.log(10), (intercept + slope * grid) / np.log(10), color=COLORS[direction], lw=1.2)
            est = alpha[(alpha["crop"] == crop) & (alpha["direction"] == direction) & (alpha["sample_type"] == "A_dynamic") & (alpha["window"] == "1986-2024") & (alpha["trend"] == "linear") & (alpha["transform"] == "tonnes") & (alpha["method"] == "OLS_HC3")].iloc[0]
            ax.text(0.03, 0.97 - 0.12 * DIRECTION_ORDER.index(direction), f"{direction} δ={est['delta']:.2f} [{est['delta_ci_low']:.2f}, {est['delta_ci_high']:.2f}]", transform=ax.transAxes, va="top", color=COLORS[direction], fontsize=7)
        ax.set_title(crop)
        ax.set_xlabel(r"$\log_{10}(\mu_i)$")
        ax.set_ylabel(r"$\log_{10}(CV_i^2)$")
    axes[0, 0].legend(frameon=False)

    identity = alpha[(alpha["window"] == "1986-2024") & (alpha["sample_type"] == "A_dynamic") & (alpha["trend"] == "linear") & (alpha["transform"] == "tonnes") & (alpha["method"] == "OLS_HC3")]
    axes[1, 1].scatter(identity["estimate"], 2 + identity["delta"], c=[COLORS[d] for d in identity["direction"]], s=45)
    lim = [min(identity["estimate"].min(), (2 + identity["delta"]).min()) - 0.05, max(identity["estimate"].max(), (2 + identity["delta"]).max()) + 0.05]
    axes[1, 1].plot(lim, lim, color="#333333", ls="--")
    axes[1, 1].set_xlim(lim); axes[1, 1].set_ylim(lim)
    axes[1, 1].set_xlabel("Directly estimated α")
    axes[1, 1].set_ylabel("2 + δ")
    axes[1, 1].set_title("E. Identity check")

    quantiles = np.linspace(0.02, 0.98, 60)
    for _, row in model_comparison[model_comparison["sample_type"] == "A_dynamic"].iterrows():
        d = full_fit[(full_fit["crop"] == row["crop"]) & (full_fit["direction"] == row["direction"]) & (full_fit["sample_type"] == "A_dynamic")]
        muq = np.quantile(d["mu"], quantiles)
        eff = m2_alpha_eff(muq, row["M2_a"], row["M2_b"])
        axes[1, 2].plot(quantiles, eff, color=COLORS[row["direction"]], alpha=0.65, lw=1.2, label=f"{row['crop']} {row['direction']}")
    axes[1, 2].axhline(1, color="#999999", ls=":")
    axes[1, 2].axhline(2, color="#555555", ls="--")
    axes[1, 2].set_xlabel("Trade-scale quantile")
    axes[1, 2].set_ylabel(r"$\alpha_{eff}(\mu)$")
    axes[1, 2].set_title("F. M2 local effective exponent")
    axes[1, 2].legend(fontsize=5, ncol=2, frameon=False)
    fig.suptitle("Why the Taylor exponent does not exceed two", fontsize=14)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig3_cv_scaling_identity")


def _binned_summary(data: pd.DataFrame, x: str, y: str, bins: int = 8) -> pd.DataFrame:
    d = data[[x, y]].dropna().copy()
    d["bin"] = pd.qcut(d[x], q=min(bins, d[x].nunique()), duplicates="drop")
    return d.groupby("bin", observed=True).agg(x=(x, "median"), y=(y, "median"), lo=(y, lambda s: s.quantile(0.25)), hi=(y, lambda s: s.quantile(0.75))).reset_index(drop=True)


def figure4(full_metrics: pd.DataFrame, phi_boot: pd.DataFrame) -> None:
    data = full_metrics[full_metrics["eligible_covariance"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.violinplot(data=data, x="crop", y="phi", hue="direction", order=CROP_ORDER, hue_order=DIRECTION_ORDER, inner="quart", cut=0, palette=COLORS, ax=axes[0, 0])
    axes[0, 0].axhline(1, color="#555555", lw=0.8)
    axes[0, 0].set_title("A. Distribution of synchronization amplification φ")
    axes[0, 0].set_ylim(bottom=max(-0.1, data["phi"].quantile(0.01)))
    for direction in DIRECTION_ORDER:
        d = data[data["direction"] == direction]
        axes[0, 1].scatter(np.log10(d["mu"]), d["C_over_D"], s=10, alpha=0.18, color=COLORS[direction])
        b = _binned_summary(d.assign(log_mu=np.log10(d["mu"])), "log_mu", "C_over_D")
        axes[0, 1].plot(b["x"], b["y"], marker="o", color=COLORS[direction], label=direction)
        axes[0, 1].fill_between(b["x"], b["lo"], b["hi"], color=COLORS[direction], alpha=0.12)
    axes[0, 1].axhline(0, color="#555555", lw=0.8)
    axes[0, 1].set_xlabel(r"$\log_{10}(\mu_i)$")
    axes[0, 1].set_ylabel("C / D = φ − 1")
    axes[0, 1].set_title("B. Covariance contribution by trade scale")
    axes[0, 1].legend(frameon=False)

    prop = data.assign(category=np.where(data["phi"] < 0.9, "φ < 0.9", np.where(data["phi"] <= 1.1, "0.9 ≤ φ ≤ 1.1", "φ > 1.1"))).groupby(["crop", "direction", "category"], observed=True).size().rename("n").reset_index()
    prop["share"] = prop["n"] / prop.groupby(["crop", "direction"])["n"].transform("sum")
    piv = prop.pivot_table(index=["crop", "direction"], columns="category", values="share", fill_value=0).reindex(pd.MultiIndex.from_product([CROP_ORDER, DIRECTION_ORDER]))
    piv.plot(kind="bar", stacked=True, color=["#4E79A7", "#BAB0AC", "#E15759"], ax=axes[1, 0])
    axes[1, 0].set_title("C. Countries below, near, or above φ = 1")
    axes[1, 0].set_ylabel("Country share")
    axes[1, 0].tick_params(axis="x", rotation=35)
    axes[1, 0].legend(fontsize=6, frameon=False)
    save_csv(prop, FIG_DATA_DIR / "Fig4_phi_categories.csv")

    pb = phi_boot.set_index("crop").reindex(CROP_ORDER).reset_index()
    x = np.arange(len(pb))
    axes[1, 1].errorbar(x, pb["estimate"], yerr=[pb["estimate"] - pb["ci_low"], pb["ci_high"] - pb["estimate"]], fmt="o", color="#6B3FA0")
    axes[1, 1].axhline(0, color="#555555", lw=0.8)
    axes[1, 1].set_xticks(x, pb["crop"], rotation=20)
    axes[1, 1].set_ylabel("median(φ export) − median(φ import)")
    axes[1, 1].set_title("D. Directional asymmetry, bootstrap 95% CI")
    fig.suptitle("Variance decomposition and directional asymmetry", fontsize=14)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig4_covariance_decomposition")


def _bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 1000) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    draws = np.array([np.median(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)])
    return float(np.median(values)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def figure5(div_full: pd.DataFrame, div_rolling: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    d = div_full.dropna(subset=["N_corr", "N_nominal_matched"])
    for crop in CROP_ORDER:
        x = d[d["crop"] == crop]
        axes[0, 0].scatter(x["N_nominal_matched"], x["N_corr"], s=16, alpha=0.5, label=crop)
    lim = [0.8, max(d["N_nominal_matched"].max(), d["N_corr"].max()) * 1.05]
    axes[0, 0].plot(lim, lim, color="#333333", ls="--")
    axes[0, 0].set_xlim(lim); axes[0, 0].set_ylim(lim)
    axes[0, 0].set_xlabel("Matched nominal effective suppliers")
    axes[0, 0].set_ylabel("Correlation-adjusted effective suppliers")
    axes[0, 0].set_title("A. Nominal versus dynamic diversification")
    axes[0, 0].legend(frameon=False)
    sns.boxplot(data=d, x="crop", y="Gap", order=CROP_ORDER, color="#6BAED6", ax=axes[0, 1])
    axes[0, 1].axhline(0, color="#555555", lw=0.8)
    axes[0, 1].set_title("B. Dynamic diversification gap")

    p = div_full[div_full["eligible_covariance"].fillna(False)].dropna(subset=["phi"])
    axes[1, 0].scatter(p["N_nominal"], p["phi"], s=12, alpha=0.25, color="#2A6FBB")
    b = _binned_summary(p, "N_nominal", "phi")
    axes[1, 0].plot(b["x"], b["y"], marker="o", color="#D07A22")
    axes[1, 0].axhline(1, color="#555555", lw=0.8)
    axes[1, 0].set_xlabel("Nominal effective suppliers")
    axes[1, 0].set_ylabel("φ")
    axes[1, 0].set_title("C. Supplier count and synchronization")

    rng = np.random.default_rng(stable_seed("diversification-rolling-ci"))
    med_rows = []
    for (crop, window), group in div_rolling.groupby(["crop", "window"]):
        for variable in ["N_nominal", "N_corr"]:
            values = group[variable].dropna().to_numpy(float)
            if len(values):
                median, lo, hi = _bootstrap_median_ci(values, rng)
                med_rows.append({"crop": crop, "window": window, "end_year": int(window[-4:]), "variable": variable, "median": median, "low": lo, "high": hi})
    med = pd.DataFrame(med_rows)
    save_csv(med, FIG_DATA_DIR / "Fig5_rolling_medians.csv")
    for variable, ls in [("N_nominal", "-"), ("N_corr", "--")]:
        pooled = med.groupby(["end_year", "variable"], as_index=False)[["median", "low", "high"]].mean()
        q = pooled[pooled["variable"] == variable]
        axes[1, 1].plot(q["end_year"], q["median"], ls=ls, lw=1.7, label=variable)
        axes[1, 1].fill_between(q["end_year"], q["low"], q["high"], alpha=0.12)
    axes[1, 1].set_title("D. Median diversification in rolling windows")
    axes[1, 1].set_xlabel("Window end year")
    axes[1, 1].set_ylabel("Effective supplier count")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Nominal versus correlation-adjusted import diversification", fontsize=14)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig5_effective_diversification")


def figure6(null_summary: pd.DataFrame, null_draws: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("median_phi", "A. Median φ"), ("median_C_over_D", "B. Median C/D"), ("alpha", "C. Taylor exponent α")]
    for ax, (metric, title) in zip(axes.flat[:3], panels):
        order = [f"{c}\n{direction}" for c in CROP_ORDER for direction in DIRECTION_ORDER]
        s = null_summary[null_summary["metric"] == metric].copy()
        s["group"] = s["crop"] + "\n" + s["direction"]
        s = s.set_index("group").reindex(order).reset_index()
        x = np.arange(len(order))
        ax.errorbar(
            x,
            s["null_median"],
            yerr=[s["null_median"] - s["null_q025"], s["null_q975"] - s["null_median"]],
            fmt="o",
            color="#4C78A8",
            ecolor="#8BA6C1",
            capsize=3,
            label="Null median and 95% interval",
        )
        ax.scatter(x, s["observed"], color="#B22222", marker="D", s=28, label="Observed", zorder=5)
        for xi, row in zip(x, s.itertuples()):
            ax.annotate(f"p={row.empirical_p_value:.3f}", (xi, row.observed), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=5.5, rotation=45)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_xticks(x, order, rotation=45, ha="right")
        if metric == "median_phi":
            ax.axhline(1, color="#777777", lw=0.7, ls=":")
        elif metric == "median_C_over_D":
            ax.axhline(0, color="#777777", lw=0.7, ls=":")
        ax.legend(frameon=False, fontsize=6)
    s = null_summary.copy()
    s["label"] = s["crop"] + " " + s["direction"] + " " + s["metric"]
    s = s.sort_values("z_score")
    axes[1, 1].barh(s["label"], s["z_score"], color=np.where(s["z_score"] >= 0, "#D07A22", "#2A6FBB"))
    axes[1, 1].axvline(0, color="#555555", lw=0.8)
    axes[1, 1].set_xlabel("Observed-minus-null z-score")
    axes[1, 1].set_title("D. Standardized observed–null difference")
    axes[1, 1].tick_params(axis="y", labelsize=5.5)
    fig.suptitle("Circular-shift temporal synchronization null model", fontsize=14)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig6_temporal_null_model")


def supplemental_figures(core: pd.DataFrame, alpha: pd.DataFrame, full_metrics: pd.DataFrame, edge_data: pd.DataFrame, div_sens: pd.DataFrame) -> None:
    # S1 mirror diagnostics.
    mirror = pd.read_csv(DATA_DIR / "annual_global_import_export_mirror.csv")
    fig, axes = plt.subplots(4, 2, figsize=(11, 12))
    for i, crop in enumerate(CROP_ORDER):
        d = mirror[mirror["crop"] == crop]
        axes[i, 0].plot(d["year"], d["Import"] / 1e6, label="Import")
        axes[i, 0].plot(d["year"], d["Export"] / 1e6, label="Export")
        axes[i, 0].set_title(crop); axes[i, 0].set_ylabel("Million tonnes")
        axes[i, 1].plot(d["year"], d["export_import_ratio"], color="#6B3FA0")
        axes[i, 1].axhline(1, color="#555555", ls="--")
        axes[i, 1].set_title(f"{crop}: export/import ratio")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS1_global_mirror_diagnostics")

    # S2 rice-code diagnostics.
    rice = pd.read_csv(FIG_DATA_DIR / "FigS2_rice_codes_annual.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, direction in zip(axes, [5610, 5910]):
        for code, d in rice[rice["Element Code"] == direction].groupby("Item Code"):
            ax.plot(d["year"], d["quantity_tonnes"] / 1e6, label=f"{code}: {d.iloc[0]['Item']}")
        ax.set_title("Import" if direction == 5610 else "Export")
        ax.set_xlabel("Year"); ax.set_ylabel("Million tonnes")
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS2_rice_item_codes")

    # S3 coverage.
    coverage = core.groupby(["crop", "direction", "year"]).agg(countries=("Reporter Country Code", lambda s: s[core.loc[s.index, "quantity_tonnes"] > 0].nunique()), edges=("quantity_tonnes", lambda s: int((s > 0).sum())), coverage=("quantity_tonnes", lambda s: float((s > 0).mean()))).reset_index()
    save_csv(coverage, FIG_DATA_DIR / "FigS3_coverage.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for (crop, direction), d in coverage.groupby(["crop", "direction"]):
        label = f"{crop} {direction}"
        axes[0].plot(d["year"], d["countries"], label=label)
        axes[1].plot(d["year"], d["edges"])
        axes[2].plot(d["year"], d["coverage"])
    axes[0].set_title("Countries with positive trade"); axes[1].set_title("Positive bilateral edges"); axes[2].set_title("Positive edge share")
    axes[0].legend(fontsize=5, ncol=2, frameon=False)
    fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS3_data_coverage")

    main = alpha[alpha["method"] == "OLS_HC3"].copy()
    # S4 trend handling.
    d = main[(main["window"] == "1986-2024") & (main["sample_type"] == "A_dynamic") & (main["transform"] == "tonnes")]
    fig, ax = plt.subplots(figsize=(11, 5)); sns.pointplot(data=d, x="crop", y="estimate", hue="trend", dodge=0.35, ax=ax); ax.axhline(2, color="#555555", ls="--"); ax.set_ylabel("α"); ax.set_title("Trend treatment sensitivity"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS4_trend_robustness")
    # S5 window lengths.
    d = main[(main["sample_type"] == "A_dynamic") & (main["trend"] == "linear") & (main["transform"] == "tonnes") & (main["window_type"].isin(["rolling15", "rolling20", "rolling25"]))]
    fig, ax = plt.subplots(figsize=(11, 5)); sns.lineplot(data=d, x="end_year", y="estimate", hue="window_type", style="direction", estimator="median", errorbar=None, ax=ax); ax.axhline(2, color="#555555", ls="--"); ax.set_title("15-, 20-, and 25-year rolling-window estimates"); ax.set_ylabel("Median α across crops"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS5_window_length_robustness")
    # S6 samples.
    d = main[(main["window_type"] == "rolling20") & (main["trend"] == "linear") & (main["transform"] == "tonnes")]
    fig, ax = plt.subplots(figsize=(11, 5)); sns.lineplot(data=d, x="end_year", y="estimate", hue="sample_type", style="direction", estimator="median", errorbar=None, ax=ax); ax.set_title("Dynamic versus fixed-country sample"); ax.set_ylabel("Median α across crops"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS6_dynamic_fixed_samples")
    # S7 estimators.
    d = alpha[(alpha["window"] == "1986-2024") & (alpha["sample_type"] == "A_dynamic") & (alpha["trend"] == "linear") & (alpha["transform"] == "tonnes")]
    fig, ax = plt.subplots(figsize=(11, 5)); sns.pointplot(data=d, x="crop", y="estimate", hue="method", dodge=0.4, ax=ax); ax.axhline(2, color="#555555", ls="--"); ax.set_title("OLS, Huber, and Theil–Sen estimates"); ax.set_ylabel("α"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS7_estimators")
    # S8 LOCO.
    d = main[(main["window"] == "1986-2024") & (main["sample_type"] == "A_dynamic") & (main["trend"] == "linear") & (main["transform"] == "tonnes")].copy(); d["label"] = d["crop"] + " " + d["direction"]
    fig, ax = plt.subplots(figsize=(9, 5)); y = np.arange(len(d)); ax.errorbar(d["estimate"], y, xerr=[d["estimate"] - d["loco_alpha_min"], d["loco_alpha_max"] - d["estimate"]], fmt="o"); ax.set_yticks(y, d["label"]); ax.axvline(2, color="#555555", ls="--"); ax.set_title("Leave-one-country-out alpha range"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS8_loco_influence")
    # S9 edge scaling.
    sampled = edge_data.sample(min(len(edge_data), 60_000), random_state=stable_seed("edge-plot"))
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=False, sharey=False)
    for r, direction in enumerate(DIRECTION_ORDER):
        for c, crop in enumerate(CROP_ORDER):
            q = sampled[(sampled["crop"] == crop) & (sampled["direction"] == direction)]
            axes[r, c].scatter(np.log10(q["mean"]), np.log10(q["variance"]), s=3, alpha=0.15, color=COLORS[direction]); axes[r, c].set_title(f"{crop} {direction}"); axes[r, c].set_xlabel("log10 edge mean"); axes[r, c].set_ylabel("log10 edge variance")
    fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS9_edge_mean_variance_scaling")
    # S10 thresholds and S11 matrix method.
    sens = div_sens.dropna(subset=["N_corr"])
    fig, ax = plt.subplots(figsize=(10, 5)); sns.boxplot(data=sens[sens["matrix_method"] == "ledoit_wolf"], x="share_threshold", y="Gap", hue="coverage_threshold", ax=ax); ax.axhline(0, color="#555555"); ax.set_title("N_corr sensitivity to share and coverage thresholds"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS10_diversification_thresholds")
    fig, ax = plt.subplots(figsize=(10, 5)); sns.boxplot(data=sens, x="crop", y="Gap", hue="matrix_method", order=CROP_ORDER, ax=ax); ax.axhline(0, color="#555555"); ax.set_title("Sample versus Ledoit–Wolf correlation matrices"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS11_correlation_matrix_methods")
    # S12 full specification curve.
    spec = main[(main["sample_type"] == "A_dynamic")].copy().sort_values("estimate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 5)); ax.scatter(np.arange(len(spec)), spec["estimate"], c=spec["direction"].map(COLORS), s=9, alpha=0.65); ax.axhline(1, color="#999999", ls=":"); ax.axhline(2, color="#555555", ls="--"); ax.set_xlabel("All reported OLS specifications, sorted"); ax.set_ylabel("α"); ax.set_title("Complete specification curve"); fig.tight_layout(); save_figure(fig, FIG_SUPP_DIR, "FigS12_specification_curve")


def create_all_figures(core: pd.DataFrame, results: dict, diversification: dict, null: dict, phi_boot: pd.DataFrame, edge_data: pd.DataFrame) -> None:
    set_style()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure1(results["alpha"], results["full_fit"], results["model_comparison"])
        figure2(results["alpha"])
        figure3(results["alpha"], results["full_fit"], results["model_comparison"])
        figure4(results["full_metrics"], phi_boot)
        figure5(diversification["full"], diversification["main"])
        figure6(null["summary"], null["draws"])
        supplemental_figures(core, results["alpha"], results["full_metrics"], edge_data, diversification["sensitivity"])
    LOGGER.info("Created main and supplemental figures")
