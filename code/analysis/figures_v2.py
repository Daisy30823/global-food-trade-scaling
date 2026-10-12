from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DATA_DIR, FIG_MAIN_DIR, FIG_SUPP_DIR
from .utils import get_logger, save_csv

LOGGER = get_logger()
CROPS = ["Wheat", "Rice", "Maize", "Soybeans"]
DIRECTIONS = ["Import", "Export"]
COLORS = {"Import": "#2F6B9A", "Export": "#C26732"}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, directory, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure1(full_fit: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.2))
    source_rows = []
    for row, direction in enumerate(DIRECTIONS):
        for col, crop in enumerate(CROPS):
            ax = axes[row, col]
            data = full_fit[
                (full_fit["crop"] == crop)
                & (full_fit["direction"] == direction)
                & (full_fit["sample_type"] == "A_dynamic")
                & full_fit["eligible_covariance"]
                & full_fit["phi"].gt(0)
            ].copy()
            x = np.log10(data["mu"])
            y = np.log10(data["variance"])
            ax.scatter(x, y, s=12, alpha=0.38, color=COLORS[direction], linewidths=0)
            fit = decomposition[
                (decomposition["crop"] == crop)
                & (decomposition["direction"] == direction)
                & (decomposition["component"] == "alpha_V")
            ].iloc[0]
            intercept = float(np.mean(np.log(data["variance"])) - fit.estimate * np.mean(np.log(data["mu"])))
            grid = np.linspace(x.min(), x.max(), 100)
            ax.plot(grid, (intercept + fit.estimate * grid * np.log(10)) / np.log(10), color="#202020", lw=1.6)
            influence = data.nlargest(1, "cooks_distance").iloc[0]
            ax.annotate(str(influence["country"]), (np.log10(influence.mu), np.log10(influence.variance)), xytext=(3, 3), textcoords="offset points", fontsize=6)
            ax.set_title(f"{crop} · {direction}")
            ax.text(
                0.03,
                0.97,
                f"α={fit.estimate:.3f}\nHC3 [{fit.hc3_ci_low:.3f}, {fit.hc3_ci_high:.3f}]\n2-stage [{fit.two_stage_ci_low:.3f}, {fit.two_stage_ci_high:.3f}]\nn={int(fit.n_reporting_economies)}",
                transform=ax.transAxes,
                va="top",
                fontsize=6.5,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            if row == 1:
                ax.set_xlabel(r"$log_{10}$ mean tonnes")
            if col == 0:
                ax.set_ylabel(r"$log_{10}$ residual variance")
            source_rows.append(data.assign(panel_crop=crop, panel_direction=direction))
    save_csv(pd.concat(source_rows, ignore_index=True), FIG_DATA_DIR / "Fig1_country_data_v2.csv")
    fig.suptitle("Figure 1. Full-period subquadratic Taylor scaling, 1986–2024", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig1_full_period_subquadratic_taylor_scaling")


def _forest(ax, data: pd.DataFrame, estimate: str, low: str, high: str, labels: list[str], title: str, reference=None) -> None:
    y = np.arange(len(data))
    ax.hlines(y, data[low], data[high], color="#7C8796", lw=1.1)
    ax.vlines(data[low], y - 0.08, y + 0.08, color="#7C8796", lw=0.8)
    ax.vlines(data[high], y - 0.08, y + 0.08, color="#7C8796", lw=0.8)
    ax.scatter(data[estimate], y, color="#374151", s=17, zorder=3)
    ax.set_yticks(y, labels)
    if reference is not None:
        for value, style in reference:
            ax.axvline(value, color="#777777", ls=style, lw=0.8)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.18)


def figure2(decomposition: pd.DataFrame, rolling: pd.DataFrame, paired_alpha: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    full = decomposition[decomposition["component"] == "alpha_V"].copy()
    full["order"] = full["crop"].map({c: i for i, c in enumerate(CROPS)}) * 2 + full["direction"].map({"Import": 0, "Export": 1})
    full = full.sort_values("order")
    _forest(
        axes[0, 0], full, "estimate", "two_stage_ci_low", "two_stage_ci_high",
        [f"{r.crop} {r.direction}" for r in full.itertuples()], "A. Full-period α with two-stage 95% CI", [(1, ":"), (2, "--")]
    )
    for ax, direction, title in [(axes[0, 1], "Import", "B. Import rolling α"), (axes[1, 0], "Export", "C. Export rolling α")]:
        for crop in CROPS:
            data = rolling[(rolling["crop"] == crop) & (rolling["direction"] == direction)].sort_values("end_year")
            ax.plot(data["end_year"], data["estimate"], label=crop, lw=1.4)
            ax.fill_between(data["end_year"], data["block_boot_ci_low"], data["block_boot_ci_high"], alpha=0.08)
        ax.axhline(2, color="#777777", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("20-year window end")
        ax.set_ylabel("α")
    axes[0, 1].legend(frameon=False, ncol=2)
    paired = paired_alpha.set_index("crop").reindex(CROPS).reset_index()
    _forest(
        axes[1, 1], paired, "estimate_import_minus_export", "ci_low", "ci_high", paired["crop"].tolist(),
        "D. Paired α(import − export)", [(0, "--")]
    )
    fig.suptitle("Figure 2. Temporal robustness and paired direction comparison", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig2_temporal_and_paired_direction")


def figure3(full_fit: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 4)
    for col, crop in enumerate(CROPS):
        ax = fig.add_subplot(grid[0, col])
        for direction in DIRECTIONS:
            data = full_fit[
                (full_fit["crop"] == crop)
                & (full_fit["direction"] == direction)
                & (full_fit["sample_type"] == "A_dynamic")
                & full_fit["eligible_covariance"]
            ]
            ax.scatter(np.log10(data["mu"]), np.log10(data["cv2"]), s=9, alpha=0.28, color=COLORS[direction], label=direction)
            x = np.log10(data["mu"].to_numpy(float))
            y = np.log10(data["cv2"].to_numpy(float))
            slope, intercept = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, intercept + slope * xx, color=COLORS[direction], lw=1.3)
        ax.set_title(f"{chr(65 + col)}. {crop}")
        ax.set_xlabel(r"$log_{10}(mu)$")
        if col == 0:
            ax.set_ylabel(r"$log_{10}(CV^2)$")
        if col == 3:
            ax.legend(frameon=False)
    ax = fig.add_subplot(grid[1, :2])
    order = decomposition.copy()
    order["label"] = order["crop"] + " " + order["direction"] + " · " + order["component"]
    order["rank"] = order["crop"].map({c: i for i, c in enumerate(CROPS)}) * 6 + order["direction"].map({"Import": 0, "Export": 3}) + order["component"].map({"alpha_V": 0, "alpha_D": 1, "beta_phi": 2})
    order = order.sort_values("rank")
    _forest(ax, order, "estimate", "two_stage_ci_low", "two_stage_ci_high", order["label"].tolist(), "E. Exact exponent components", [(0, ":"), (2, "--")])
    ax = fig.add_subplot(grid[1, 2:])
    wide = decomposition.pivot(index=["crop", "direction"], columns="component", values="estimate").reset_index()
    ax.scatter(wide["alpha_V"], wide["alpha_D"] + wide["beta_phi"], c=wide["direction"].map(COLORS), s=45)
    lim = [min(wide["alpha_V"].min(), (wide["alpha_D"] + wide["beta_phi"]).min()) - 0.03, max(wide["alpha_V"].max(), (wide["alpha_D"] + wide["beta_phi"]).max()) + 0.03]
    ax.plot(lim, lim, color="#333333", ls="--")
    for row in wide.itertuples():
        ax.annotate(f"{row.crop} {row.direction[0]}", (row.alpha_V, row.alpha_D + row.beta_phi), xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"Direct $alpha_V$")
    ax.set_ylabel(r"$alpha_D + \beta_\phi$")
    ax.set_title("F. Identity check (45° line)")
    fig.suptitle("Figure 3. Exact origin of the Taylor exponent", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig3_exact_origin_taylor_exponent")


def figure4(full_metrics: pd.DataFrame, paired_phi: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    paired_country_rows = []
    for crop in CROPS:
        group = full_metrics[(full_metrics["crop"] == crop) & full_metrics["eligible_covariance"] & full_metrics["phi"].gt(0)]
        wide = group.pivot(index="country_code", columns="direction", values="phi").dropna()
        for row in wide.itertuples():
            paired_country_rows.append({"crop": crop, "country_code": row.Index, "phi_import": row.Import, "phi_export": row.Export, "difference": row.Export - row.Import})
    country = pd.DataFrame(paired_country_rows)
    save_csv(country, FIG_DATA_DIR / "Fig4_paired_country_phi.csv")
    sample = country.groupby("crop", group_keys=False).apply(lambda x: x.sample(min(35, len(x)), random_state=17), include_groups=False).reset_index()
    x_positions = {"Import": 0, "Export": 1}
    for _, row in sample.iterrows():
        axes[0, 0].plot([0, 1], [row.phi_import, row.phi_export], color="#8B95A5", alpha=0.22, lw=0.7)
    axes[0, 0].set_xticks([0, 1], ["Import", "Export"])
    axes[0, 0].axhline(1, color="#777777", ls=":")
    axes[0, 0].set_ylabel("φ")
    axes[0, 0].set_title("A. Paired reporting-economy φ")
    paired = paired_phi.set_index("crop").reindex(CROPS).reset_index()
    _forest(axes[0, 1], paired, "median_difference", "median_ci_low", "median_ci_high", paired["crop"].tolist(), "B. Median φ(export − import)", [(0, "--")])
    beta = decomposition[decomposition["component"] == "beta_phi"].copy()
    beta["rank"] = beta["crop"].map({c: i for i, c in enumerate(CROPS)}) * 2 + beta["direction"].map({"Import": 0, "Export": 1})
    beta = beta.sort_values("rank")
    _forest(axes[1, 0], beta, "estimate", "two_stage_ci_low", "two_stage_ci_high", [f"{r.crop} {r.direction}" for r in beta.itertuples()], "C. βφ with two-stage 95% CI", [(0, "--")])
    y = np.arange(len(paired))
    axes[1, 1].errorbar(paired["share_export_gt_import"], y, xerr=[paired["share_export_gt_import"] - paired["share_ci_low"], paired["share_ci_high"] - paired["share_export_gt_import"]], fmt="o", color="#5C3A8E", capsize=2)
    axes[1, 1].axvline(0.5, color="#777777", ls="--")
    axes[1, 1].set_yticks(y, paired["crop"])
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xlabel("Share with φexport > φimport")
    axes[1, 1].set_title("D. Directional share")
    fig.suptitle("Figure 4. Directional covariance asymmetry", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig4_directional_covariance_asymmetry")


def _p_label(value: float) -> str:
    return "p<0.001" if value <= 1 / 1001 else f"p={value:.3f}"


def write_figure5_pooling_audit() -> None:
    from .utils import write_json

    write_json(
        {
            "pooled_statistic_rule": "direct country-level statistic within each crop-direction-window",
            "average_of_crop_medians_used": False,
            "average_of_ci_bounds_used": False,
        },
        FIG_DATA_DIR / "Fig5_pooling_audit.json",
    )


def figure5(circular: pd.DataFrame, all_windows: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric, title in [(axes[0, 0], "median_phi", "A. Observed vs circular-shift null median φ"), (axes[0, 1], "alpha", "B. Observed vs circular-shift null α")]:
        data = circular[circular["metric"] == metric].copy()
        data["rank"] = data["crop"].map({c: i for i, c in enumerate(CROPS)}) * 2 + data["direction"].map({"Import": 0, "Export": 1})
        data = data.sort_values("rank")
        x = np.arange(len(data))
        ax.errorbar(x, data["null_median"], yerr=[data["null_median"] - data["null_q025"], data["null_q975"] - data["null_median"]], fmt="o", color="#5B7FA3", capsize=2, label="Null")
        ax.scatter(x, data["observed"], marker="D", color="#A23E48", s=28, label="Observed", zorder=4)
        for xi, row in zip(x, data.itertuples()):
            ax.annotate(_p_label(row.empirical_p_value), (xi, row.observed), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=5.5, rotation=45)
        ax.set_xticks(x, [f"{r.crop}\n{r.direction}" for r in data.itertuples()], rotation=30, ha="right")
        ax.set_title(title)
        ax.legend(frameon=False)
    rolling = all_windows[all_windows["window_id"].astype(str).str.startswith("roll20_")]
    for ax, metric, title in [(axes[1, 0], "median_phi", "C. Rolling observed − null φ"), (axes[1, 1], "alpha", "D. Rolling observed − null α")]:
        data = rolling[rolling["metric"] == metric].copy()
        data["group"] = data["crop"] + " " + data["direction"]
        matrix = data.pivot(index="group", columns="end_year", values="observed") - data.pivot(index="group", columns="end_year", values="null_mean")
        order = [f"{crop} {direction}" for crop in CROPS for direction in DIRECTIONS]
        matrix = matrix.reindex(order)
        scale = np.nanmax(np.abs(matrix.to_numpy()))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set_yticks(np.arange(len(matrix)), matrix.index)
        ticks = np.arange(0, len(matrix.columns), 3)
        ax.set_xticks(ticks, matrix.columns[ticks], rotation=45)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Figure 5. Temporal synchronization null model", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig5_temporal_synchronization_null")
    write_figure5_pooling_audit()


def figure6_coverage(reporting: pd.DataFrame, sovereign: pd.DataFrame, affected: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    wide = reporting.pivot(index=["crop", "direction"], columns="specification", values=["alpha_V", "beta_phi"]).reset_index()
    for ax, metric, title in [(axes[0, 0], "alpha_V", "A. Reporting-aware α"), (axes[0, 1], "beta_phi", "B. Reporting-aware βφ")]:
        a = wide[(metric, "blank_as_zero")]
        b = wide[(metric, "reporting_aware_complete_observations")]
        ax.scatter(a, b, c=wide[("direction", "")].map(COLORS), s=42)
        lim = [min(a.min(), b.min()) - 0.03, max(a.max(), b.max()) + 0.03]
        ax.plot(lim, lim, color="#777777", ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Blank-as-zero")
        ax.set_ylabel("Reporting-aware")
        ax.set_title(title)
    s = sovereign.pivot(index=["crop", "direction"], columns="sample", values="alpha_V").reset_index()
    axes[1, 0].scatter(s["all_reporting_economies_areas"], s["restricted_sovereign_state_sensitivity"], c=s["direction"].map(COLORS), s=42)
    lim = [min(s.iloc[:, -2:].min()) - 0.03, max(s.iloc[:, -2:].max()) + 0.03]
    axes[1, 0].plot(lim, lim, color="#777777", ls="--")
    axes[1, 0].set_xlim(lim); axes[1, 0].set_ylim(lim)
    axes[1, 0].set_xlabel("All reporting economies/areas")
    axes[1, 0].set_ylabel("Restricted sovereign-state sensitivity")
    axes[1, 0].set_title("C. Entity-coverage sensitivity")
    top = affected.nlargest(15, "noncoverage_years")
    axes[1, 1].barh(np.arange(len(top)), top["noncoverage_years"], color="#7B8794")
    axes[1, 1].set_yticks(np.arange(len(top)), (top["country"] + " · " + top["crop"] + " " + top["direction"]).tolist(), fontsize=6)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Reporter-year noncoverage count")
    axes[1, 1].set_title("D. Most affected reporting economies/areas")
    fig.suptitle("Figure 6. Robustness and entity-coverage sensitivity", fontsize=13)
    fig.tight_layout()
    save_figure(fig, FIG_MAIN_DIR, "Fig6_reporting_and_entity_coverage_sensitivity")


def diversification_figure(diversification: dict, main: bool) -> None:
    data = diversification["main"]
    eligible = data[data["main_eligible"]].dropna(subset=["Gap_source"])
    summary = diversification["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for crop in CROPS:
        group = eligible[eligible["crop"] == crop]
        axes[0].scatter(group["N_nominal_matched"], group["N_corr_source"], s=14, alpha=0.45, label=crop)
    maximum = max(eligible["N_nominal_matched"].max(), eligible["N_corr_source"].max())
    axes[0].plot([1, maximum], [1, maximum], color="#777777", ls="--")
    axes[0].set_xlabel("Matched nominal suppliers"); axes[0].set_ylabel("Source-risk-adjusted suppliers"); axes[0].legend(frameon=False)
    axes[1].hist(eligible["Gap_source"], bins=25, color="#5B7FA3", alpha=0.85)
    axes[1].axvline(0, color="#777777", ls="--"); axes[1].set_xlabel("Gap_source"); axes[1].set_ylabel("Reporting economies")
    lw = summary[summary["matrix_method"] == "ledoit_wolf"].copy()
    x = np.arange(len(lw))
    axes[2].errorbar(x, lw["median_gap_main_eligible"], yerr=[lw["median_gap_main_eligible"] - lw["median_gap_ci_low"], lw["median_gap_ci_high"] - lw["median_gap_main_eligible"]], fmt="o", capsize=2)
    axes[2].axhline(0, color="#777777", ls="--")
    axes[2].set_xticks(x, [f"s={r.share_threshold:.1%}\nc={r.coverage_threshold:.0%}" for r in lw.itertuples()], rotation=45, ha="right")
    axes[2].set_ylabel("Median Gap_source")
    fig.suptitle("Source-risk-adjusted diversification (leave-one-importer-out)", fontsize=12)
    fig.tight_layout()
    directory = FIG_MAIN_DIR if main else FIG_SUPP_DIR
    stem = "Fig6_source_risk_adjusted_diversification" if main else "FigS_source_risk_adjusted_diversification"
    save_figure(fig, directory, stem)


def create_all_figures_v2(
    full_fit: pd.DataFrame,
    full_metrics: pd.DataFrame,
    decomposition: dict,
    rolling: pd.DataFrame,
    paired: dict,
    circular_null: dict,
    all_window_null: pd.DataFrame,
    diversification: dict,
    reporting: dict,
    sovereign: pd.DataFrame,
) -> list[str]:
    set_style()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure1(full_fit, decomposition["summary"])
        figure2(decomposition["summary"], rolling, paired["alpha"])
        figure3(full_fit, decomposition["summary"])
        figure4(full_metrics, paired["phi"], decomposition["summary"])
        figure5(circular_null["summary"], all_window_null)
        if diversification["stable"]:
            diversification_figure(diversification, main=True)
            sixth = "Fig6_source_risk_adjusted_diversification"
        else:
            figure6_coverage(reporting["comparison"], sovereign, reporting["affected"])
            diversification_figure(diversification, main=False)
            sixth = "Fig6_reporting_and_entity_coverage_sensitivity"
    LOGGER.info("Created six v2 main figures; diversification stable=%s", diversification["stable"])
    return [
        "Fig1_full_period_subquadratic_taylor_scaling",
        "Fig2_temporal_and_paired_direction",
        "Fig3_exact_origin_taylor_exponent",
        "Fig4_directional_covariance_asymmetry",
        "Fig5_temporal_synchronization_null",
        sixth,
    ]
