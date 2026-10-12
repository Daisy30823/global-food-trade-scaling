from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ROOT, WORK_DIR


CROPS = ["Wheat", "Rice", "Maize", "Soybeans"]
DIRECTIONS = ["Import", "Export"]
COLORS = {"Import": "#2F6B9A", "Export": "#C26732"}
MARKERS = {"Import": "o", "Export": "s"}
REPO_ROOT = ROOT.parent
# Released compact inputs and archived figures are never overwritten by a run.
# Freshly generated figure data and panels go to the ignored ``work/`` tree.
BUNDLED_DATA = REPO_ROOT / "data" / "figure_data"
BUNDLED_TABLES = REPO_ROOT / "results" / "tables"
DATA = BUNDLED_DATA
TABLES = BUNDLED_TABLES
WORK_DATA = WORK_DIR / "figure_data"
WORK_TABLES = WORK_DIR / "figure_tables"
MAIN = WORK_DIR / "paper_figures" / "main_panels"
SUPP = WORK_DIR / "paper_figures" / "supplement_panels"
FIGURE_INDEX = WORK_DIR / "paper_figures" / "figure_index.csv"


SOURCE_FILES = {
    "country_metrics.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "Fig1_country_data_v2.csv",
    "rolling_alpha.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "rolling20_residual_block_bootstrap.csv",
    "paired_phi_country.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "paired_phi_country_pairs.csv",
    "global_mirror.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "FigS1_global_mirror.csv",
    "rice_scope.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "FigS2_rice_code30_vs31.csv",
    "reporting_coverage.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "FigS3_reporting_coverage.csv",
    "diversification.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "FigS10_S11_diversification.csv",
    "alpha_specifications.csv": WORK_DIR / "outputs_v2_1" / "figure_data" / "FigS12_specification_curve.csv",
    "edge_scaling.csv": WORK_DIR / "outputs_v2" / "figure_data" / "FigS9_edge_mean_variance.csv",
    "decomposition.csv": WORK_DIR / "outputs_v2_1" / "tables" / "exact_decomposition_residual_block.csv",
    "paired_alpha.csv": WORK_DIR / "outputs_v2_1" / "tables" / "paired_alpha_residual_block.csv",
    "paired_phi_summary.csv": WORK_DIR / "outputs_v2_1" / "tables" / "paired_phi_valid_inference.csv",
    "paired_phi_reporting_aware.csv": WORK_DIR / "outputs_v2_1" / "tables" / "paired_phi_reporting_aware.csv",
    "null_adjusted_country.csv": WORK_DIR / "outputs_v2_1" / "tables" / "country_level_null_adjusted_phi.csv",
    "null_adjusted_paired.csv": WORK_DIR / "outputs_v2_1" / "tables" / "paired_null_adjusted_direction.csv",
    "global_circular_null.csv": WORK_DIR / "outputs_v2" / "tables" / "table6_null_model.csv",
    "coverage_sensitivity.csv": WORK_DIR / "outputs_v2_1" / "tables" / "reporting_entity_coverage_sensitivity.csv",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh_data_bundle() -> None:
    global DATA, TABLES
    DATA = WORK_DATA
    TABLES = WORK_TABLES
    DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for target_name, source in SOURCE_FILES.items():
        if not source.exists():
            raise FileNotFoundError(f"Required analysis output is missing: {source}")
        target = DATA / target_name
        shutil.copy2(source, target)
        rows.append(
            {
                "bundled_file": target_name,
                "analysis_source": source.relative_to(ROOT).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "source_manifest.csv", index=False)
    for source in sorted((WORK_DIR / "outputs_v2_1" / "tables").glob("*.csv")):
        shutil.copy2(source, TABLES / source.name)
    shutil.copy2(WORK_DIR / "outputs_v2" / "tables" / "table6_null_model.csv", TABLES / "global_circular_null.csv")


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.4,
            "axes.edgecolor": "#454545",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.fontsize": 7.2,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _interval(ax, y, point, low, high, color="#2F6B9A", marker="o") -> None:
    ax.hlines(y, low, high, color=color, lw=1.45)
    ax.vlines([low, high], y - 0.08, y + 0.08, color=color, lw=0.8)
    ax.scatter(point, y, color=color, marker=marker, s=25, zorder=3)


def _pad(lo: float, hi: float, frac: float = 0.04) -> tuple[float, float]:
    width = hi - lo
    return lo - width * frac, hi + width * frac


def _clean_output() -> None:
    for directory in (MAIN, SUPP):
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.glob("*.pdf"):
            path.unlink()
        for path in directory.glob("*.png"):
            path.unlink()


def _main_figures(index: list[dict]) -> None:
    metrics = pd.read_csv(DATA / "country_metrics.csv")
    rolling = pd.read_csv(DATA / "rolling_alpha.csv")
    decomposition = pd.read_csv(DATA / "decomposition.csv")
    paired_alpha = pd.read_csv(DATA / "paired_alpha.csv")
    phi_pairs = pd.read_csv(DATA / "paired_phi_country.csv")
    phi_summary = pd.read_csv(DATA / "paired_phi_summary.csv")
    null_paired = pd.read_csv(DATA / "null_adjusted_paired.csv")

    eligible = metrics[metrics["eligible_covariance"].astype(bool) & metrics["phi"].gt(0)].copy()
    main_decomp = decomposition[(decomposition["scheme"] == "residual_two_stage") & (decomposition["block_length"] == 4)].copy()

    log_mu = np.log10(eligible["mu"])
    log_var = np.log10(eligible["variance"])
    xlim = _pad(log_mu.min(), log_mu.max())
    ylim = _pad(log_var.min(), log_var.max())
    sequence = 0
    for direction in DIRECTIONS:
        for crop in CROPS:
            sequence += 1
            g = eligible[(eligible["crop"] == crop) & (eligible["direction"] == direction)]
            fit = main_decomp[(main_decomp["crop"] == crop) & (main_decomp["direction"] == direction) & (main_decomp["component"] == "alpha_V")].iloc[0]
            x = np.log10(g["mu"].to_numpy(float))
            y = np.log10(g["variance"].to_numpy(float))
            slope = float(fit["estimate"])
            intercept = float(np.mean(y) - slope * np.mean(x))
            grid = np.linspace(x.min(), x.max(), 100)
            fig, ax = plt.subplots(figsize=(3.45, 2.75))
            ax.scatter(x, y, s=12, alpha=0.42, color=COLORS[direction], edgecolors="none")
            ax.plot(grid, intercept + slope * grid, color="#202020", lw=1.5)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_xlabel(r"$\log_{10}$ mean trade (tonnes)")
            ax.set_ylabel(r"$\log_{10}$ residual variance")
            ax.set_title(f"{crop} · {direction}", loc="left")
            ax.text(0.97, 0.05, f"α={slope:.3f}\n95% CI [{fit.percentile_ci_low:.3f}, {fit.percentile_ci_high:.3f}]\nn={int(fit.n_reporting_economies)}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
            stem = f"M1_{sequence:02d}_{crop.lower()}_{direction.lower()}_scaling"
            _save(fig, MAIN, stem)
            index.append(_row("Main 1", stem, "Full-period Taylor scaling", "country_metrics.csv; decomposition.csv"))

    rx = _pad(rolling["end_year"].min(), rolling["end_year"].max(), 0.02)
    ry = _pad(min(rolling["percentile_ci_low"].min(), 1.8), max(rolling["percentile_ci_high"].max(), 2.0), 0.04)
    for i, crop in enumerate(CROPS, 1):
        fig, ax = plt.subplots(figsize=(3.55, 2.75))
        for direction in DIRECTIONS:
            g = rolling[(rolling["crop"] == crop) & (rolling["direction"] == direction)].sort_values("end_year")
            ax.plot(g["end_year"], g["estimate"], color=COLORS[direction], marker=MARKERS[direction], ms=2.3, lw=1.05, label=direction)
            ax.fill_between(g["end_year"].to_numpy(float), g["percentile_ci_low"].to_numpy(float), g["percentile_ci_high"].to_numpy(float), color=COLORS[direction], alpha=0.13, linewidth=0)
        pair = paired_alpha[paired_alpha["crop"] == crop].iloc[0]
        ax.axhline(2, color="#777777", ls="--", lw=0.8)
        ax.set_xlim(rx); ax.set_ylim(ry)
        ax.set_xticks([2005, 2010, 2015, 2020, 2024])
        ax.set_xlabel("End year of 20-year window")
        ax.set_ylabel("Taylor exponent α")
        ax.set_title(crop, loc="left")
        ax.legend(ncol=2, loc="upper left")
        ax.text(0.98, 0.04, f"full-period Δα(I−E)={pair.estimate:.3f}\n95% CI [{pair.percentile_ci_low:.3f}, {pair.percentile_ci_high:.3f}]", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.6, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
        stem = f"M2_{i:02d}_{crop.lower()}_rolling_alpha"
        _save(fig, MAIN, stem)
        index.append(_row("Main 2", stem, "Temporal robustness and paired directional α", "rolling_alpha.csv; paired_alpha.csv"))

    decomp_xlim = _pad(np.log10(eligible["D"]).min(), np.log10(eligible["D"]).max())
    decomp_ylim = _pad(np.log10(eligible["variance"]).min(), np.log10(eligible["variance"]).max())
    for i, crop in enumerate(CROPS, 1):
        fig, ax = plt.subplots(figsize=(3.45, 2.75))
        for direction in DIRECTIONS:
            g = eligible[(eligible["crop"] == crop) & (eligible["direction"] == direction)]
            ax.scatter(np.log10(g["D"]), np.log10(g["variance"]), s=11, alpha=0.42, color=COLORS[direction], marker=MARKERS[direction], edgecolors="none", label=direction)
        ax.set_xlim(decomp_xlim); ax.set_ylim(decomp_ylim)
        ax.set_xlabel(r"$\log_{10}$ diagonal edge variance, $D$")
        ax.set_ylabel(r"$\log_{10}$ aggregate variance, $V$")
        ax.set_title(crop, loc="left"); ax.legend(ncol=2)
        stem = f"M3_{i:02d}_{crop.lower()}_variance_decomposition"
        _save(fig, MAIN, stem)
        index.append(_row("Main 3", stem, "Aggregate versus diagonal edge variance", "country_metrics.csv"))

    wide = main_decomp.pivot(index=["crop", "direction"], columns="component", values="estimate").reset_index()
    lo = min(wide["alpha_D"].min(), wide["alpha_V"].min()) - 0.05
    hi = max(wide["alpha_D"].max(), wide["alpha_V"].max()) + 0.05
    fig, ax = plt.subplots(figsize=(3.45, 3.05))
    for direction in DIRECTIONS:
        g = wide[wide["direction"] == direction]
        ax.scatter(g["alpha_D"], g["alpha_V"], color=COLORS[direction], marker=MARKERS[direction], s=34, label=direction)
        for r in g.itertuples():
            ax.annotate(r.crop[0], (r.alpha_D, r.alpha_V), xytext=(3, 3), textcoords="offset points", fontsize=6.8)
    ax.plot([lo, hi], [lo, hi], color="#777777", ls="--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Diagonal-variance exponent, $\alpha_D$")
    ax.set_ylabel(r"Aggregate-variance exponent, $\alpha_V$")
    ax.legend(ncol=2)
    stem = "M3_05_alphaV_vs_alphaD"
    _save(fig, MAIN, stem)
    index.append(_row("Main 3", stem, "Comparison of aggregate and diagonal exponents", "decomposition.csv"))

    beta = main_decomp[main_decomp["component"] == "beta_phi"].copy()
    beta["order"] = beta["crop"].map({c: i for i, c in enumerate(CROPS)}) * 2 + beta["direction"].map({"Import": 0, "Export": 1})
    beta = beta.sort_values("order", ascending=False)
    fig, ax = plt.subplots(figsize=(3.8, 3.45))
    for y, r in enumerate(beta.itertuples()):
        _interval(ax, y, r.estimate, r.percentile_ci_low, r.percentile_ci_high, COLORS[r.direction], MARKERS[r.direction])
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(beta)), beta["crop"] + " · " + beta["direction"])
    ax.set_xlabel(r"Synchronization elasticity, $\beta_{\phi}$")
    stem = "M3_06_beta_phi_intervals"
    _save(fig, MAIN, stem)
    index.append(_row("Main 3", stem, "Synchronization elasticity", "decomposition.csv"))

    all_x = np.log(phi_pairs["phi_import"]); all_y = np.log(phi_pairs["phi_export"])
    pair_lim = _pad(min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max()))
    for i, crop in enumerate(CROPS, 1):
        g = phi_pairs[phi_pairs["crop"] == crop]
        summary = phi_summary[phi_summary["crop"] == crop].iloc[0]
        x, y = np.log(g["phi_import"]), np.log(g["phi_export"])
        fig, ax = plt.subplots(figsize=(3.35, 3.05))
        ax.scatter(x, y, s=13, alpha=0.48, color="#5B7FA3", edgecolors="none")
        ax.plot(pair_lim, pair_lim, color="#444444", ls="--", lw=0.8)
        ax.set_xlim(pair_lim); ax.set_ylim(pair_lim)
        ax.set_xlabel(r"$\log \phi$, import")
        ax.set_ylabel(r"$\log \phi$, export")
        ax.set_title(crop, loc="left")
        ax.text(0.04, 0.96, f"typical E/I ratio={summary.typical_phi_ratio:.2f}\n95% CI [{summary.typical_phi_ratio_ci_low:.2f}, {summary.typical_phi_ratio_ci_high:.2f}]\nn={int(summary.n_pairs)}", transform=ax.transAxes, va="top", fontsize=6.7, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
        stem = f"M4_{i:02d}_{crop.lower()}_paired_phi"
        _save(fig, MAIN, stem)
        index.append(_row("Main 4", stem, "Paired import-export synchronization", "paired_phi_country.csv; paired_phi_summary.csv"))

    g = null_paired.set_index("crop").loc[CROPS].reset_index().iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    for y, r in enumerate(g.itertuples()):
        _interval(ax, y, r.paired_median_export_minus_import, r.bootstrap_ci_low, r.bootstrap_ci_high, "#2F6B9A")
    ax.axvline(0, color="#555555", ls="--", lw=0.9)
    ax.set_yticks(np.arange(len(g)), g["crop"])
    ax.set_xlabel(r"Median $\Delta_{sync}$, export − import")
    stem = "M5_01_null_adjusted_direction"
    _save(fig, MAIN, stem)
    index.append(_row("Main 5", stem, "Country-level null-adjusted directional synchronization", "null_adjusted_paired.csv"))


def _supplement_figures(index: list[dict]) -> None:
    mirror = pd.read_csv(DATA / "global_mirror.csv")
    rice = pd.read_csv(DATA / "rice_scope.csv")
    coverage = pd.read_csv(DATA / "reporting_coverage.csv")
    alpha = pd.read_csv(DATA / "alpha_specifications.csv")
    edge = pd.read_csv(DATA / "edge_scaling.csv")
    div = pd.read_csv(DATA / "diversification.csv")
    aware = pd.read_csv(DATA / "paired_phi_reporting_aware.csv")
    decomp = pd.read_csv(DATA / "decomposition.csv")
    circular = pd.read_csv(DATA / "global_circular_null.csv")

    ratio_ylim = _pad(mirror["export_import_ratio"].min(), mirror["export_import_ratio"].max())
    for i, crop in enumerate(CROPS, 1):
        g = mirror[mirror["crop"] == crop]
        fig, ax = plt.subplots(figsize=(3.45, 2.55)); ax.plot(g["year"], g["export_import_ratio"], color="#2F6B9A", lw=1.2); ax.axhline(1, color="#777777", ls="--", lw=0.8)
        ax.set_ylim(ratio_ylim); ax.set_xlabel("Year"); ax.set_ylabel("Reported export/import ratio"); ax.set_title(crop, loc="left")
        stem = f"S01_{i:02d}_{crop.lower()}_global_mirror"; _save(fig, SUPP, stem); index.append(_row("Supplement 1", stem, "Global mirror diagnostic", "global_mirror.csv"))

    rice = rice.sort_values(["direction", "item_code"], ascending=[False, True]); fig, ax = plt.subplots(figsize=(3.9, 2.6))
    for y, r in enumerate(rice.itertuples()): _interval(ax, y, r.alpha, r.ci_low, r.ci_high, COLORS[r.direction], "o" if r.item_code == 30 else "s")
    ax.axvline(2, color="#777777", ls="--", lw=0.8); ax.set_yticks(np.arange(len(rice)), [f"Code {r.item_code} · {r.direction}" for r in rice.itertuples()]); ax.set_xlabel("Taylor exponent α")
    stem = "S02_01_rice_code_scope"; _save(fig, SUPP, stem); index.append(_row("Supplement 2", stem, "Rice code scope sensitivity", "rice_scope.csv"))

    cov_ylim = _pad(coverage["reporter_active"].min(), coverage["reporter_active"].max())
    for i, crop in enumerate(CROPS, 1):
        fig, ax = plt.subplots(figsize=(3.45, 2.55))
        for direction in DIRECTIONS:
            g = coverage[(coverage["crop"] == crop) & (coverage["direction"] == direction)]
            ax.plot(g["year"], g["reporter_active"], color=COLORS[direction], label=direction, lw=1.2)
        ax.set_ylim(cov_ylim); ax.set_xlabel("Year"); ax.set_ylabel("Active reporting economies/areas"); ax.set_title(crop, loc="left"); ax.legend(ncol=2)
        stem = f"S03_{i:02d}_{crop.lower()}_coverage"; _save(fig, SUPP, stem); index.append(_row("Supplement 3", stem, "Reporting coverage", "reporting_coverage.csv"))

    base = alpha[(alpha["window_id"] == "full_1986_2024") & (alpha["sample_type"] == "A_dynamic") & (alpha["method"] == "OLS_HC3")]
    _categorical_points(base, ["raw", "linear", "quadratic"], "trend", "S04_01_detrending_methods", "Detrending-method sensitivity", index, "Supplement 4")

    win = alpha[(alpha["sample_type"] == "A_dynamic") & (alpha["trend"] == "linear") & (alpha["method"] == "OLS_HC3") & alpha["window_type"].isin(["rolling15", "rolling20", "rolling25"])]
    wy = _pad(win["estimate"].min(), max(win["estimate"].max(), 2.0))
    for i, crop in enumerate(CROPS, 1):
        fig, ax = plt.subplots(figsize=(3.55, 2.65))
        for direction in DIRECTIONS:
            for wt, ls in (("rolling15", ":"), ("rolling20", "-"), ("rolling25", "--")):
                g = win[(win["crop"] == crop) & (win["direction"] == direction) & (win["window_type"] == wt)].sort_values("end_year")
                ax.plot(g["end_year"], g["estimate"], color=COLORS[direction], ls=ls, lw=1, label=f"{direction} {wt[-2:]} y")
        ax.axhline(2, color="#777777", ls="--", lw=0.8); ax.set_ylim(wy); ax.set_xlabel("Window end year"); ax.set_ylabel("Taylor exponent α"); ax.set_title(crop, loc="left"); ax.legend(ncol=2, fontsize=5.9)
        stem = f"S05_{i:02d}_{crop.lower()}_window_length"; _save(fig, SUPP, stem); index.append(_row("Supplement 5", stem, "Window-length sensitivity", "alpha_specifications.csv"))

    samp = alpha[(alpha["window_id"] == "full_1986_2024") & (alpha["trend"] == "linear") & (alpha["method"] == "OLS_HC3")]
    _categorical_points(samp, ["A_dynamic", "B_fixed"], "sample_type", "S06_01_sample_definitions", "Dynamic versus fixed samples", index, "Supplement 6")

    method = base[base["trend"] == "linear"].sort_values(["crop", "direction"])
    fig, ax = plt.subplots(figsize=(6.8, 2.75)); x = np.arange(len(method))
    for j, (col, label) in enumerate((("alpha", "OLS"), ("huber_alpha", "Huber"), ("theil_sen_alpha", "Theil–Sen"))): ax.scatter(x + (j - 1) * 0.2, method[col], label=label, s=25)
    labels = (method["crop"] + "\n" + method["direction"]).tolist(); ax.set_xticks(x, labels); ax.axhline(2, color="#777777", ls="--", lw=0.8); ax.set_ylabel("Taylor exponent α"); ax.legend(ncol=3)
    stem = "S07_01_robust_estimators"; _save(fig, SUPP, stem); index.append(_row("Supplement 7", stem, "Robust-estimator sensitivity", "alpha_specifications.csv"))

    fig, ax = plt.subplots(figsize=(4.2, 3.4)); loco = method.iloc[::-1]
    for y, r in enumerate(loco.itertuples()): _interval(ax, y, r.estimate, r.loco_alpha_min, r.loco_alpha_max, COLORS[r.direction], MARKERS[r.direction])
    ax.axvline(2, color="#777777", ls="--", lw=0.8); ax.set_yticks(np.arange(len(loco)), loco["crop"] + " · " + loco["direction"]); ax.set_xlabel("Leave-one-economy-out α range")
    stem = "S08_01_leave_one_out"; _save(fig, SUPP, stem); index.append(_row("Supplement 8", stem, "Leave-one-reporting-economy-out influence", "alpha_specifications.csv"))

    edge = edge[(edge["mean"] > 0) & (edge["variance"] > 0)]
    ex = _pad(np.log10(edge["mean"]).min(), np.log10(edge["mean"]).max()); ey = _pad(np.log10(edge["variance"]).min(), np.log10(edge["variance"]).max())
    for i, crop in enumerate(CROPS, 1):
        g = edge[edge["crop"] == crop]; fig, ax = plt.subplots(figsize=(3.45, 2.7)); ax.hexbin(np.log10(g["mean"]), np.log10(g["variance"]), gridsize=35, mincnt=1, cmap="Blues"); ax.set_xlim(ex); ax.set_ylim(ey); ax.set_xlabel(r"$\log_{10}$ edge mean"); ax.set_ylabel(r"$\log_{10}$ edge variance"); ax.set_title(crop, loc="left")
        stem = f"S09_{i:02d}_{crop.lower()}_edge_scaling"; _save(fig, SUPP, stem); index.append(_row("Supplement 9", stem, "Bilateral-edge scaling", "edge_scaling.csv"))

    lw = div[div["matrix_method"] == "ledoit_wolf"].copy().iloc[::-1]; fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for y, r in enumerate(lw.itertuples()): _interval(ax, y, r.median_gap_main_eligible, r.median_gap_ci_low, r.median_gap_ci_high)
    ax.axvline(0, color="#777777", ls="--", lw=0.8); ax.set_yticks(np.arange(len(lw)), [f"share {r.share_threshold:.1%}; coverage {r.coverage_threshold:.0%}" for r in lw.itertuples()]); ax.set_xlabel("Median diversification gap")
    stem = "S10_01_diversification_thresholds"; _save(fig, SUPP, stem); index.append(_row("Supplement 10", stem, "Diversification-threshold sensitivity", "diversification.csv"))

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for j, matrix_method in enumerate(("ledoit_wolf", "sample")):
        g = div[div["matrix_method"] == matrix_method].sort_values(["share_threshold", "coverage_threshold"]); ax.scatter(np.arange(len(g)) + (j - 0.5) * 0.18, g["median_gap_main_eligible"], label=matrix_method, s=28)
    ax.axhline(0, color="#777777", ls="--", lw=0.8); ax.set_xticks(np.arange(len(g)), [f"{r.share_threshold:.1%}\n{r.coverage_threshold:.0%}" for r in g.itertuples()]); ax.set_ylabel("Median diversification gap"); ax.legend(ncol=2)
    stem = "S11_01_correlation_method"; _save(fig, SUPP, stem); index.append(_row("Supplement 11", stem, "Correlation-matrix method sensitivity", "diversification.csv"))

    spec = alpha[(alpha["transform"] == "tonnes") & alpha["estimate"].notna()]
    sy = _pad(spec["estimate"].min(), max(spec["estimate"].max(), 2.0))
    for i, crop in enumerate(CROPS, 1):
        fig, ax = plt.subplots(figsize=(3.45, 2.6))
        for direction in DIRECTIONS:
            vals = np.sort(spec[(spec["crop"] == crop) & (spec["direction"] == direction)]["estimate"].to_numpy()); ax.plot(np.linspace(0, 1, len(vals)), vals, color=COLORS[direction], label=direction, lw=1.15)
        ax.axhline(2, color="#777777", ls="--", lw=0.8); ax.set_ylim(sy); ax.set_xlabel("Prespecified specification quantile"); ax.set_ylabel("Taylor exponent α"); ax.set_title(crop, loc="left"); ax.legend(ncol=2)
        stem = f"S12_{i:02d}_{crop.lower()}_specification_curve"; _save(fig, SUPP, stem); index.append(_row("Supplement 12", stem, "Prespecified specification curve", "alpha_specifications.csv"))

    g = aware[aware["specification"] == "reporting_aware"].set_index("crop").loc[CROPS].reset_index().iloc[::-1]; fig, ax = plt.subplots(figsize=(3.9, 2.7))
    for y, r in enumerate(g.itertuples()): _interval(ax, y, r.typical_phi_ratio, r.typical_phi_ratio_ci_low, r.typical_phi_ratio_ci_high)
    ax.axvline(1, color="#777777", ls="--", lw=0.8); ax.set_yticks(np.arange(len(g)), g["crop"]); ax.set_xlabel("Typical export/import φ ratio")
    stem = "S13_01_reporting_aware_phi"; _save(fig, SUPP, stem); index.append(_row("Supplement 13", stem, "Reporting-aware paired synchronization", "paired_phi_reporting_aware.csv"))

    main = decomp[(decomp["scheme"] == "residual_two_stage") & (decomp["block_length"] == 4)].pivot(index=["crop", "direction"], columns="component", values="estimate").reset_index(); x = main["alpha_D"] + main["beta_phi"]; y = main["alpha_V"]; lo = min(x.min(), y.min()) - 0.02; hi = max(x.max(), y.max()) + 0.02
    fig, ax = plt.subplots(figsize=(3.25, 3.0)); ax.scatter(x, y, c=main["direction"].map(COLORS), s=34); ax.plot([lo, hi], [lo, hi], color="#555555", ls="--", lw=0.8)
    for r, xx, yy in zip(main.itertuples(), x, y): ax.annotate(f"{r.crop[0]}{r.direction[0]}", (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel(r"$\alpha_D+\beta_{\phi}$"); ax.set_ylabel(r"$\alpha_V$")
    stem = "S14_01_algebraic_identity"; _save(fig, SUPP, stem); index.append(_row("Supplement 14", stem, "Algebraic identity validation", "decomposition.csv"))

    for i, (metric, label) in enumerate((("median_phi", "Median synchronization factor, φ"), ("alpha", "Taylor exponent, α")), 1):
        g = circular[circular["metric"] == metric].copy(); g["order"] = g["crop"].map({c: j for j, c in enumerate(CROPS)}) * 2 + g["direction"].map({"Import": 0, "Export": 1}); g = g.sort_values("order", ascending=False)
        fig, ax = plt.subplots(figsize=(4.25, 3.35)); yy = np.arange(len(g)); ax.hlines(yy, g["null_q025"], g["null_q975"], color="#8A8A8A", lw=1.2); ax.scatter(g["null_median"], yy, facecolors="none", edgecolors="#666666", label="Circular-shift null"); ax.scatter(g["observed"], yy, c=g["direction"].map(COLORS), marker="s", label="Observed")
        ax.set_yticks(yy, g["crop"] + " · " + g["direction"]); ax.set_xlabel(label); ax.legend(ncol=2, fontsize=6.5)
        stem = f"S15_{i:02d}_global_null_{metric}"; _save(fig, SUPP, stem); index.append(_row("Supplement 15", stem, "Global circular-shift null diagnostic", "global_circular_null.csv"))


def _categorical_points(data: pd.DataFrame, levels: list[str], column: str, stem: str, claim: str, index: list[dict], logical: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 2.75)); ref = data[data[column] == levels[0]].sort_values(["crop", "direction"]); x = np.arange(len(ref))
    display = {"A_dynamic": "dynamic", "B_fixed": "fixed"}
    for j, level in enumerate(levels):
        g = data[data[column] == level].sort_values(["crop", "direction"]); ax.scatter(x + (j - (len(levels) - 1) / 2) * 0.2, g["estimate"], label=display.get(level, level), s=25)
    ax.set_xticks(x, (ref["crop"] + "\n" + ref["direction"]).tolist()); ax.axhline(2, color="#777777", ls="--", lw=0.8); ax.set_ylabel("Taylor exponent α"); ax.legend(ncol=len(levels))
    _save(fig, SUPP, stem); index.append(_row(logical, stem, claim, "alpha_specifications.csv"))


def _row(logical: str, stem: str, claim: str, data_source: str) -> dict:
    return {"logical_figure": logical, "panel_stem": stem, "claim_or_diagnostic": claim, "data_source": data_source, "png_dpi": 600, "vector_format": "PDF"}


def generate_paper_figures(refresh_data: bool = False) -> pd.DataFrame:
    if refresh_data:
        refresh_data_bundle()
    missing = [name for name in SOURCE_FILES if not (DATA / name).exists()]
    if missing:
        raise FileNotFoundError("Packaged figure data missing: " + ", ".join(missing))
    set_style(); _clean_output(); index: list[dict] = []
    _main_figures(index); _supplement_figures(index)
    frame = pd.DataFrame(index)
    FIGURE_INDEX.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIGURE_INDEX, index=False)
    return frame
