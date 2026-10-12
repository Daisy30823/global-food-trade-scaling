from __future__ import annotations

import warnings
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CROPS, OUT as V2_OUT
from .v2_1 import FIGURE_DATA_DIR, FIGURE_MAIN_DIR, FIGURE_SUPP_DIR, save_csv


CROP_ORDER = list(CROPS.values())
COLORS = {"Import": "#31688E", "Export": "#D17C29"}
MARKERS = {"Import": "o", "Export": "s"}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
        "axes.labelsize": 8.5, "axes.edgecolor": "#444444", "axes.linewidth": 0.7,
        "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.55,
        "grid.alpha": 0.7, "legend.frameon": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, directory, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _p(value: float) -> str:
    if not np.isfinite(value):
        return "p=NA"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def _interval(ax, y, point, low, high, color="#31688E", marker="o") -> None:
    ax.plot([low, high], [y, y], color=color, lw=1.6)
    ax.scatter([point], [y], color=color, marker=marker, s=28, zorder=3)


def figure1(full_fit: pd.DataFrame, alpha_table: pd.DataFrame) -> None:
    # v2.1 expressly leaves Figure 1 unchanged. Copying the validated v2
    # artifacts byte-for-byte also makes that requirement independently auditable.
    stem = "Fig1_full_period_subquadratic_taylor_scaling"
    for suffix in (".png", ".pdf"):
        shutil.copy2(V2_OUT / "figures" / "main" / f"{stem}{suffix}", FIGURE_MAIN_DIR / f"{stem}{suffix}")
    shutil.copy2(
        V2_OUT / "figure_data" / "Fig1_country_data_v2.csv",
        FIGURE_DATA_DIR / "Fig1_country_data_v2.csv",
    )


def figure2(rolling: pd.DataFrame, paired_alpha: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True, sharey=True)
    for ax, crop in zip(axes.ravel(), CROP_ORDER):
        for direction in ("Import", "Export"):
            g = rolling[(rolling["crop"] == crop) & (rolling["direction"] == direction)].sort_values("end_year")
            ax.plot(g["end_year"], g["estimate"], color=COLORS[direction], marker=MARKERS[direction], ms=2.8, lw=1.1, label=direction)
            ax.fill_between(g["end_year"].to_numpy(float), g["percentile_ci_low"].to_numpy(float),
                            g["percentile_ci_high"].to_numpy(float), color=COLORS[direction], alpha=0.13, linewidth=0)
        p = paired_alpha[paired_alpha["crop"] == crop].iloc[0]
        ax.text(0.02, 0.03, f"Full-period Δα (I−E)={p['estimate']:.3f}\nresidual-block CI [{p['percentile_ci_low']:.3f}, {p['percentile_ci_high']:.3f}]",
                transform=ax.transAxes, fontsize=7.2, va="bottom")
        ax.axhline(2, color="#777777", ls="--", lw=0.8)
        ax.set_title(crop)
        ax.set_xlabel("Rolling 20-year window end")
        ax.set_ylabel("Taylor exponent α")
    axes[0, 0].legend(loc="upper left", ncol=2)
    fig.suptitle("Figure 2. Descriptive rolling estimates with residual moving-block uncertainty", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, FIGURE_MAIN_DIR, "Fig2_temporal_and_paired_direction")


def figure3(full_metrics: pd.DataFrame, decomposition: pd.DataFrame) -> None:
    eligible = full_metrics[full_metrics["eligible_covariance"] & full_metrics["phi"].gt(0)].copy()
    main = decomposition[(decomposition["scheme"] == "residual_two_stage") & (decomposition["block_length"] == 4)]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    for i, crop in enumerate(CROP_ORDER):
        ax = axes.ravel()[i]
        for direction in ("Import", "Export"):
            g = eligible[(eligible["crop"] == crop) & (eligible["direction"] == direction)]
            ax.scatter(np.log(g["D"]), np.log(g["variance"]), s=13, alpha=0.5, color=COLORS[direction], label=direction)
        ax.set_title(f"{chr(65+i)}. {crop}")
        ax.set_xlabel("log diagonal edge variance, D")
        ax.set_ylabel("log aggregate variance, V")
    axes[0, 0].legend(ncol=2)
    ax = axes[1, 1]
    wide = main.pivot(index=["crop", "direction"], columns="component", values="estimate").reset_index()
    for direction in ("Import", "Export"):
        g = wide[wide["direction"] == direction]
        ax.scatter(g["alpha_D"], g["alpha_V"], color=COLORS[direction], marker=MARKERS[direction], s=36, label=direction)
        for r in g.itertuples(): ax.annotate(r.crop[0], (r.alpha_D, r.alpha_V), xytext=(3, 3), textcoords="offset points", fontsize=7)
    lo = min(wide["alpha_D"].min(), wide["alpha_V"].min()) - 0.05
    hi = max(wide["alpha_D"].max(), wide["alpha_V"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], color="#777777", ls="--", lw=0.8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("αD"); ax.set_ylabel("αV"); ax.set_title("E. αV versus αD")
    ax = axes[1, 2]
    beta = main[main["component"] == "beta_phi"].copy()
    beta["label"] = beta["crop"] + " · " + beta["direction"]
    beta = beta.sort_values(["crop", "direction"])
    for y, r in enumerate(beta.itertuples()):
        _interval(ax, y, r.estimate, r.percentile_ci_low, r.percentile_ci_high, COLORS[r.direction], MARKERS[r.direction])
    ax.axvline(0, color="#777777", ls="--", lw=0.8)
    ax.set_yticks(np.arange(len(beta)), beta["label"])
    ax.set_xlim(-0.17, 0.15)
    ax.set_xlabel("βφ")
    ax.set_title("F. Synchronization elasticity βφ")
    fig.suptitle("Figure 3. Algebraic decomposition of the Taylor exponent", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, FIGURE_MAIN_DIR, "Fig3_algebraic_decomposition_taylor_exponent")


def figure4(pairs: pd.DataFrame, paired_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, crop in zip(axes.ravel(), CROP_ORDER):
        g = pairs[pairs["crop"] == crop]
        x, y = np.log(g["phi_import"]), np.log(g["phi_export"])
        ax.scatter(x, y, s=17, alpha=0.5, color="#5B7FA3", edgecolors="none")
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color="#333333", ls="--", lw=0.9)
        r = paired_summary[paired_summary["crop"] == crop].iloc[0]
        ax.text(0.03, 0.97, f"typical E/I φ ratio={r['typical_phi_ratio']:.2f}\n95% CI [{r['typical_phi_ratio_ci_low']:.2f}, {r['typical_phi_ratio_ci_high']:.2f}]\nexact sign {_p(r['exact_sign_test_p'])}",
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.set_title(crop); ax.set_xlabel("log φ import"); ax.set_ylabel("log φ export")
    fig.suptitle("Figure 4. Paired directional synchronization asymmetry", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, FIGURE_MAIN_DIR, "Fig4_directional_covariance_asymmetry")


def figure5(circular: pd.DataFrame) -> None:
    data = circular[circular["metric"].isin(["median_phi", "alpha"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, title in zip(axes, ("median_phi", "alpha"), ("A. Median φ", "B. Taylor exponent α")):
        g = data[data["metric"] == metric].copy().sort_values(["crop", "direction"])
        g["label"] = g["crop"] + " · " + g["direction"]
        y = np.arange(len(g))
        ax.scatter(g["null_median"], y, marker="o", facecolors="none", edgecolors="#777777", label="circular-shift null median")
        ax.scatter(g["observed"], y, marker="s", c=g["direction"].map(COLORS), label="observed")
        for yi, row in zip(y, g.itertuples()): ax.plot([row.null_median, row.observed], [yi, yi], color="#BBBBBB", lw=0.8)
        ax.set_yticks(y, g["label"]); ax.set_title(title); ax.set_xlabel(metric.replace("_", " "))
    axes[0].legend(fontsize=7)
    fig.suptitle("Figure 5. Circular-shift synchronization null (nonredundant metrics)", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, FIGURE_MAIN_DIR, "Fig5_temporal_synchronization_null")


def figure6(null_paired: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    g = null_paired.set_index("crop").loc[CROP_ORDER].reset_index()
    for y, row in enumerate(g.itertuples()):
        _interval(ax, y, row.paired_median_export_minus_import, row.bootstrap_ci_low, row.bootstrap_ci_high, "#31688E")
        ax.text(max(row.bootstrap_ci_high, row.paired_median_export_minus_import) + 0.02, y,
                f"sign {_p(row.exact_sign_test_p)}", va="center", fontsize=7.5)
    ax.axvline(0, color="#555555", ls="--", lw=0.9)
    ax.set_yticks(np.arange(len(g)), g["crop"])
    ax.set_xlabel("median Δsync export − import")
    ax.set_title("Figure 6. Country-level null-adjusted directional synchronization")
    fig.tight_layout()
    _save(fig, FIGURE_MAIN_DIR, "Fig6_country_level_null_adjusted_synchronization")


def supplement_figures(
    alpha: pd.DataFrame, full_metrics: pd.DataFrame, reporting_paired: pd.DataFrame,
    decomposition: pd.DataFrame, rolling: pd.DataFrame,
) -> list[str]:
    stems = []
    def done(fig, stem):
        _save(fig, FIGURE_SUPP_DIR, stem); stems.append(stem)

    mirror = pd.read_csv(V2_OUT / "figure_data" / "FigS1_global_mirror.csv")
    save_csv(mirror, FIGURE_DATA_DIR / "FigS1_global_mirror.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, crop in zip(axes.ravel(), CROP_ORDER):
        g = mirror[mirror["crop"] == crop]
        ax.plot(g["year"], g["export_import_ratio"], color="#31688E"); ax.axhline(1, color="#777777", ls="--")
        ax.set_title(crop); ax.set_ylabel("reported export/import ratio")
    fig.suptitle("S1. Global import–export mirror diagnostic"); fig.tight_layout(rect=(0,0,1,.96)); done(fig, "FigS1_global_import_export_mirror")

    rice = pd.read_csv(V2_OUT / "data" / "rice_code30_vs31_taylor_sensitivity.csv")
    save_csv(rice, FIGURE_DATA_DIR / "FigS2_rice_code30_vs31.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rice = rice.sort_values(["direction", "item_code"]); y = np.arange(len(rice))
    for yi, r in zip(y, rice.itertuples()): _interval(ax, yi, r.alpha, r.ci_low, r.ci_high, COLORS[r.direction], "o" if r.item_code == 30 else "s")
    ax.axvline(2, color="#777777", ls="--"); ax.set_yticks(y, [f"Code {r.item_code} · {r.direction}" for r in rice.itertuples()]); ax.set_xlabel("α")
    ax.set_title("S2. Rice Code 30 and Code 31 scope sensitivity"); fig.tight_layout(); done(fig, "FigS2_rice_code30_vs31_scope")

    activity = pd.read_parquet(V2_OUT / "data" / "reporter_year_reporting_activity.parquet")
    coverage = activity.groupby(["crop", "direction", "year"], as_index=False)["reporter_active"].sum()
    save_csv(coverage, FIGURE_DATA_DIR / "FigS3_reporting_coverage.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, crop in zip(axes.ravel(), CROP_ORDER):
        for direction in ("Import", "Export"):
            g = coverage[(coverage["crop"] == crop) & (coverage["direction"] == direction)]
            ax.plot(g["year"], g["reporter_active"], color=COLORS[direction], label=direction)
        ax.set_title(crop); ax.set_ylabel("active reporting economies/areas")
    axes[0,0].legend(); fig.suptitle("S3. Reporting coverage by year"); fig.tight_layout(rect=(0,0,1,.96)); done(fig, "FigS3_data_coverage")

    base = alpha[(alpha["window_id"] == "full_1986_2024") & (alpha["sample_type"] == "A_dynamic") & (alpha["method"] == "OLS_HC3")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, trend in enumerate(("raw", "linear", "quadratic")):
        g = base[base["trend"] == trend].sort_values(["crop", "direction"])
        x = np.arange(len(g)) + (i - 1) * .22; ax.scatter(x, g["estimate"], label=trend, s=24)
    labels = (base[base["trend"] == "linear"].sort_values(["crop","direction"])["crop"] + "\n" + base[base["trend"] == "linear"].sort_values(["crop","direction"])["direction"]).tolist()
    ax.set_xticks(np.arange(len(labels)), labels); ax.axhline(2,color="#777",ls="--"); ax.set_ylabel("α"); ax.legend(ncol=3); ax.set_title("S4. Detrending-method sensitivity")
    fig.tight_layout(); done(fig, "FigS4_detrending_method")

    win = alpha[(alpha["sample_type"] == "A_dynamic") & (alpha["trend"] == "linear") & (alpha["method"] == "OLS_HC3") & alpha["window_type"].isin(["rolling15","rolling20","rolling25"])]
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for ax,crop in zip(axes.ravel(),CROP_ORDER):
        for direction in ("Import","Export"):
            for wt,ls in (("rolling15",":"),("rolling20","-"),("rolling25","--")):
                g=win[(win.crop==crop)&(win.direction==direction)&(win.window_type==wt)].sort_values("end_year")
                ax.plot(g.end_year,g.estimate,color=COLORS[direction],ls=ls,lw=1,label=f"{direction} {wt[-2:]}")
        ax.set_title(crop); ax.axhline(2,color="#777",ls="--"); ax.set_ylabel("α")
    axes[0,0].legend(ncol=2,fontsize=6); fig.suptitle("S5. Prespecified 15-, 20-, and 25-year windows"); fig.tight_layout(rect=(0,0,1,.96)); done(fig,"FigS5_window_length")

    samp = alpha[(alpha.window_id=="full_1986_2024")&(alpha.trend=="linear")&(alpha.method=="OLS_HC3")]
    fig, ax=plt.subplots(figsize=(8.5,4.5));
    for i,s in enumerate(("A_dynamic","B_fixed")):
        g=samp[samp.sample_type==s].sort_values(["crop","direction"]); ax.scatter(np.arange(len(g))+(i-.5)*.2,g.estimate,label=s,s=28)
    labels=(g.crop+"\n"+g.direction).tolist(); ax.set_xticks(np.arange(len(labels)),labels); ax.axhline(2,color="#777",ls="--"); ax.set_ylabel("α"); ax.legend(); ax.set_title("S6. Dynamic and fixed reporting-economy samples"); fig.tight_layout(); done(fig,"FigS6_dynamic_fixed_samples")

    method = base[base.trend=="linear"].sort_values(["crop","direction"])
    fig, ax=plt.subplots(figsize=(8.5,4.5))
    for i,(col,label) in enumerate((("alpha","OLS"),("huber_alpha","Huber"),("theil_sen_alpha","Theil–Sen"))): ax.scatter(np.arange(len(method))+(i-1)*.2,method[col],label=label,s=26)
    labels=(method.crop+"\n"+method.direction).tolist(); ax.set_xticks(np.arange(len(labels)),labels); ax.axhline(2,color="#777",ls="--"); ax.set_ylabel("α"); ax.legend(ncol=3); ax.set_title("S7. OLS, Huber, and Theil–Sen estimates"); fig.tight_layout(); done(fig,"FigS7_robust_estimators")

    fig, ax=plt.subplots(figsize=(8,4.8)); loco=method.copy();
    for y,r in enumerate(loco.itertuples()): _interval(ax,y,r.estimate,r.loco_alpha_min,r.loco_alpha_max,COLORS[r.direction],MARKERS[r.direction])
    ax.axvline(2,color="#777",ls="--"); ax.set_yticks(np.arange(len(loco)),(loco.crop+" · "+loco.direction)); ax.set_xlabel("α leave-one-country-out range"); ax.set_title("S8. Leave-one-reporting-economy-out influence"); fig.tight_layout(); done(fig,"FigS8_leave_one_country_out")

    edge=pd.read_csv(V2_OUT/"figure_data"/"FigS9_edge_mean_variance.csv"); edge=edge[(edge["mean"]>0)&(edge["variance"]>0)]
    fig,axes=plt.subplots(2,2,figsize=(10,7));
    for ax,crop in zip(axes.ravel(),CROP_ORDER):
        g=edge[edge.crop==crop]; ax.hexbin(np.log(g["mean"]),np.log(g["variance"]),gridsize=35,mincnt=1,cmap="Blues"); ax.set_title(crop); ax.set_xlabel("log edge mean"); ax.set_ylabel("log edge variance")
    fig.suptitle("S9. Bilateral-edge Taylor scaling"); fig.tight_layout(rect=(0,0,1,.96)); done(fig,"FigS9_edge_level_taylor_scaling")

    div=pd.read_csv(V2_OUT/"tables"/"table_v2_source_diversification_sensitivity.csv"); save_csv(div,FIGURE_DATA_DIR/"FigS10_S11_diversification.csv")
    fig,ax=plt.subplots(figsize=(8,4.8)); lw=div[div.matrix_method=="ledoit_wolf"].copy(); x=np.arange(len(lw))
    for xi,r in zip(x,lw.itertuples()): _interval(ax,xi,r.median_gap_main_eligible,r.median_gap_ci_low,r.median_gap_ci_high,"#31688E")
    ax.axvline(0,color="#777",ls="--"); ax.set_yticks(x,[f"share {r.share_threshold:.1%}, coverage {r.coverage_threshold:.0%}" for r in lw.itertuples()]); ax.set_xlabel("median diversification gap"); ax.set_title("S10. Diversification-threshold sensitivity (no stable signal)"); fig.tight_layout(); done(fig,"FigS10_diversification_thresholds")

    fig,ax=plt.subplots(figsize=(8,4.8));
    for i,m in enumerate(("ledoit_wolf","sample")):
        g=div[div.matrix_method==m].sort_values(["share_threshold","coverage_threshold"]); ax.scatter(np.arange(len(g))+(i-.5)*.18,g.median_gap_main_eligible,label=m,s=30)
    ax.axhline(0,color="#777",ls="--"); ax.set_xticks(np.arange(len(g)),[f"{r.share_threshold:.1%}\n{r.coverage_threshold:.0%}" for r in g.itertuples()]); ax.set_ylabel("median gap"); ax.legend(); ax.set_title("S11. Correlation-matrix method sensitivity"); fig.tight_layout(); done(fig,"FigS11_correlation_matrix_method")

    spec=alpha[(alpha["transform"]=="tonnes") & alpha["estimate"].notna()].copy(); save_csv(spec,FIGURE_DATA_DIR/"FigS12_specification_curve.csv")
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharey=True)
    for ax,crop in zip(axes.ravel(),CROP_ORDER):
        g=spec[spec.crop==crop];
        for direction in ("Import","Export"):
            vals=np.sort(g[g.direction==direction].estimate.to_numpy()); ax.plot(np.linspace(0,1,len(vals)),vals,color=COLORS[direction],label=direction)
        ax.axhline(2,color="#777",ls="--"); ax.set_title(crop); ax.set_xlabel("prespecified specification quantile"); ax.set_ylabel("α")
    axes[0,0].legend(); fig.suptitle("S12. Prespecified specification curve (not a result search)"); fig.tight_layout(rect=(0,0,1,.96)); done(fig,"FigS12_specification_curve")

    aware=reporting_paired[reporting_paired.specification=="reporting_aware"].set_index("crop").loc[CROP_ORDER].reset_index()
    fig,ax=plt.subplots(figsize=(7.5,4.8));
    for y,r in enumerate(aware.itertuples()): _interval(ax,y,r.typical_phi_ratio,r.typical_phi_ratio_ci_low,r.typical_phi_ratio_ci_high,"#31688E")
    ax.axvline(1,color="#777",ls="--"); ax.set_yticks(np.arange(len(aware)),aware.crop); ax.set_xlabel("typical export/import φ ratio"); ax.set_title("S13. Reporting-aware paired synchronization"); fig.tight_layout(); done(fig,"FigS13_reporting_aware_paired_phi")

    main=decomposition[(decomposition.scheme=="residual_two_stage")&(decomposition.block_length==4)].pivot(index=["crop","direction"],columns="component",values="estimate").reset_index()
    fig,ax=plt.subplots(figsize=(6,5)); x=main.alpha_D+main.beta_phi; y=main.alpha_V
    ax.scatter(x,y,c=main.direction.map(COLORS),s=38); lo=min(x.min(),y.min())-.02; hi=max(x.max(),y.max())+.02; ax.plot([lo,hi],[lo,hi],color="#555",ls="--")
    for r,xx,yy in zip(main.itertuples(),x,y): ax.text(xx,yy,r.crop[0],fontsize=7)
    ax.set_xlabel("αD + βφ"); ax.set_ylabel("αV"); ax.set_title("S14. Algebraic identity validation"); fig.tight_layout(); done(fig,"FigS14_algebraic_identity")
    return stems


def create_all_figures_v2_1(
    full_fit: pd.DataFrame, alpha: pd.DataFrame, full_metrics: pd.DataFrame,
    rolling: pd.DataFrame, paired_alpha: pd.DataFrame, paired_phi: dict,
    decomposition: pd.DataFrame, circular: pd.DataFrame, null_adjusted: pd.DataFrame,
    reporting_paired: pd.DataFrame,
) -> dict:
    set_style()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure1(full_fit, alpha)
        figure2(rolling, paired_alpha)
        figure3(full_metrics, decomposition)
        figure4(paired_phi["pairs"], paired_phi["summary"])
        figure5(circular)
        figure6(null_adjusted)
        supplements = supplement_figures(alpha, full_metrics, reporting_paired, decomposition, rolling)
    return {
        "main": [
            "Fig1_full_period_subquadratic_taylor_scaling", "Fig2_temporal_and_paired_direction",
            "Fig3_algebraic_decomposition_taylor_exponent", "Fig4_directional_covariance_asymmetry",
            "Fig5_temporal_synchronization_null", "Fig6_country_level_null_adjusted_synchronization",
        ],
        "supplement": supplements,
    }
