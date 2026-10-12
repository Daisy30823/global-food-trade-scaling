from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CROPS, DATA_DIR, REPORT_DIR, ROOT, SEED
from .utils import write_json


def _main_alpha(alpha: pd.DataFrame) -> pd.DataFrame:
    return alpha[
        (alpha["window"] == "1986-2024")
        & (alpha["sample_type"] == "A_dynamic")
        & (alpha["trend"] == "linear")
        & (alpha["transform"] == "tonnes")
        & (alpha["method"] == "OLS_HC3")
    ].copy()


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def choose_narrative(alpha: pd.DataFrame, full_metrics: pd.DataFrame, phi_boot: pd.DataFrame, div_full: pd.DataFrame, null_summary: pd.DataFrame) -> tuple[str, str]:
    main = _main_alpha(alpha)
    delta_negative = bool(len(main) and (main["delta_ci_high"] < 0).mean() >= 0.75)
    alpha_subquadratic = bool(len(main) and (main["ci_high"] < 2).mean() >= 0.75)
    phi = full_metrics[full_metrics["eligible_covariance"]].groupby(["crop", "direction"])["phi"].median().unstack()
    direction_lower = bool(len(phi) and (phi.get("Import", np.nan) < phi.get("Export", np.nan)).mean() >= 0.75)
    import_near_or_below = bool("Import" in phi and (phi["Import"] <= 1.1).mean() >= 0.75)
    significant_direction = bool(len(phi_boot) and (phi_boot["ci_low"] > 0).mean() >= 0.75)
    gap = div_full["Gap"].dropna()
    positive_gap = bool(len(gap) and gap.median() > 0 and (gap > 0).mean() >= 0.60)
    null_phi = null_summary[null_summary["metric"] == "median_phi"]
    synchronized = bool(len(null_phi) and ((null_phi["observed"] > null_phi["null_q975"]) & (null_phi["empirical_p_value"] < 0.05)).mean() >= 0.75)

    if alpha_subquadratic and delta_negative and import_near_or_below and direction_lower:
        return "A", "稳定的次二次标度与下降的相对波动最一致；进口来源协方差总体弱于出口目的地协方差，部分结果支持来源替代。"
    if positive_gap and synchronized and bool((main["estimate"].sub(2).abs() < 0.25).mean() >= 0.50):
        return "B", "名义来源数量系统性高于相关性调整后的有效来源数，且真实同期共变高于循环位移零模型。"
    if direction_lower and significant_direction:
        return "C", "相近的宏观标度可能掩盖进口来源与出口目的地之间不同的协方差结构。"
    return "Conservative", "结果主要支持描述性的 Taylor 标度；现有证据不足以稳定归入协方差陷阱、贸易分散有效或方向不对称三种强机制叙事。"


def build_machine_summary(
    prep: dict,
    table1: pd.DataFrame,
    alpha: pd.DataFrame,
    model_comparison: pd.DataFrame,
    full_metrics: pd.DataFrame,
    phi_boot: pd.DataFrame,
    div_full: pd.DataFrame,
    null_summary: pd.DataFrame,
    tests_passed: int,
) -> dict:
    main = _main_alpha(alpha)
    narrative, narrative_text = choose_narrative(alpha, full_metrics, phi_boot, div_full, null_summary)
    alpha_records = main[["crop", "direction", "estimate", "ci_low", "ci_high", "delta", "delta_ci_low", "delta_ci_high", "n_countries"]].to_dict("records")
    model_main = model_comparison[model_comparison["sample_type"] == "A_dynamic"]
    model_counts = model_main["better_model_cv_rmse"].value_counts().to_dict()
    phi_medians = (
        full_metrics[full_metrics["eligible_covariance"]]
        .groupby(["crop", "direction"])["phi"]
        .median()
        .rename("median_phi")
        .reset_index()
        .to_dict("records")
    )
    gap = div_full["Gap"].dropna()
    robust_over_two = main[(main["ci_low"] > 2) & (main["bootstrap_ci_low"] > 2)]
    return {
        "seed": SEED,
        "years": [int(table1["year_start"].min()), int(table1["year_end"].max())],
        "crop_codes": {str(k): v for k, v in CROPS.items()},
        "reporter_countries": int(prep["reporter_countries_retained"]),
        "partner_countries": int(prep["partner_countries_retained"]),
        "bilateral_crop_direction_edges": int(prep["bilateral_edges"]),
        "alpha": alpha_records,
        "robust_alpha_above_two": bool(len(robust_over_two)),
        "robust_alpha_above_two_groups": robust_over_two[["crop", "direction"]].to_dict("records"),
        "model_cv_winner_counts": {str(k): int(v) for k, v in model_counts.items()},
        "phi_medians": phi_medians,
        "phi_export_minus_import_bootstrap": phi_boot.to_dict("records"),
        "diversification": {
            "n_with_N_corr": int(len(gap)),
            "median_N_nominal_matched": float(div_full.loc[div_full["Gap"].notna(), "N_nominal_matched"].median()),
            "median_N_corr": float(div_full.loc[div_full["Gap"].notna(), "N_corr"].median()),
            "median_gap": float(gap.median()),
            "share_gap_positive": float((gap > 0).mean()),
        },
        "null_significant_005": int((null_summary["empirical_p_value"] < 0.05).sum()),
        "null_tests_total": int(len(null_summary)),
        "narrative": narrative,
        "narrative_text": narrative_text,
        "tests_passed": int(tests_passed),
    }


def _alpha_markdown(main: pd.DataFrame) -> str:
    lines = ["| 作物 | 方向 | alpha (HC3 95% CI) | delta (HC3 95% CI) | n |", "|---|---|---:|---:|---:|"]
    for row in main.sort_values(["crop", "direction"]).itertuples():
        lines.append(
            f"| {row.crop} | {row.direction} | {_fmt(row.estimate)} [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] | "
            f"{_fmt(row.delta)} [{_fmt(row.delta_ci_low)}, {_fmt(row.delta_ci_high)}] | {int(row.n_countries)} |"
        )
    return "\n".join(lines)


def write_reports(
    prep: dict,
    table1: pd.DataFrame,
    results: dict,
    diversification: dict,
    null: dict,
    phi_boot: pd.DataFrame,
    tests_passed: int,
) -> dict:
    alpha = results["alpha"]
    full_metrics = results["full_metrics"]
    model = results["model_comparison"]
    div_full = diversification["full"]
    null_summary = null["summary"]
    summary = build_machine_summary(prep, table1, alpha, model, full_metrics, phi_boot, div_full, null_summary, tests_passed)
    write_json(summary, REPORT_DIR / "results_summary.json")
    main = _main_alpha(alpha)
    rice_sensitivity = pd.read_csv(DATA_DIR / "rice_code30_vs31_taylor_sensitivity.csv")
    rice31_text = "; ".join(
        f"{row.direction} alpha={_fmt(row.alpha)} [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}]"
        for row in rice_sensitivity[rice_sensitivity["item_code"] == 31].itertuples()
    )

    model_main = model[model["sample_type"] == "A_dynamic"]
    winner_text = ", ".join(f"{k}: {v}" for k, v in summary["model_cv_winner_counts"].items())
    delta_negative = int((main["delta"] < 0).sum())
    phi_lines = []
    for row in phi_boot.itertuples():
        phi_lines.append(f"- {row.crop}: export − import = {_fmt(row.estimate)} [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}], p={_fmt(row.p_two_sided)}")
    null_sig = null_summary[null_summary["empirical_p_value"] < 0.05]
    results_md = f"""# Results

## Scope

The analysis uses only `Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv` inside the supplied archive. It covers {summary['years'][0]}–{summary['years'][1]}, four crop codes ({', '.join(summary['crop_codes'])}), and independently reported import and export quantities in tonnes. Aggregate FAOSTAT entities with codes at or above 5000 were removed by code.

## Full-period Taylor scaling

{_alpha_markdown(main)}

The exact identity `alpha = 2 + delta` holds for every fitted group to numerical tolerance below `1e-8`. Delta is negative in {delta_negative} of {len(main)} full-period crop-direction groups. A robust exponent above two requires both the HC3 and country-bootstrap lower 95% limits to exceed two; this occurs in {len(summary['robust_alpha_above_two_groups'])} groups.

Rice Item 31 sensitivity gives {rice31_text}. Item 30 remains the main definition because it is already expressed in milled-rice equivalent and is not added to codes 27–32.

## Power law versus linear–quadratic crossover

Leave-one-country-out log-scale RMSE selects: {winner_text}. Model-level RMSE, MAE, AIC, BIC, nonnegative M2 coefficients, and residuals are reported in `outputs/tables/table3_model_comparison.csv` and `outputs/figure_data/Fig1_model_fits.csv`. This comparison is predictive/descriptive, not causal.

## Covariance and directional asymmetry

The decomposition is exact up to floating-point error: `V = D + C`. The crop-stratified bootstrap estimates of the difference in median synchronization amplification are:

{chr(10).join(phi_lines)}

Negative covariance shares are retained. Values of `phi < 1` indicate net offsetting across edges; values above one indicate synchronous amplification.

## Diversification

For the {summary['diversification']['n_with_N_corr']} country-crop observations satisfying the 1% source-share, five-positive-year, positive-residual-variance and 80% coverage rules, median matched nominal effective suppliers are {_fmt(summary['diversification']['median_N_nominal_matched'])}, versus median correlation-adjusted suppliers of {_fmt(summary['diversification']['median_N_corr'])}. Median `Gap` is {_fmt(summary['diversification']['median_gap'])}; the share with positive gaps is {_fmt(summary['diversification']['share_gap_positive'])}.

## Circular-shift null model

The null preserves each bilateral edge's residual sequence and independently changes its temporal alignment. {len(null_sig)} of {len(null_summary)} crop-direction-metric comparisons have two-sided empirical p-values below 0.05. These tests isolate the importance of contemporaneous edge covariance; they do not identify climate, policy, production, price, or food-security mechanisms.

## Evidence-weighted narrative

**{summary['narrative']}** — {summary['narrative_text']}

Rolling windows overlap heavily and are descriptive trajectories, not independent observations or evidence of a phase transition.
"""
    (REPORT_DIR / "results_report.md").write_text(results_md, encoding="utf-8")

    methods_md = f"""# Methods

## Data and filtering

The sole source is the supplied FAOSTAT Detailed Trade Matrix archive. The pipeline streams the `NOFLAG` table in 40,000-row chunks, filters before reshaping, and uses Element 5610 for importer-reported quantities and Element 5910 for exporter-reported quantities. The focal items are Wheat (15), Rice paddy in milled equivalent (30), Maize (56), and Soya beans (236). Rice codes 27–32 are diagnosed separately; code 31 is retained as a sensitivity dataset and is never added to code 30. Years are Y1986–Y2024 and the unit is tonnes. Blank year cells in an existing bilateral record and explicit zeros are both retained as zero; no interpolation or imputation is used.

Reporter is the importing country and Partner is the source in Element 5610 (`j → i`). Reporter is the exporting country and Partner is the destination in Element 5910 (`i → j`). The two reporting systems are never added or averaged. FAOSTAT geography codes below 5000 are retained; aggregate entities at or above 5000 are removed. Full kept/removed lists are in `outputs/data`.

## Taylor scaling and samples

Country totals are sums across partners. For each window, within-country total flows are linearly detrended while their arithmetic mean is calculated on the original tonne series. The residual sample variance uses `ddof=1`. Eligibility requires positive mean and variance, at least 20 positive years in 1986–2024, at least 15 in a 20-year window, and 75% positive years in 15-, 19-, 24-, and 25-year sensitivity windows. Dynamic sample A is reselected in each window; fixed sample B is the intersection across all 20-year windows. Countries with one active partner remain eligible for Taylor scaling but not covariance analysis.

Log variance is regressed on log mean by OLS with HC3 standard errors. Huber and Theil–Sen estimates, 1,000 country bootstrap draws, leave-one-country-out ranges, Cook's distance, and tests against alpha=1 and alpha=2 are reported. Windows comprise 1986–2024; 20-year rolling windows ending 2005–2024; non-overlapping 1986–2004 and 2005–2024 periods; and 15- and 25-year sensitivities. Raw, linear, quadratic, and country-total `log1p(F)` trend specifications are kept distinct.

## Scale identity and crossover model

For each country, `CV²=V/mu²`, implying `log(V)=2 log(mu)+log(CV²)`. The OLS slope delta from log CV² on log mean is verified against the direct Taylor exponent by `alpha=2+delta`, with tolerance `1e-8`. M1 is the pure power law. M2 is `V=a mu+b mu²`, fitted by nonnegative log-parameter optimization to minimize squared log residuals. Models are compared by leave-one-country-out log RMSE/MAE, in-sample log RMSE/MAE, AIC and BIC.

## Covariance, diversification, and null model

Every bilateral edge is linearly detrended within the same window. Total residual variance is decomposed into `D`, the sum of edge variances, and `C=V-D`, the full off-diagonal covariance contribution. `phi=V/D`, `C/D=phi-1`, and `CovarianceShare=C/V`; negative values are not truncated. Identity checks use `max(1e-8, 1e-8*V)`.

Import diversification uses window-total shares. `N_nominal=1/sum(s²)`. Main correlation matrices retain sources with at least 1% share, five positive years and positive residual variance, require at least 80% coverage, renormalize retained shares, and use Ledoit–Wolf shrinkage. `N_corr=1/(s'Rs)` and `Gap=log(N_nominal_matched)-log(N_corr)`. Sensitivities span 0.5/1/2% shares, 70/80/90% coverage and sample versus Ledoit–Wolf correlations.

The synchronization null independently circularly shifts each detrended edge sequence 1,000 times using seed {SEED}, preserving its values, mean, variance, length and circular autocorrelation while breaking cross-edge calendar alignment. Observed statistics are compared with null means, medians, 2.5%/97.5% quantiles, empirical p-values and z-scores. Full null draws are retained for the main figure, and complete summary statistics are produced for every unique full, rolling, non-overlapping and sensitivity window.

## Reproducibility

Run `python run_all.py` from the project root. Formal figures and narrative reports are generated only after all ten automated invariance tests pass. Randomized procedures use seed {SEED} or deterministic sub-seeds derived from it.
"""
    (REPORT_DIR / "methods.md").write_text(methods_md, encoding="utf-8")

    captions = """# Figure captions

**Figure 1. Full-period Taylor scaling.** Country-level mean total trade and the sample variance of linearly detrended total-flow residuals, 1986–2024, in tonnes. Columns are Wheat, Rice (Item 30), Maize and Soybeans; rows are importer- and exporter-reported quantities. Solid lines are M1 power-law fits, dashed lines are nonnegative M2 linear–quadratic fits, and shading is the 95% country-bootstrap band. Labels identify the three largest Cook's distances per panel.

**Figure 2. Taylor exponent estimates and temporal evolution.** Panel A shows full-period HC3 estimates. Panels B and C show 20-year rolling estimates for dynamic and fixed country samples; adjacent windows overlap heavily and the trajectories are descriptive rather than independent time observations. Panel D compares full-period import and export exponents using country-bootstrap intervals.

**Figure 3. Why the Taylor exponent does not exceed two.** Panels A–D show squared coefficient of variation against mean trade scale. Panel E verifies the exact OLS identity `alpha=2+delta`. Panel F shows the local effective exponent implied by M2 across observed trade-scale quantiles; when both nonnegative components contribute, it lies between one and two.

**Figure 4. Variance decomposition and directional asymmetry.** Panel A compares `phi=V/D`; Panel B relates the normalized covariance term `C/D` to trade scale; Panel C classifies countries using the prespecified 0.9–1.1 near-independence band; Panel D gives crop-stratified bootstrap differences in export minus import median phi. Negative covariance contributions are retained.

**Figure 5. Nominal versus correlation-adjusted import diversification.** Panel A compares matched nominal and Ledoit–Wolf correlation-adjusted effective supplier counts. Panel B shows the dynamic diversification gap. Panel C relates nominal supplier diversity to synchronization amplification. Panel D gives rolling-window medians with country-bootstrap intervals. Missing adjusted counts reflect failure of the prespecified source-selection or 80% coverage rule.

**Figure 6. Circular-shift temporal null model.** Observed median phi, median C/D and Taylor alpha are compared with 1,000 independent edge-level circular-shift realizations. The final panel reports observed-minus-null z-scores. The null tests contemporaneous covariance only and does not identify the cause of synchronization.
"""
    (REPORT_DIR / "figure_captions.md").write_text(captions, encoding="utf-8")
    return summary
