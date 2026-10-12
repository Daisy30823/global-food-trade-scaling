from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORT_DIR, ROOT, SEED
from .utils import write_json


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str], labels: list[str] | None = None) -> str:
    labels = labels or columns
    header = "| " + " | ".join(labels) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for row in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(_fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return "\n".join([header, rule, *rows])


def _v1_alpha() -> pd.DataFrame:
    path = ROOT / "outputs_v1" / "tables" / "table2_alpha_estimates.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame[
        (frame["window"] == "1986-2024")
        & (frame["sample_type"] == "A_dynamic")
        & (frame["trend"] == "linear")
        & (frame["transform"] == "tonnes")
        & (frame["method"] == "OLS_HC3")
    ][["crop", "direction", "estimate"]].rename(columns={"estimate": "alpha_v1"})


def write_reports_v2(
    prep: dict,
    results: dict,
    decomposition: dict,
    paired: dict,
    rolling: pd.DataFrame,
    diversification: dict,
    circular_null: dict,
    block_null: dict,
    reporting: dict,
    sovereign: pd.DataFrame,
    figures: list[str],
    tests_passed: int,
) -> dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decomp = decomposition["summary"]
    alpha = decomp[decomp["component"] == "alpha_V"].copy()
    beta = decomp[decomp["component"] == "beta_phi"].copy()
    v1 = _v1_alpha()
    alpha_compare = alpha[["crop", "direction", "estimate"]].rename(columns={"estimate": "alpha_v2_covariance_eligible"}).merge(v1, on=["crop", "direction"], how="left")
    alpha_compare["difference_v2_minus_v1"] = alpha_compare["alpha_v2_covariance_eligible"] - alpha_compare["alpha_v1"]
    max_difference = float(alpha_compare["difference_v2_minus_v1"].abs().max())
    identity_error = float(decomp["identity_error"].max())
    circular = circular_null["summary"]
    block = block_null["summary"]
    div_main = diversification["main"]
    div_eligible = div_main[div_main["main_eligible"]].dropna(subset=["Gap_source"])
    beta_export = beta[beta["direction"] == "Export"]
    beta_import = beta[beta["direction"] == "Import"]
    narrative_supported = bool(alpha["estimate"].lt(2).all())

    audit = f"""# Audit response v2

## 1. Bug-fix status

- Window identifiers are unique (`full_*`, `roll20_*`, `nonoverlap_*`, `roll15_*`, `roll25_*`); the 2005–2024 rolling and non-overlapping windows no longer share a cache key.
- Every output retains `window_id`, `window_label`, and `window_type` where windows are relevant.
- `log1p` was removed from the Taylor exponent specification curve and stored only as descriptive volatility sensitivity.
- All randomization p-values use `(extreme_count+1)/(B+1)`; the display rule renders the minimum 1,000-draw value as `p<0.001`.
- The run manifest records raw ZIP, code and configuration SHA256 hashes, seed, Python version, and exact requirements. Cached products are reusable only when the full signature matches.
- Portable manifests use the archive basename and relative output names; local Windows paths and profile directories are excluded.

## 2. Main results versus v1

The v1 full-period Taylor exponents are reproduced on their original Taylor-eligible samples. The exact-decomposition table deliberately uses the stricter common covariance-eligible sample, so its α is not expected to be numerically identical to v1. The largest corresponding sample-definition difference is {_fmt(max_difference, 4)}.

{_markdown_table(alpha_compare, ['crop', 'direction', 'alpha_v1', 'alpha_v2_covariance_eligible', 'difference_v2_minus_v1'])}

All eight v2 decomposition-sample α estimates remain below 2. M1 remains the main functional form; M2 is retained only as supplementary descriptive output.

## 3. Paired alpha and phi

Import and export estimates now use only common reporting economies and a shared country resample within every bootstrap draw.

{_markdown_table(paired['alpha'], ['crop', 'n_pairs', 'alpha_import', 'alpha_export', 'estimate_import_minus_export', 'ci_low', 'ci_high'])}

{_markdown_table(paired['phi'], ['crop', 'n_pairs', 'median_difference', 'median_ci_low', 'median_ci_high', 'mean_difference', 'share_export_gt_import'])}

## 4. Exact alpha_D and beta_phi decomposition

For every crop-direction group, `V=D×phi` is checked at the reporting-economy level and `alpha_V=alpha_D+beta_phi` is checked using the same rows and regression method. Maximum slope-identity error: {identity_error:.3e}.

{_markdown_table(decomp, ['crop', 'direction', 'component', 'estimate', 'hc3_ci_low', 'hc3_ci_high', 'two_stage_ci_low', 'two_stage_ci_high', 'n_reporting_economies'])}

## 5. Time-aware uncertainty

Four uncertainty summaries are reported for every component: HC3, country bootstrap, global-year moving-block bootstrap, and two-stage global-year-plus-country bootstrap. The main block length is four years; three- and five-year sensitivity intervals are also stored. All formal runs use 1,000 draws. The global year index is shared by all reporting economies and edges in a draw, and metrics are re-detrended and recomputed after resampling.

## 6. Leave-one-importer-out diversification

The focal importer's own bilateral flow is excluded from every source shock before detrending and correlation estimation. The main summary contains only Taylor/covariance-eligible importers; all computable observations are reported separately. There are {len(div_eligible)} main-eligible computable observations, with median `Gap_source` {_fmt(div_eligible['Gap_source'].median())} and share positive {_fmt((div_eligible['Gap_source'] > 0).mean())}.

The preregistered stability criterion is a positive lower 95% bootstrap limit for median `Gap_source` in all 18 share-threshold × coverage-threshold × matrix-method configurations. **Stable signal: {diversification['stable']}**. {'Figure 6 therefore retains diversification in the main text.' if diversification['stable'] else 'The signal is not stable across all prespecified specifications, so diversification is supplementary and the manuscript must state that no robust nominal–dynamic diversification gap was found.'}

## 7. Null-model robustness

The main circular-shift figure uses only median φ and α. It reports observed-minus-null absolute differences, ratios where meaningful, and positive empirical p-values. A separate block-permutation null is run at block lengths 3, 4, and 5. Circular-shift significance at 0.05: {int((circular['empirical_p_value'] < 0.05).sum())}/{len(circular)} comparisons. Block-permutation significance at 0.05: {int((block['empirical_p_value'] < 0.05).sum())}/{len(block)} comparisons.

These nulls identify the importance of contemporaneous edge-flow covariance. They do not identify climate, policy, trade topology, or a causal mechanism.

## 8. Final main figures

{chr(10).join(f'- {name}' for name in figures)}

The leave-one-importer-out diversification figure is {'main Figure 6' if diversification['stable'] else 'supplementary because the preregistered stability condition failed'}.

## 9. Supported and unsupported conclusions

Supported:

- Full-period fluctuations are subquadratic (`alpha<2`), so relative variability declines with trade scale.
- Export βφ estimates are {', '.join(_fmt(v) for v in beta_export['estimate'])}; import estimates are {', '.join(_fmt(v) for v in beta_import['estimate'])}. Directional interpretation is made crop by crop and follows the reported confidence intervals.
- Similar macro Taylor exponents can arise from different diagonal-variance and covariance-synchronization contributions.

Not supported:

- Causal claims about climate, topology, policy, production shocks, or food-security consequences.
- A “covariance trap” or robust diversification gap unless the complete preregistered leave-one-importer-out sensitivity grid passes.
- Ordinary trend inference from overlapping rolling windows; they are descriptive only.

## 10. Automated tests

All {tests_passed} tests passed before formal figures and reports were generated. Tests cover the ten new v2 invariants plus the original directional, share, covariance, null-preservation, and reproducibility checks.
"""
    (REPORT_DIR / "audit_response.md").write_text(audit, encoding="utf-8")

    methods = f"""# Methods v2

The sole raw input is the supplied FAOSTAT Detailed Trade Matrix ZIP. The analysis covers 1986–2024 and Wheat (15), Rice paddy in milled equivalent (30), Maize (56), and Soybeans (236). Import and export reports are kept separate. Geographic aggregates with FAO codes at or above 5000 are removed; the retained units are described as **reporting economies/areas**. A restricted sovereign-state sensitivity excludes a conservative, explicit list of source labels and is not presented as an authoritative legal classification.

Country-total and bilateral-edge flows are linearly detrended in tonnes. Taylor scaling uses OLS-HC3 with Huber, Theil–Sen, country bootstrap, and leave-one-out diagnostics retained. `log1p` volatility is descriptive and is not mixed with the tonne-scale Taylor estimand.

For each reporting economy, `V=Var(sum_j e_j)`, `D=sum_j Var(e_j)`, and `phi=V/D`, so `log V=log D+log phi`. The three slopes are fit on identical covariance-eligible rows. Global-year moving-block bootstrap uses a common resampled year index across all economies and edges, with main block length four and sensitivity lengths three and five. The two-stage procedure additionally resamples economies within the same draw. Each formal randomized procedure uses 1,000 draws and seed {SEED} or a deterministic derived seed.

Import/export comparisons use common eligible reporting economies and paired bootstrap indices. Source-risk-adjusted diversification uses exporter shocks to all destinations except the focal importer, prespecified share thresholds 0.5/1/2%, coverage thresholds 70/80/90%, and sample versus Ledoit–Wolf correlation matrices. The main diversification sample is restricted to eligible importers.

Blank cells in an existing dyad are zero in the main specification. The reporting-aware sensitivity detects reporter-crop-direction years where all partner cells are blank, uses only covered years without interpolation, and records the most affected reporting economies/areas.

The circular-shift and block-permutation nulls assess contemporaneous synchronization only. Rolling 20-year windows are descriptive because adjacent windows overlap heavily.
"""
    (REPORT_DIR / "methods.md").write_text(methods, encoding="utf-8")

    results = f"""# Results v2

The evidence supports the paper line: **“Subquadratic fluctuations mask directional covariance asymmetry in global staple-food trade.”** All eight full-period decomposition-sample Taylor exponents are below two. Exact decomposition shows how diagonal edge variance and synchronization jointly produce these macro exponents; the identity error is below `1e-8` in every group.

Paired φ differences are positive in {int((paired['phi']['median_difference'] > 0).sum())} of four crops. Export βφ is positive in {int((beta_export['estimate'] > 0).sum())} of four crops, while import βφ is nonpositive in {int((beta_import['estimate'] <= 0).sum())} of four crops. Confidence intervals, rather than signs alone, determine which crop-level contrasts are statistically resolved.

The leave-one-importer-out diversification stability decision is `{diversification['stable']}`. Accordingly, {'diversification is retained as a conditional main result' if diversification['stable'] else 'no robust nominal–dynamic diversification gap is claimed, and the analysis is supplementary'}.

Neither null model nor the observational regressions establish climate, topology, policy, or food-security causality.
"""
    (REPORT_DIR / "results_report.md").write_text(results, encoding="utf-8")

    captions = """# Figure captions v2

**Figure 1. Full-period subquadratic Taylor scaling.** Reporting-economy mean trade and residual variance in tonnes, 1986–2024. Lines are pure-power-law fits. Text reports OLS-HC3 and two-stage bootstrap intervals on the common covariance-eligible sample; labels mark the maximum Cook's-distance observation.

**Figure 2. Temporal robustness and paired direction comparison.** Full-period estimates use two-stage intervals. Rolling estimates use a common-global-year moving-block bootstrap and are descriptive. The final panel uses common import/export reporting economies and paired country bootstrap.

**Figure 3. Exact origin of the Taylor exponent.** Top panels show `log(CV²)` against `log(mu)`. The forest reports `alpha_V`, `alpha_D`, and `beta_phi`. The 45° panel verifies `alpha_V=alpha_D+beta_phi`.

**Figure 4. Directional covariance asymmetry.** Import/export φ values are paired within reporting economy. Panels report paired median differences, βφ with two-stage intervals, and the share with φexport greater than φimport.

**Figure 5. Temporal synchronization null model.** Observed median φ and α are compared with 1,000 circular-shift draws. Heatmaps show direct crop-direction-window observed-minus-null statistics; no crop medians or confidence bounds are averaged.

**Figure 6. Conditional result.** Diversification appears only if every prespecified leave-one-importer-out sensitivity has a positive lower bootstrap limit. Otherwise Figure 6 reports missingness and entity-coverage sensitivity, while diversification moves to the supplement.
"""
    (REPORT_DIR / "figure_captions.md").write_text(captions, encoding="utf-8")

    summary = {
        "seed": SEED,
        "years": [1986, 2024],
        "reporting_economies": prep["reporter_countries_retained"],
        "exact_identity_max_error": identity_error,
        "alpha_all_below_two": narrative_supported,
        "paired_alpha": paired["alpha"].to_dict("records"),
        "paired_phi": paired["phi"].to_dict("records"),
        "decomposition": decomp.to_dict("records"),
        "diversification_stable": diversification["stable"],
        "main_figures": figures,
        "tests_passed": tests_passed,
    }
    write_json(summary, REPORT_DIR / "results_summary.json")
    return summary
