# Code–experiment–figure map

This table maps the final analysis pathway as archived. It is a provenance map,
not evidence that every row has been rerun in this release.

| Analysis or result | Primary code | Inputs / outputs | Archived panels |
|---|---|---|---|
| FAOSTAT ZIP streaming, filtering, duplicate aggregation, importer/exporter panel construction | `code/analysis/data_prep.py`, `code/analysis/statistics.py` | Raw ZIP; four items (15, 30, 56, 236), elements 5610/5910, 1986–2024 | Upstream for all panels |
| Edge-wise linear detrending; aggregate variance `V`, edge variance sum `D`, covariance `C`, synchronization `phi=V/D` | `code/analysis/statistics.py` | Country-by-direction trade panels | Main 1, Main 3–4; Supplement 14 |
| Aggregate size–fluctuation scaling and country-level regressions | `code/analysis/analysis_runner.py` | Detrended country metrics | Main 1; Supplement 1–2, 6–8, 12 |
| Rolling 20-year scaling and 15/25-year sensitivity | `code/analysis/analysis_runner.py`, `code/analysis/v2_1.py` | Rolling panels; residual-block draws | Main 2; Supplement 5 |
| Variance/covariance decomposition, common-sample identity | `code/analysis/v2_1.py` | Full-period residual panels and bootstrap draws | Main 3; Supplement 14 |
| Paired import–export alpha and synchronization comparisons | `code/analysis/v2_1.py` | Matched-country panels, country / residual-block bootstrap | Main 3–4; Supplement 13 |
| Circular time-shift null model | `code/analysis/analysis_runner.py`, `code/analysis/v2_1.py` | Edge time series and deterministic seeds | Main 5; Supplement 15 |
| Reporting coverage, rice-code, edge, diversification, specification, and leave-one-out robustness | `code/analysis/advanced_v2.py`, `code/analysis/v2_1.py` | Prespecified sensitivity outputs | Supplement 1–13 |
| Formal v2 pipeline | `code/run_all.py`, `code/run_all_v2_1.py` | Raw ZIP -> `work/outputs_v2*` and reports | Supplies compact figure data |
| Independent one-panel-per-file figures | `code/make_paper_figures.py`, `code/analysis/paper_figures.py` | `data/figure_data/` or regenerated work data | 23 main + 31 supplementary panels |
| Added matrix/activity and macro-scaling robustness experiments | `code/experiments/run_robustness_experiments.py`, `code/experiments/run_scaling_stability.py` | Raw ZIP; writes only `work/` | `results/additional_robustness/` |

`code/analysis/figures_v2_1.py` also exposes the four manuscript-oriented
builders `figure1`, `figure2`, `figure3`, and `figure4`, followed by
`supplement_figures`. The independent-panel renderer further separates the
circular-null comparison as `M5_01_null_adjusted_direction`; this is why the
archived panel inventory has five logical main groups although the v2.1 module
has four numbered primary-figure builders.

## Archived figure inventory

`results/figures_index_archived.csv` lists 54 independent panels: 23 `Main*`
rows and 31 `Supplement*` rows. The archived release copies are SVG+PNG;
the code renderer writes PDF+PNG into `work/paper_figures/`. This format
discrepancy is recorded in `RELEASE_CHECKLIST.md` and must be resolved before
claiming byte-level figure reproduction.

## Version status

`code/legacy/` holds three unreferenced earlier plotting/reporting modules
(`figures.py`, `reporting.py`, `reporting_v2.py`) for audit only. They are not
imported by the current entry points and should not be used to reproduce the
submission results unless their provenance is independently confirmed.
