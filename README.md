# Scaling Laws and Synchronization in Global Staple-Food Trade Networks

## Overview

This repository is a local, pre-publication release candidate for the analysis
of scale–fluctuation relationships and synchronization in global bilateral
trade networks for wheat, rice, maize, and soybeans. It preserves the archived
analysis logic while separating code, compact figure inputs, candidate results,
and non-distributable raw data.

The analysis uses annual FAOSTAT Detailed Trade Matrix quantities for 1986–2024
and treats import (element 5610) and export (element 5910) networks separately.
The final code detrends bilateral-edge time series, quantifies aggregate and
edge-specific variability, and studies how covariance-driven synchronization
relates to aggregate scaling.

## Research questions

1. How does aggregate fluctuation variance scale with the size of national
   import and export trade networks?
2. How do edge-specific variance and net covariance contribute to this scaling?
3. How does the synchronization factor differ between paired import and export
   networks and across crops?
4. Are these patterns stable across rolling windows, circular time-shift null
   models, reporting/coverage choices, and prespecified robustness checks?

## Data sources

The raw input is the FAOSTAT *Detailed trade matrix (Global, National –
Annual)* dataset, using quantity observations in tonnes. The target study
period is 1986–2024; item codes are 15 (Wheat), 30 (Rice), 56 (Maize), and 236
(Soybeans). The source archive is intentionally not distributed here. See
[`data/README.md`](data/README.md) for its expected members, checksum, source
links, attribution/reuse caution, and setup.

## Repository structure

```text
code/                 Formal entry points, analysis modules, tests, experiments
data/figure_data/     Compact, derived CSV inputs for panel rendering
results/              Candidate public tables and archived panel copies
work/                 Ignored regenerated artifacts (created when commands run)
CODE_RESULTS_MAP.md   Code–experiment–figure correspondence
REPRODUCIBILITY.md    Reproduction protocol and verified scope
RELEASE_CHECKLIST.md  Required author actions before publication
使用说明_中文.md         Chinese directory and update guide
```

`results/` is included for local review only. It must not be interpreted as
authorization to disclose unpublished figures or results.

## Requirements

- Python 3.12.x
- Packages pinned in [`code/requirements.txt`](code/requirements.txt)
- The exact FAOSTAT source ZIP for a full raw-data rerun

Create an isolated environment and install the pinned dependencies:

```bash
python -m pip install -r code/requirements.txt
```

## Installation

Obtain the exact source ZIP yourself and either put it in the ignored location
below or set `TRADE_ZIP` to its local path:

```text
data/raw/Trade_DetailedTradeMatrix_E_All_Data.zip
```

PowerShell example:

```powershell
$env:TRADE_ZIP = '.\path\to\Trade_DetailedTradeMatrix_E_All_Data.zip'
```

Confirm the SHA-256 in `data/README.md` before running any analysis.

## Reproducing the results

Render independent panels from the compact bundled CSVs without reprocessing
raw data:

```bash
python code/make_paper_figures.py
```

Run the complete archived analysis sequence (raw-data processing, formal
analyses, panel generation, and tests):

```bash
python code/reproduce_all.py
```

The latter is resource-intensive and has not been run to completion in the
present release environment. It writes only to ignored `work/`. For command
order, expected outputs, resource caution, and comparison rules, read
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Main figures and analyses

The archived inventory contains 23 main panels and 31 supplementary panels:

- Main 1: aggregate scale–fluctuation relationships;
- Main 2: rolling-window scaling;
- Main 3: variance/covariance decomposition;
- Main 4: paired import–export synchronization;
- Main 5: circular time-shift null comparison;
- Supplementary panels: scope, coverage, detrending, window, estimator,
  activity, edge, reporting-aware, identity, and null-model diagnostics.

See [`CODE_RESULTS_MAP.md`](CODE_RESULTS_MAP.md) for the authoritative
code-to-result mapping, including the two additional robustness workflows.

## Data availability

The source FAOSTAT archive is **not included**. It should be retrieved from
FAOSTAT and cited/attributed according to its current dataset metadata and
terms. The compact CSVs under `data/figure_data/` and candidate outputs under
`results/` require author confirmation before public release because this is a
pre-publication package.

## Code availability

This code-and-documentation package is available in the authors' designated
[private GitHub repository](https://github.com/Daisy30823/global-food-trade-scaling).
It intentionally excludes the FAOSTAT archive, candidate derived data, and
candidate results. No Zenodo record has been created. The authors must complete
[`RELEASE_CHECKLIST.md`](RELEASE_CHECKLIST.md) before any future public release
or archival deposit.

## Citation

**To be completed by the authors before public release.** Do not invent a DOI,
journal, or publication year. Suggested placeholder:

```text
[Author list]. [Manuscript title]. [Journal or preprint server, if applicable].
[Year]. DOI: [to be assigned].
```

Also cite the exact FAOSTAT Detailed Trade Matrix release accessed for the
study, including access date and URL.

## License

No repository license has been selected. This is intentional: authorship,
co-author consent, journal policy, and the applicable FAOSTAT/third-party
terms must be checked before choosing a code license and any separate data or
figure license. Until then, no permission to reuse this release candidate is
granted.
