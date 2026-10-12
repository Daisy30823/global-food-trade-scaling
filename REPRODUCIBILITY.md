# Reproducibility protocol

## Verified scope

The code is configured for Python 3.12 and the pinned packages in
`code/requirements.txt`. Path portability was checked by static compilation.
The figure-only command was run using the bundled compact data and produced a
complete 54-row index with 54 PDF and 54 PNG files. The full raw-data pipeline
and its 2,000-draw analyses have **not** been executed in this release
environment; no numerical claim should be inferred from the static or
figure-rendering checks.

## Inputs and configuration

1. Download the exact FAOSTAT archive described in `data/README.md`.
2. Verify SHA-256:

   ```powershell
   Get-FileHash -Algorithm SHA256 $env:TRADE_ZIP
   ```

3. Point the code at it (do not commit it):

   ```powershell
   $env:TRADE_ZIP = '.\path\to\Trade_DetailedTradeMatrix_E_All_Data.zip'
   ```

   Alternatively place the ZIP in the ignored path `data/raw/`.

## Environment

```bash
python --version  # expected: Python 3.12.x
python -m pip install -r code/requirements.txt
```

The archive expands to multi-gigabyte CSV files and the formal pipeline runs
multiple 2,000-draw bootstrap/null-model analyses. It was not resource-
benchmarked here: provision substantial temporary disk space and run a small
environment check before attempting the full command.

## Commands

Generate panels from the included compact result data only:

```bash
python code/make_paper_figures.py
```

This creates PDF+PNG panels and `figure_index.csv` under ignored
`work/paper_figures/`; it does not recalculate trade statistics.

Run the full preregistered/final archived sequence:

```bash
python code/reproduce_all.py
```

This runs v2, v2.1, regenerates figure data, creates panels, and runs tests.
All derived files remain in ignored `work/`.

Run the two added robustness workflows only after the full environment has
been installed and the raw ZIP is available:

```bash
python code/experiments/run_robustness_experiments.py
python code/experiments/run_scaling_stability.py
```

The second command consumes the first command's `work/robustness_experiments/`
outputs. Both preserve the supplied sample rules, seed (`20260902`), years,
and their 2,000-draw setting.

## What must be compared after a full rerun

- the input ZIP SHA-256 above;
- `work/outputs_v2*/data/run_manifest.json` and output inventories;
- 54 rows in the generated figure index (23 Main, 31 Supplement);
- numeric CSVs in `work/` versus `data/figure_data/` and `results/tables/`;
- figure style and format, noting the archived SVG+PNG versus generated
  PDF+PNG discrepancy.

Do not replace any archived panel or result file during this comparison.
