# Pre-release audit checklist

Status labels: **PASS** = inspected; **BLOCKED** = requires author action;
**NOT RUN** = not empirically reproduced in this environment.

| Check | Status | Evidence / required action |
|---|---|---|
| Original submission archive preserved | PASS | All new files are confined to `GitHub_Release/`; no original file was moved or edited. |
| Raw data excluded | PASS | The 428 MB ZIP is not copied; `.gitignore` excludes `data/raw/` and `*.zip`. |
| Source identity | PASS | SHA-256: `edf36ebefc487862077deb44d421a784facb647cf38a5002a0f98ad9cbbdd90e`; seven expected archive members inspected. |
| Portable main paths | PASS | Copied code uses `TRADE_ZIP`/`data/raw/` and writes generated artifacts to ignored `work/`. |
| Absolute personal path | PASS in release copy | A former author-specific absolute path in an added robustness wrapper was replaced only in the copied release script. |
| Static Python syntax | PASS | `python -m compileall -q code` passed under Python 3.12.14. |
| Dependencies available for full run | BLOCKED | The available Python 3.12 lacked SciPy; install exact pinned requirements in an isolated environment. |
| Full raw-data pipeline | NOT RUN | No v2/v2.1/bootstrap/null-model full execution was attempted. |
| Full test suite | NOT RUN | The default Python 3.13 had no `pytest`; the available Python 3.12 lacked SciPy. |
| Figure-only regeneration | PASS (rendering only) | The default local Python 3.13 (with plotting packages available, but without pytest) ran `python code/make_paper_figures.py` and produced a 54-row index, 54 PDFs, and 54 PNGs; 23 Main and 31 Supplement rows were present with no missing expected panel. This did not recalculate raw-data statistics or validate the pinned Python 3.12 environment. |
| Archived panel inventory | PASS | `figures_index_archived.csv`: 54 rows = 23 main + 31 supplementary; copied files: 46 main and 62 supplementary (SVG+PNG pairs). |
| Format consistency | BLOCKED | Current archival figures are SVG+PNG, while `paper_figures.py` renders PDF+PNG. Confirm the final authoritative figure package and update the documentation/index only after review. |
| Legacy code | PASS / review needed | Three unreferenced modules moved only in the release copy to `code/legacy/`; retain for audit, exclude from documented execution. |
| Manuscript and unpublished materials | PASS | No main-text DOCX, supplementary draft/report, or Word composite figures were copied. |
| Personal data / credentials | PASS (text scan) | No token/API-key pattern found in release code. Re-scan before publishing any future additions. |
| FAOSTAT reuse | BLOCKED | The FAOSTAT catalog labels the dataset CC BY 4.0, but FAO terms note potential third-party exceptions. Verify the exact download metadata and give required attribution. |
| Paper citations | BLOCKED | The project note says the current main text lacks a formal bibliography/in-text citations; correct this independently before submission. |
| Repository licence | BLOCKED | No code licence was selected because authorship, co-author consent, and third-party terms have not been confirmed. Do not publish until a licence decision is made. |
| Result/figure disclosure | BLOCKED | `data/figure_data/`, `results/`, and added robustness figures are local candidate materials. Confirm journal embargo/publisher/co-author approval before pushing. |
| Citation metadata | BLOCKED | Replace README placeholders after a manuscript title, author list, venue/preprint decision, and DOI (if any) are author-confirmed. |

## Required author decisions before GitHub or Zenodo

1. Confirm the final manuscript citation and whether any preprint/data embargo
   applies.
2. Confirm which result tables and all 54 panels may be public before
   submission or acceptance.
3. Verify the FAOSTAT download version, attribution, and any third-party
   limitations for the exact archive used.
4. Choose an explicit licence for code and, if appropriate, a separate licence
   for derived data/figures.
5. Install the pinned environment, run the full pipeline once, retain the log
   and manifests, then compare all reported estimates and panels without
   overwriting this archive.
