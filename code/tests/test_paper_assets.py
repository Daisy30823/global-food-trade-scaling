from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_independent_paper_panel_inventory_is_complete():
    index = pd.read_csv(ROOT / "work" / "paper_figures" / "figure_index.csv")
    assert len(index) == 54
    assert (index["logical_figure"].str.startswith("Main").sum()) == 23
    assert (index["logical_figure"].str.startswith("Supplement").sum()) == 31
    for row in index.itertuples():
        folder = "main_panels" if row.logical_figure.startswith("Main") else "supplement_panels"
        for suffix in (".pdf", ".png"):
            path = ROOT / "work" / "paper_figures" / folder / f"{row.panel_stem}{suffix}"
            assert path.exists()
            assert path.stat().st_size > 1_000
