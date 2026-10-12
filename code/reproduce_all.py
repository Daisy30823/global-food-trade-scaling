from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def run(script: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / script), *args]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def run_tests() -> None:
    audit = REPO_ROOT / "work" / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        f"--junitxml={audit / 'final_pytest.xml'}",
    ]
    print("Running:", " ".join(map(str, command)))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    archive = Path(os.environ.get("TRADE_ZIP", REPO_ROOT / "data" / "raw" / "Trade_DetailedTradeMatrix_E_All_Data.zip"))
    if not archive.exists():
        raise FileNotFoundError(
            "Set TRADE_ZIP to the downloaded FAOSTAT archive, or place it at "
            "data/raw/Trade_DetailedTradeMatrix_E_All_Data.zip."
        )
    run("run_all.py")
    run("run_all_v2_1.py")
    run("make_paper_figures.py", "--refresh-data")
    run_tests()
    print("Complete. Newly generated files are under work/ (which is git-ignored).")


if __name__ == "__main__":
    main()
