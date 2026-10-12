import os
from pathlib import Path

SEED = 20260902
YEARS = list(range(1986, 2025))
YEAR_COLS = [f"Y{y}" for y in YEARS]

CROPS = {
    15: "Wheat",
    30: "Rice",
    56: "Maize",
    236: "Soybeans",
}
RICE_CODES = list(range(27, 33))
ELEMENTS = {5610: "Import", 5910: "Export"}

# ``ROOT`` is the code directory; inputs and generated artifacts are kept
# outside it so a clone can be relocated without editing source code.
ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
WORK_DIR = REPO_ROOT / "work"
_default_zip = REPO_ROOT / "data" / "raw" / "Trade_DetailedTradeMatrix_E_All_Data.zip"
ZIP_PATH = Path(os.environ.get("TRADE_ZIP", _default_zip))
OUT = WORK_DIR / "outputs_v2"
DATA_DIR = OUT / "data"
TABLE_DIR = OUT / "tables"
FIG_MAIN_DIR = OUT / "figures" / "main"
FIG_SUPP_DIR = OUT / "figures" / "supplement"
FIG_DATA_DIR = OUT / "figure_data"
LOG_DIR = OUT / "logs"
REPORT_DIR = WORK_DIR / "report_v2"

N_BOOT = 1000
BLOCK_LENGTH = 4
BLOCK_LENGTH_SENSITIVITY = (3, 5)

MAIN_MEMBER = "Trade_DetailedTradeMatrix_E_All_Data_NOFLAG.csv"
EXPECTED_MEMBERS = [
    "Trade_DetailedTradeMatrix_E_All_Data.csv",
    MAIN_MEMBER,
    "Trade_DetailedTradeMatrix_E_Elements.csv",
    "Trade_DetailedTradeMatrix_E_Flags.csv",
    "Trade_DetailedTradeMatrix_E_ItemCodes.csv",
    "Trade_DetailedTradeMatrix_E_PartnerCountries.csv",
    "Trade_DetailedTradeMatrix_E_ReporterCountries.csv",
]


def ensure_directories() -> None:
    for path in [
        DATA_DIR,
        TABLE_DIR,
        FIG_MAIN_DIR,
        FIG_SUPP_DIR,
        FIG_DATA_DIR,
        LOG_DIR,
        REPORT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
