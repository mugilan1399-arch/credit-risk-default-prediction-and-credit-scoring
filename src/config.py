"""Column names, model constants and stated assumptions.

Everything here was previously scattered across notebook cells as bare
module-level names. Nothing is derived from the data; these are choices.
"""

from pathlib import Path

import numpy as np

# --- paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# --- columns ----------------------------------------------------------------

TARGET = "SeriousDlqin2yrs"

LATE_PAYMENT_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

SENTINEL_VALUE = 96  # 96 and 98 are placeholder codes, not real late-payment counts

# --- split / seeds ----------------------------------------------------------

RANDOM_STATE = 42
VAL_SIZE = 0.2
N_SPLITS = 5

# --- cleaning ---------------------------------------------------------------

UTILISATION_CAP_QUANTILE = 0.99

# --- threshold selection ----------------------------------------------------

FN_COST = 5  # relative cost of missing an actual defaulter (lost exposure on a bad loan)
FP_COST = 1  # relative cost of declining a good customer (lost marginal profit)

# --- scorecard scaling (Siddiqi, Intelligent Credit Scoring) -----------------

BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20  # points to double the odds

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

N_BINS = 5

# Zero-inflated count columns: qcut collapses these to a single bin because every
# percentile checkpoint lands on the same repeated value, so use fixed cutoffs.
CUSTOM_BIN_EDGES = {col: [-np.inf, 0, 1, 2, np.inf] for col in LATE_PAYMENT_COLS}
CUSTOM_BIN_LABELS = ["0", "1", "2", "3+"]

# A borrower whose binned score misses the continuous score by more than one full
# doubling of the odds is sent for individual review.
NEEDS_REVIEW_THRESHOLD = PDO

# --- Basel IRB (CRE31 / CRE32) ----------------------------------------------

CONFIDENCE = 0.999  # supervisory solvency standard: 99.9% over one year

PD_FLOOR = 0.0005  # CRE 32.58
PD_FLOOR_QRRE_REVOLVER = 0.0010  # CRE 32.58
LGD_FLOOR = {  # CRE 32.58
    "residential_mortgage": 0.05,
    "qrre": 0.50,
    "unsecured_other_retail": 0.30,
}

# --- pricing assumptions (none of these come from the dataset) --------------

FUNDING = 0.04  # average annual rate paid on borrowed funds
OPEX = 0.03  # operational expenses
MISSED_OUT_PR = 0.12  # expected annual missed-out profit rate
CCF = 0.50  # credit conversion factor on the undrawn portion of a revolving line
LGD_UNSECURED = [0.3, 0.45, 0.6]  # floor for unsecured other retail is 0.30
