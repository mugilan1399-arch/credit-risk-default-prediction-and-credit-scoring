"""Backwards-compatibility shim.

CreditDataCleaner now lives in src/cleaning.py. This module stays because
logreg_pipeline.joblib was pickled against `pipeline_utils.CreditDataCleaner`,
and joblib resolves that path on load. Deleting this file would break the saved
model. New code should import from src.cleaning directly.
"""

from src.cleaning import CreditDataCleaner  # noqa: F401
from src.config import LATE_PAYMENT_COLS  # noqa: F401

__all__ = ["CreditDataCleaner", "LATE_PAYMENT_COLS"]
