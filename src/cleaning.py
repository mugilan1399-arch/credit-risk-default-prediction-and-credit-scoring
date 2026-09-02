"""The cleaning transformer, kept inside a Pipeline so nothing leaks from val to train.

Three fixes, all established in Part 1 of the notebook:

1. Revolving utilisation is capped at the 99th percentile *of the training fold*.
2. Where MonthlyIncome is missing, DebtRatio holds a payment amount rather than a
   ratio, so it is divided by the training median income to put it back on scale.
3. Rows carrying the 96/98 sentinel codes in the late-payment columns are set to
   NaN so the downstream imputer fills them with the median.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import LATE_PAYMENT_COLS, SENTINEL_VALUE, UTILISATION_CAP_QUANTILE


class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """Learns its cap and median on fit, applies them on transform.

    Parameters
    ----------
    sentinel_rule : {"any", "all"}
        Whether a row is treated as sentinel-coded when *any* late-payment column
        is >= 96, or only when *all three* are.

        NOTE: the notebook investigation (cells 41/43) used "all", on the finding
        that the 96/98 codes always occur together on the same 214 rows. The
        transformer that produced the committed results used "any". On this
        dataset the two agree, so the saved model is unaffected -- but "all" is
        the stricter reading and the one the write-up argues for. Default is kept
        at "any" to reproduce the existing numbers exactly; pass "all" to switch.
    """

    def __init__(self, sentinel_rule="any"):
        self.sentinel_rule = sentinel_rule

    def fit(self, X, y=None):
        self.utilization_cap_ = X["RevolvingUtilizationOfUnsecuredLines"].quantile(
            UTILISATION_CAP_QUANTILE
        )
        self.income_median_ = X["MonthlyIncome"].median()
        return self

    def transform(self, X):
        X = X.copy()

        X["RevolvingUtilizationOfUnsecuredLines"] = X[
            "RevolvingUtilizationOfUnsecuredLines"
        ].clip(upper=self.utilization_cap_)

        income_missing = X["MonthlyIncome"].isnull()
        X.loc[income_missing, "DebtRatio"] = (
            X.loc[income_missing, "DebtRatio"] / self.income_median_
        )

        # getattr, not self.sentinel_rule: logreg_pipeline.joblib was pickled
        # before this parameter existed, so old instances have no such attribute.
        rule = getattr(self, "sentinel_rule", "any")

        flags = X[LATE_PAYMENT_COLS] >= SENTINEL_VALUE
        sentinel_mask = flags.all(axis=1) if rule == "all" else flags.any(axis=1)
        X.loc[sentinel_mask, LATE_PAYMENT_COLS] = np.nan

        return X


def sentinel_rows(X):
    """Rows where all three late-payment columns carry a sentinel code."""
    return (X[LATE_PAYMENT_COLS] >= SENTINEL_VALUE).all(axis=1)
