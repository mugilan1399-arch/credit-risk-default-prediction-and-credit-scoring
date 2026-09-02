"""The two model pipelines.

Both wrap cleaning, imputation and (for the logistic) scaling inside a single
sklearn Pipeline, so every statistic is learned on the training fold only. This
is what makes the cross-validated AUCs trustworthy rather than optimistic.
"""

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.cleaning import CreditDataCleaner
from src.config import RANDOM_STATE


def build_logreg_pipeline(random_state=RANDOM_STATE, **cleaner_kwargs):
    """Production candidate: interpretable, and the one the scorecard is built from."""
    return Pipeline(
        [
            ("cleaner", CreditDataCleaner(**cleaner_kwargs)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=random_state)),
        ]
    )


def build_xgb_pipeline(random_state=RANDOM_STATE, **cleaner_kwargs):
    """Challenger: establishes the achievable discrimination ceiling.

    No scaler -- trees are invariant to monotone rescaling of the features.
    """
    return Pipeline(
        [
            ("cleaner", CreditDataCleaner(**cleaner_kwargs)),
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(random_state=random_state)),
        ]
    )


# Search space for the tuned challenger (RandomizedSearchCV over the xgb pipeline).
XGB_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": ("randint", 100, 500),
    "model__max_depth": ("randint", 3, 8),
    "model__learning_rate": ("uniform", 0.01, 0.29),
    "model__subsample": ("uniform", 0.6, 0.4),
    "model__colsample_bytree": ("uniform", 0.6, 0.4),
    "model__min_child_weight": ("randint", 1, 8),
}


def xgb_param_distributions():
    """Materialise XGB_PARAM_DISTRIBUTIONS into live scipy.stats objects."""
    from scipy.stats import randint, uniform

    makers = {"randint": randint, "uniform": uniform}
    return {
        key: makers[kind](a, b) for key, (kind, a, b) in XGB_PARAM_DISTRIBUTIONS.items()
    }
