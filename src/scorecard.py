"""Risk tiers and the points scorecard.

Score = Offset + Factor * log(odds), with Factor = PDO / ln(2)
(Siddiqi, Intelligent Credit Scoring, 2nd ed.)

The scorecard is built by decomposing the fitted logit into per-feature
contributions, then averaging those contributions within bins. Binning loses
information, so every borrower's binned score is compared against their exact
continuous score and the large misses are flagged for manual review.
"""

import numpy as np
import pandas as pd

from src.config import (
    CUSTOM_BIN_EDGES,
    CUSTOM_BIN_LABELS,
    FACTOR,
    N_BINS,
    NEEDS_REVIEW_THRESHOLD,
    OFFSET,
)

TIER_LABELS = ["A", "B", "C", "D", "E"]


def logit_contributions(X_scaled, model):
    """Per-feature contribution to the logit, plus the logit itself.

    contribution_i = scaled_feature_i * coef_i, and the row sum plus the
    intercept reconstructs the logit the model actually used.
    """
    coefficients = pd.Series(model.coef_[0], index=X_scaled.columns)
    contributions = X_scaled * coefficients
    logit = contributions.sum(axis=1) + model.intercept_[0]
    return contributions, logit


def score_from_logit(logit):
    """Map the logit to points. logit = -log(odds of good), hence the minus."""
    return OFFSET - FACTOR * logit


def assign_tiers(proba, labels=TIER_LABELS):
    """Equal-count quintiles, A (safest) to E (riskiest).

    NOTE: equal-count bucketing is a convenience, not a rating grade structure.
    Grades are supposed to be monotonic in *observed* default rate, statistically
    distinguishable (non-overlapping confidence intervals), and constrained
    against concentration. tier_summary below tests the first of those three.
    """
    return pd.qcut(proba, len(labels), labels=labels)


def tier_summary(results, tier_col="risk_tier", proba_col="val_proba", y_col="y_val"):
    """Borrower counts, realised default rate and mean predicted PD per tier."""
    summary = results.groupby(tier_col, observed=True).agg(
        n_borrowers=(y_col, "size"),
        n_defaults=(y_col, "sum"),
        actual_default_rate=(y_col, "mean"),
        predicted_mean_proba=(proba_col, "mean"),
    )
    summary.attrs["monotonic"] = summary["actual_default_rate"].is_monotonic_increasing
    return summary


def bin_feature(series, column, n_bins=N_BINS, use_custom_bins=True):
    """Equal-frequency bins, except for the zero-inflated count columns.

    Set use_custom_bins=False for the naive pass, where qcut is applied to every
    column -- including the late-payment counts, which it collapses to a single
    bin. That pass is worth keeping in the write-up because the failure is the
    reason the custom edges exist.
    """
    if use_custom_bins and column in CUSTOM_BIN_EDGES:
        return pd.cut(series, bins=CUSTOM_BIN_EDGES[column], labels=CUSTOM_BIN_LABELS)
    return pd.qcut(series, q=n_bins, duplicates="drop")


def build_scorecard(X_imputed, contributions, model, columns=None, n_bins=N_BINS,
                    use_custom_bins=True):
    """Build the points table and the per-borrower reconstructed score.

    Returns (scorecard_df, reconstructed_score, base_points).
    """
    columns = list(X_imputed.columns) if columns is None else list(columns)
    base_points = OFFSET - FACTOR * model.intercept_[0]

    binned_features = {}
    bin_points_by_feature = {}
    rows = []

    for col in columns:
        binned = bin_feature(
            X_imputed[col], col, n_bins=n_bins, use_custom_bins=use_custom_bins
        )
        bin_points = (-FACTOR * contributions[col]).groupby(binned, observed=True).mean()

        binned_features[col] = binned
        bin_points_by_feature[col] = bin_points

        counts = binned.value_counts()
        for interval, points in bin_points.items():
            rows.append(
                {
                    "feature": col,
                    "bin": str(interval),
                    "points": round(points, 1),
                    "n_borrowers": int(counts[interval]),
                }
            )

    reconstructed_points = pd.DataFrame(
        {col: binned_features[col].map(bin_points_by_feature[col]) for col in columns},
        index=X_imputed.index,
    )
    reconstructed_score = base_points + reconstructed_points.sum(axis=1)

    scorecard_df = pd.DataFrame(rows)
    scorecard_df.attrs["reconstructed_points"] = reconstructed_points
    return scorecard_df, reconstructed_score, base_points


def reconstruction_error(reconstructed_score, exact_score):
    """Absolute gap between the binned table score and the continuous score."""
    return (reconstructed_score - exact_score).abs()


def flag_for_review(error, threshold=NEEDS_REVIEW_THRESHOLD):
    """Borrowers the table cannot price to within one doubling of the odds."""
    return error > threshold
