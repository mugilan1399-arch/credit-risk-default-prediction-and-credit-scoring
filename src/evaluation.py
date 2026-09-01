"""Discrimination, threshold selection and cross-validation."""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import FN_COST, FP_COST, N_SPLITS, RANDOM_STATE


def gini(y_true, y_proba):
    """Gini = 2*AUC - 1. Banks quote this, not AUC."""
    return 2 * roc_auc_score(y_true, y_proba) - 1


def ks_statistic(y_true, y_proba):
    """Kolmogorov-Smirnov: the widest gap between the good and bad CDFs."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def discrimination_summary(y_true, y_proba):
    """AUC, Gini and KS in one dict, for a results table."""
    return {
        "auc": roc_auc_score(y_true, y_proba),
        "gini": gini(y_true, y_proba),
        "ks": ks_statistic(y_true, y_proba),
    }


def best_f1_threshold(y_true, y_proba):
    """Threshold maximising unweighted F1.

    Treats a missed defaulter and a declined good customer as equally costly,
    which they are not -- see cost_optimal_threshold for the version that prices
    the two errors differently.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    f1 = f1[:-1]  # the last precision/recall point has no matching threshold
    best = int(np.argmax(f1))
    return {
        "threshold": thresholds[best],
        "precision": precisions[best],
        "recall": recalls[best],
        "f1": f1[best],
        "curve": (precisions, recalls, thresholds, f1, best),
    }


def cost_optimal_threshold(y_true, y_proba, fn_cost=FN_COST, fp_cost=FP_COST, grid=None):
    """Sweep thresholds and pick the one minimising fn_cost*FN + fp_cost*FP.

    The 5:1 ratio is an assumption, not a measurement -- vary it and report the
    sensitivity rather than presenting one number as if it were derived.
    """
    grid = np.linspace(0.01, 0.99, 99) if grid is None else np.asarray(grid)

    costs = []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(y_true, (y_proba >= t).astype(int)).ravel()
        costs.append(fn_cost * fn + fp_cost * fp)
    costs = np.array(costs)

    best = int(np.argmin(costs))
    tn, fp, fn, tp = confusion_matrix(y_true, (y_proba >= 0.5).astype(int)).ravel()

    return {
        "threshold": grid[best],
        "cost": costs[best],
        "cost_at_half": fn_cost * fn + fp_cost * fp,
        "grid": grid,
        "costs": costs,
        "best_index": best,
    }


def cv_auc(pipeline, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Stratified k-fold AUC. Returns the fold scores plus mean and std."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}
