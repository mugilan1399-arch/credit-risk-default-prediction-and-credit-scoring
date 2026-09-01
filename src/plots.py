"""Charts. Every function takes an axis or makes one, and returns it.

Kept apart from the modelling code so the notebook can stay the place where
things are looked at, and src/ stays the place where things are computed.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.basel import (
    capital_over_lgd,
    r_other_retail,
    r_qrre,
    r_residential_mortgage,
)
from src.config import LGD_UNSECURED, PD_FLOOR


def roc_comparison(y_true, proba_by_model, ax=None):
    """One ROC curve per model, with AUC in the legend."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    for label, proba in proba_by_model.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, proba):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    return ax


def precision_recall(f1_result, title, ax=None):
    """Precision-recall curve with the F1-optimal point marked.

    Takes the dict returned by evaluation.best_f1_threshold.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    precisions, recalls, _, f1, best = f1_result["curve"]
    ax.plot(recalls, precisions, label="Precision-Recall curve")
    ax.scatter(
        recalls[best],
        precisions[best],
        color="red",
        zorder=5,
        label=f"Best F1 = {f1[best]:.2f} (threshold={f1_result['threshold']:.2f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    return ax


def cost_curve(cost_result, fn_cost, fp_cost, ax=None):
    """Total misclassification cost against threshold, minimum marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    grid, costs, best = cost_result["grid"], cost_result["costs"], cost_result["best_index"]
    ax.plot(grid, costs)
    ax.scatter(
        grid[best],
        costs[best],
        color="red",
        zorder=5,
        label=f"Min cost at threshold={grid[best]:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel(f"Total cost (FN_cost={fn_cost}, FP_cost={fp_cost})")
    ax.set_title("Expected Cost vs. Threshold")
    ax.legend()
    return ax


def tier_calibration(summary, ax=None):
    """Realised default rate per tier as bars, mean predicted PD as a line.

    The gap between the two is the calibration story: bars far from the line
    mean the model ranks well but does not price well.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    tiers = summary.index.astype(str)
    ax.bar(tiers, summary["actual_default_rate"], label="Actual default rate")
    ax.plot(
        tiers,
        summary["predicted_mean_proba"],
        color="red",
        marker="o",
        label="Predicted mean probability",
    )
    ax.set_xlabel("Risk tier")
    ax.set_ylabel("Default rate")
    ax.set_title("Actual vs. predicted default rate by tier")
    ax.legend()
    return ax


def irb_curves(ax=None):
    """The three retail IRB curves, with the crossings and the PD floor marked."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    pd_grid = np.linspace(5e-4, 0.30, 600)
    other = capital_over_lgd(pd_grid, r_other_retail(pd_grid))
    qrre = capital_over_lgd(pd_grid, r_qrre(pd_grid))
    mortgage = capital_over_lgd(pd_grid, r_residential_mortgage(pd_grid))

    ax.plot(pd_grid, other, color="C0", lw=2.4, label="Other retail")
    ax.plot(pd_grid, qrre, color="C1", lw=2.0, label="QRRE, R = 0.04")
    ax.plot(pd_grid, mortgage, color="C2", lw=2.0, ls="--",
            label="Residential mortgage, R = 0.15")

    # other-retail R starts at 0.16 (above mortgage's 0.15) and decays to 0.03,
    # so it crosses both of the flat curves
    for diff, name in ((other - mortgage, "mortgage"), (other - qrre, "QRRE")):
        crossing = pd_grid[np.argmin(np.abs(diff))]
        y = capital_over_lgd(crossing, r_other_retail(crossing))
        ax.scatter(crossing, y, color="red", zorder=5, s=25)
        ax.annotate(
            f"other retail crosses\n{name} at PD $\\approx$ {crossing:.3%}",
            xy=(crossing, y),
            xytext=(crossing * 1.6, 0.30),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="grey"),
        )

    ax.axvline(PD_FLOOR, color="crimson", lw=1.0)
    ax.annotate("PD input floor\n(5bp)", xy=(PD_FLOOR, 0.42),
                xytext=(PD_FLOOR * 1.25, 0.42), color="crimson", fontsize=8)

    ax.set_xlabel("PD")
    ax.set_ylabel("K / LGD  (capital per unit of LGD)")
    ax.set_title("Retail IRB curves")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def apr_decision(apr_frame, thresholds, decline_rates, lgds=LGD_UNSECURED, ax=None):
    """Break-even APR per LGD assumption, with the decline fence annotated."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    labels = [f"LGD={lgd * 100}%" for lgd in lgds]
    ax.boxplot([apr_frame[c] for c in labels], tick_labels=labels)
    ax.set_ylim((0, 15))
    ax.set_ylabel("break-even APR (%)")

    y_top = ax.get_ylim()[1]
    label_clearance, arrow_len = 0.45, 2

    for i, col in enumerate(labels, start=1):
        thr = min(thresholds[col], y_top - 0.1)
        text_y = min(thr + 1.6, y_top - (label_clearance + arrow_len + 0.2))
        x_text = i - 0.05

        ax.hlines(thr, i - 0.35, i + 0.35, colors="crimson", linestyles="--",
                  linewidth=1.2, zorder=5)
        ax.annotate("", xy=(i - 0.255, thr), xytext=(x_text - 0.205, text_y),
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.2), zorder=5)
        ax.text(x_text, text_y, f"{decline_rates[col]:.1f}% rejected", ha="right",
                va="bottom", fontsize=9, color="crimson", fontweight="bold", zorder=6)

        tail = text_y + label_clearance
        ax.annotate("", xy=(x_text - 0.205, tail + arrow_len), xytext=(x_text - 0.205, tail),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.4), zorder=6)

    return ax
