"""Basel IRB risk-weight function and the break-even pricing decision.

The risk-weight function is the ASRF model: a single systematic factor, a
supervisory asset correlation R, and a 99.9% one-year solvency standard. It
returns unexpected loss only -- the '- PD' term subtracts expected loss, which
is meant to be covered by provisions rather than capital.

Scope caveats carried over from the write-up, and still true:
  - a row here is a *borrower*, not a loan, while IRB assigns capital per exposure;
  - the target is 90+ DPD over two years, not IRB's one-year long-run average;
  - LGD uses regulatory floors, because the dataset carries no recovery data.
"""

import numpy as np
from scipy.stats import norm

from src.config import (
    CCF,
    CONFIDENCE,
    FUNDING,
    LGD_UNSECURED,
    MISSED_OUT_PR,
    OPEX,
    PD_FLOOR,
)

G = norm.ppf  # inverse standard normal
N = norm.cdf  # standard normal CDF


# --- supervisory asset correlations (CRE31) ---------------------------------


def r_residential_mortgage(pd_):
    """Residential mortgage exposures: flat R = 0.15.  CRE 31.14"""
    return np.full(np.shape(np.asarray(pd_, dtype=float)), 0.15)


def r_qrre(pd_):
    """Qualifying revolving retail: flat R = 0.04.  CRE 31.15"""
    return np.full(np.shape(np.asarray(pd_, dtype=float)), 0.04)


def r_other_retail(pd_):
    """All other retail: R declines from 0.16 to 0.03 as PD rises.  CRE 31.16"""
    pd_ = np.asarray(pd_, dtype=float)
    w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35.0))
    return 0.03 * w + 0.16 * (1 - w)


# --- the risk-weight function -----------------------------------------------


def capital_over_lgd(pd_, R):
    """UL-only capital requirement K per unit of LGD (dimensionless).

    Useful when LGD is unknown: the curves stay comparable without inventing a
    loss-given-default number the data cannot support.
    """
    pd_ = np.asarray(pd_, dtype=float)
    conditional_pd = N((G(pd_) + np.sqrt(R) * G(CONFIDENCE)) / np.sqrt(1 - R))
    return conditional_pd - pd_


def capital_K(pd_, lgd, R):
    """Capital requirement K per unit of EAD."""
    return lgd * capital_over_lgd(pd_, R)


def risk_weight(pd_, lgd, R):
    """Risk weight per unit of EAD = K x 12.5."""
    return 12.5 * capital_K(pd_, lgd, R)


# --- horizon and exposure conversions ---------------------------------------


def pd_two_year_to_one_year(pd_2y):
    """Convert a two-year default probability to a one-year one.

    Assumes a constant hazard across the two years, which gives
    PD_1y = 1 - sqrt(1 - PD_2y). It satisfies the properties we want -- roughly
    PD_2y/2 for small PD_2y, bounded in [0,1], monotonic, and below the
    45-degree line because longer exposure carries more risk.

    This is an assumption, not a calibration. It fixes the horizon mismatch but
    does NOT anchor the PD to a long-run average central tendency, which is what
    IRB actually requires. Treat the capital numbers as illustrative until a
    calibration step exists.
    """
    return 1 - np.sqrt(1 - np.asarray(pd_2y, dtype=float))


def _like(template, values):
    """Return `values` as a Series on `template`'s index when it has one."""
    if hasattr(template, "index"):
        import pandas as pd

        return pd.Series(values, index=template.index, name=getattr(template, "name", None))
    return values


def floor_utilisation(utilisation, floor=None):
    """Raise every utilisation below the floor up to it, before pricing.

    Opex is charged per unit of limit but the APR is earned on the drawn
    balance, so a thin revenue base sends the break-even APR to infinity at
    zero utilisation and into the thousands of percent just above it. Flooring
    prices those accounts as if they drew like the median borrower.

    `floor` defaults to the median of the utilisation passed in -- taken before
    any flooring, so it describes the observed population rather than the
    adjusted one. Pass the training-fold median explicitly if you want the
    floor to be a fitted quantity rather than one read off the set being scored.

    Note this is a floor, not a zero-fill: at the median it lifts half the book,
    not just the accounts drawing nothing.
    """
    values = np.maximum(np.asarray(utilisation, dtype=float), 0.0)
    floor = float(np.median(values)) if floor is None else float(floor)
    return _like(utilisation, np.maximum(values, floor))


def ccf_per_row(utilisation, base_ccf=CCF):
    """Credit conversion factor per account."""
    util = np.asarray(utilisation, dtype=float)

    ccf = np.full(util.shape, float(base_ccf))
    ccf[(util > 2 * np.percentile(util, 75) - np.percentile(util, 25)) & (util < 1)] = 1.0
    ccf[util == 1] = 0.0

    return _like(utilisation, ccf)


def ead_per_limit(utilisation, ccf=None):
    """EAD per unit of credit limit: U + CCF*(1 - U).

    With no balances or limits in the data, utilisation is the only handle on
    exposure. Future drawn/limit is assumed to match the observed ratio.

    `ccf` defaults to the per-account schedule in ccf_per_row. Utilisation is
    NOT clipped at 1 here: over-limit accounts are real, and clipping them away
    would discard exactly the rows the schedule exists to handle.
    """
    util = np.maximum(np.asarray(utilisation, dtype=float), 0.0)
    factor = ccf_per_row(util) if ccf is None else ccf
    return _like(utilisation, util + np.asarray(factor, dtype=float) * (1 - util))


# --- break-even pricing ------------------------------------------------------


def break_even_apr(pd_1y, lgd, ead, drawn, funding=FUNDING, opex=OPEX,
                   missed_out_pr=MISSED_OUT_PR, correlation_fn=r_other_retail):
    """Break-even APR that covers expected loss, funding, opex and capital."""
    pd_1y = np.asarray(pd_1y, dtype=float)
    k = capital_K(pd_1y, lgd, correlation_fn(pd_1y))
    numerator = (
        pd_1y * lgd * ead
        + funding * drawn
        + opex
        + (missed_out_pr - funding) * k * ead
    )
    return numerator / ((1 - pd_1y) * drawn)


def break_even_apr_by_lgd(pd_1y, ead, drawn, lgds=LGD_UNSECURED, **kwargs):
    """break_even_apr across several LGD assumptions, as percentages."""
    import pandas as pd

    out = pd.DataFrame()
    for lgd in lgds:
        out[f"LGD={lgd * 100}%"] = break_even_apr(pd_1y, lgd, ead, drawn, **kwargs) * 100
    return out


DECLINE_QUANTILE = 0.25


def tukey_upper_fence(frame):
    """Q3 + 1.5*IQR -- the same cut matplotlib's boxplot whiskers draw.

    No longer the decline rule; kept because the boxplot's y-limits are set
    from the whisker extents, which is the same calculation.
    """
    q1, q3 = frame.quantile(0.25), frame.quantile(0.75)
    return q3 + 1.5 * (q3 - q1)


def apr_threshold(frame, q=DECLINE_QUANTILE):
    """Decline anyone whose break-even APR exceeds the q-th percentile.

    The story is that the lender can only charge a competitive rate, so any
    account needing more than that to break even is turned away.

    Note what q=0.25 implies mechanically: the threshold is the lower quartile,
    so it lands exactly on the bottom edge of the box in the chart, and the
    decline rate is 75% by construction rather than by anything in the data.
    """
    return frame.quantile(q)


def decline_rate(apr_frame, thresholds=None):
    """Share of borrowers priced above the threshold, i.e. declined, per LGD."""
    thresholds = apr_threshold(apr_frame) if thresholds is None else thresholds
    return (apr_frame[apr_frame > thresholds].count() / len(apr_frame)) * 100


def floored_pd(pd_1y, floor=PD_FLOOR):
    """Apply the supervisory PD input floor (5bp).  CRE 32.58"""
    return np.maximum(np.asarray(pd_1y, dtype=float), floor)
