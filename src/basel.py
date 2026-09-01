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


def fill_zero_utilisation(utilisation, fill=None):
    """Replace zero utilisation with a stand-in value before pricing.

    Opex is charged per unit of limit but the APR is earned on the drawn
    balance, so a borrower drawing nothing has no revenue base and no finite
    break-even APR. Rather than drop those accounts, substitute a utilisation
    and price them as if they behaved like the median borrower.

    `fill` defaults to the median of the utilisation passed in, zeros included.
    Pass the training-fold median explicitly if you want the substitution to be
    a fitted quantity rather than one read off the set being scored.

    Returns the same type it was given, so a Series keeps its index.
    """
    values = np.clip(np.asarray(utilisation, dtype=float), 0, 1)
    fill = float(np.median(values)) if fill is None else float(fill)

    filled = np.where(values == 0, fill, values)

    if hasattr(utilisation, "index"):
        import pandas as pd

        return pd.Series(filled, index=utilisation.index, name=getattr(utilisation, "name", None))
    return filled


def ead_per_limit(utilisation, ccf=CCF):
    """EAD per unit of credit limit: U + CCF*(1 - U).

    With no balances or limits in the data, utilisation is the only handle on
    exposure. Future drawn/limit is assumed to match the observed ratio.
    """
    util = np.clip(np.asarray(utilisation, dtype=float), 0, 1)
    return util + ccf * (1 - util)


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


def tukey_upper_fence(frame):
    """Q3 + 1.5*IQR -- the same cut matplotlib's boxplot whiskers draw."""
    q1, q3 = frame.quantile(0.25), frame.quantile(0.75)
    return q3 + 1.5 * (q3 - q1)


def decline_rate(apr_frame, thresholds=None):
    """Share of borrowers priced above the fence, i.e. declined, per LGD column."""
    thresholds = tukey_upper_fence(apr_frame) if thresholds is None else thresholds
    return (apr_frame[apr_frame > thresholds].count() / len(apr_frame)) * 100


def floored_pd(pd_1y, floor=PD_FLOOR):
    """Apply the supervisory PD input floor (5bp).  CRE 32.58"""
    return np.maximum(np.asarray(pd_1y, dtype=float), floor)
