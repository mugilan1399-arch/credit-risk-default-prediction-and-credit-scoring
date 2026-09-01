"""Check that src/ reproduces the notebook's numbers.

Run from the repo root:  python verify_refactor.py

This is a reproduction check, not a test suite. It refits the pipelines and
compares the headline figures against what the notebook and README report, so a
refactor that quietly changed behaviour shows up immediately.
"""

import numpy as np
import pandas as pd

from src import basel, evaluation, scorecard
from src.cleaning import CreditDataCleaner, sentinel_rows
from src.config import FACTOR, FN_COST, FP_COST, LATE_PAYMENT_COLS, OFFSET
from src.data import features_and_target, load_raw, train_val_split
from src.pipelines import build_logreg_pipeline, build_xgb_pipeline

pd.set_option("display.width", 120)


def check(label, value, expected, tol):
    ok = abs(value - expected) <= tol
    print(f"  [{'ok' if ok else 'FAIL'}] {label}: {value:.4f} (expected ~{expected}, tol {tol})")
    return ok


def main():
    results = []

    print("\n1. Data and split")
    train, test, sample_entry = load_raw()
    X, y = features_and_target(train)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    print(f"  train {X_train.shape}, val {X_val.shape}")
    results.append(check("overall default rate", y.mean(), 0.0668, 0.002))

    print("\n2. Sentinel rule: does 'any' agree with 'all' on this data?")
    n_all = int(sentinel_rows(X_train).sum())
    n_any = int((X_train[LATE_PAYMENT_COLS] >= 96).any(axis=1).sum())
    print(f"  all-three: {n_all} rows | any-of-three: {n_any} rows")
    print(f"  [{'ok' if n_all == n_any else 'DIFFER'}] the two rules pick the same rows")
    results.append(n_all == n_any)

    print("\n3. Logistic pipeline")
    logreg = build_logreg_pipeline().fit(X_train, y_train)
    lr_proba = logreg.predict_proba(X_val)[:, 1]
    lr = evaluation.discrimination_summary(y_val, lr_proba)
    print(f"  AUC {lr['auc']:.4f} | Gini {lr['gini']:.4f} | KS {lr['ks']:.4f}")
    results.append(check("logreg holdout AUC", lr["auc"], 0.849, 0.02))

    print("\n4. XGBoost challenger")
    xgb = build_xgb_pipeline().fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    xg = evaluation.discrimination_summary(y_val, xgb_proba)
    print(f"  AUC {xg['auc']:.4f} | Gini {xg['gini']:.4f} | KS {xg['ks']:.4f}")
    results.append(check("xgboost holdout AUC", xg["auc"], 0.859, 0.02))
    print(f"  [{'ok' if xg['auc'] > lr['auc'] else 'FAIL'}] challenger out-discriminates the logistic")
    results.append(xg["auc"] > lr["auc"])

    print("\n5. Thresholds")
    f1 = evaluation.best_f1_threshold(y_val, xgb_proba)
    cost = evaluation.cost_optimal_threshold(y_val, xgb_proba)
    print(f"  F1-optimal   : {f1['threshold']:.4f} (P={f1['precision']:.3f}, R={f1['recall']:.3f})")
    print(f"  cost-optimal : {cost['threshold']:.3f} at {FN_COST}:{FP_COST}"
          f" | cost {cost['cost']:.0f} vs {cost['cost_at_half']:.0f} at 0.5")
    results.append(cost["cost"] <= cost["cost_at_half"])
    print(f"  [{'ok' if cost['cost'] <= cost['cost_at_half'] else 'FAIL'}] cost threshold beats 0.5")

    print("\n6. Scorecard scaling")
    print(f"  Factor {FACTOR:.4f} | Offset {OFFSET:.4f}")
    results.append(check("Factor = PDO/ln2", FACTOR, 28.8539, 1e-3))
    results.append(check("Offset = 600 - Factor*ln(50)", OFFSET, 487.1229, 1e-3))

    print("\n7. Tiers and points table")
    cleaner, imputer, scaler, model = (
        logreg.named_steps["cleaner"],
        logreg.named_steps["imputer"],
        logreg.named_steps["scaler"],
        logreg.named_steps["model"],
    )
    X_val_clean = cleaner.transform(X_val)
    X_val_imputed = pd.DataFrame(
        imputer.transform(X_val_clean), columns=X_val_clean.columns, index=X_val_clean.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_imputed), columns=X_val_imputed.columns, index=X_val_imputed.index
    )

    contributions, logit = scorecard.logit_contributions(X_val_scaled, model)
    rebuilt = 1 / (1 + np.exp(-logit))
    drift = float(np.abs(rebuilt.values - lr_proba).max())
    print(f"  max |reconstructed proba - predict_proba| = {drift:.2e}")
    results.append(check("logit decomposition is exact", drift, 0.0, 1e-9))

    res = pd.DataFrame({"val_proba": lr_proba, "y_val": y_val.values}, index=X_val.index)
    res["risk_tier"] = scorecard.assign_tiers(res["val_proba"])
    res["score"] = scorecard.score_from_logit(logit)

    summary = scorecard.tier_summary(res)
    print(summary.to_string())
    print(f"  [{'ok' if summary.attrs['monotonic'] else 'FAIL'}] default rate rises monotonically A->E")
    results.append(bool(summary.attrs["monotonic"]))

    table, rebuilt_score, base_points = scorecard.build_scorecard(
        X_val_imputed, contributions, model
    )
    error = scorecard.reconstruction_error(rebuilt_score, res["score"])
    flagged = scorecard.flag_for_review(error)
    share = flagged.mean()
    print(f"  points table rows: {len(table)} | base points {base_points:.1f}")
    print(f"  reconstruction error: mean {error.mean():.2f}, max {error.max():.1f}")
    print(f"  flagged for review: {int(flagged.sum())} of {len(res)} ({share:.2%})")
    results.append(check("share flagged for review", share, 0.0116, 0.006))

    print("\n8. Basel IRB")
    print(f"  R(mortgage) {float(basel.r_residential_mortgage(0.05)):.2f}"
          f" | R(QRRE) {float(basel.r_qrre(0.05)):.2f}"
          f" | R(other retail @ PD=5%) {float(basel.r_other_retail(0.05)):.4f}")
    results.append(check("mortgage correlation", float(basel.r_residential_mortgage(0.05)), 0.15, 1e-9))
    results.append(check("QRRE correlation", float(basel.r_qrre(0.05)), 0.04, 1e-9))

    lo, hi = float(basel.r_other_retail(1e-6)), float(basel.r_other_retail(0.99))
    print(f"  other-retail R decays {lo:.3f} -> {hi:.3f} as PD rises")
    results.append(check("other retail R at PD->0", lo, 0.16, 1e-3))

    k = float(basel.capital_over_lgd(0.05, basel.r_other_retail(0.05)))
    print(f"  K/LGD at PD=5% (other retail): {k:.4f}")
    results.append(k > 0)

    pd_1y = basel.pd_two_year_to_one_year(res["val_proba"].values)
    print(f"  PD 2y mean {res['val_proba'].mean():.4f} -> PD 1y mean {pd_1y.mean():.4f}")
    results.append(check("1y PD is below 2y PD", float((pd_1y < res['val_proba'].values).mean()), 1.0, 1e-9))

    print("\n9. Break-even pricing")
    raw_util = X_val_imputed["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
    util = basel.fill_zero_utilisation(raw_util)
    print(f"  zero-utilisation accounts refilled at the median: {int((raw_util == 0).sum())}")

    ead = basel.ead_per_limit(util)
    apr = basel.break_even_apr_by_lgd(basel.floored_pd(pd_1y), ead, util)
    fences = basel.tukey_upper_fence(apr)
    declines = basel.decline_rate(apr, fences)
    print(apr.describe().loc[["mean", "50%", "max"]].to_string())
    print("  decline rate per LGD:")
    for col in apr.columns:
        print(f"    {col}: fence {fences[col]:.2f}%, declined {declines[col]:.1f}%")

    # An APR of inf satisfies any range check, so test finiteness explicitly.
    n_bad = int((~np.isfinite(apr)).sum().sum())
    print(f"  [{'ok' if n_bad == 0 else 'FAIL'}] every break-even APR is finite ({n_bad} non-finite)")
    results.append(n_bad == 0)
    results.append(bool((declines > 0).all() and (declines < 50).all()))

    print("\n10. Saved model still loads through the pipeline_utils shim")
    try:
        import joblib

        saved = joblib.load("logreg_pipeline.joblib")
        saved_auc = evaluation.discrimination_summary(
            y_val, saved.predict_proba(X_val)[:, 1]
        )["auc"]
        print(f"  [ok] unpickled, holdout AUC {saved_auc:.4f}")
        results.append(True)
    except FileNotFoundError:
        print("  [skip] logreg_pipeline.joblib not present")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] {type(exc).__name__}: {exc}")
        results.append(False)

    passed, total = sum(bool(r) for r in results), len(results)
    print(f"\n{'=' * 60}\n{passed}/{total} checks passed\n{'=' * 60}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
