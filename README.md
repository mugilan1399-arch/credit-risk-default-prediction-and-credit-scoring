# Credit Default Risk Prediction

Predicting probability of serious delinquency within 2 years, using the Kaggle ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrowers, ~6.7% default rate).

## Results

| Model | AUC |
|---|---|
| Logistic regression (raw) | 0.714 |
| Logistic regression (capped utilization outlier) | 0.811 |
| Logistic regression (+ DebtRatio/income-missing fix) | 0.812 |
| Logistic regression (+ late-payment sentinel-value fix) | **0.856** |
| XGBoost (raw, default hyperparameters) | **0.860** |

## What this project actually shows

The headline result isn't "XGBoost beats logistic regression" — it's that **most of that gap was fixable data quality issues, not an inherent limitation of the linear model**:

- `RevolvingUtilizationOfUnsecuredLines` had extreme outliers (max 50,708 on a ratio that should top out near 1) distorting `StandardScaler`'s mean/std. Capping at the train-derived 99th percentile alone closed most of the gap to XGBoost.
- `DebtRatio`'s extreme values (max 329,664) turned out to correlate 92.6% with missing `MonthlyIncome` — strong evidence the raw debt amount was stored in place of a ratio whenever income was unreported.
- The three late-payment count columns showed suspicious 0.98–0.99 pairwise correlation, traced to 214 rows sharing an identical fabricated sentinel value (98, or 96) across all three columns simultaneously. This artificial correlation caused a multicollinearity-driven sign flip (a negative coefficient on `NumberOfTime60-89DaysPastDueNotWorse`, which should intuitively be risk-*increasing*). Fixing it resolved the sign flip and closed nearly all of the remaining AUC gap.

## Methodology

1. Explored missingness and class imbalance; ruled out accuracy as an evaluation metric given the ~93/7 class split.
2. Stratified train/validation split from `cs-training.csv` (Kaggle's `cs-test.csv` has no usable labels for this competition).
3. Built a `SimpleImputer` → `StandardScaler` → `LogisticRegression` baseline, and an `XGBClassifier` baseline.
4. Diagnosed and root-cause-fixed the three data quality issues above.
5. Wrapped the full cleaning → impute → scale → model sequence into a single `sklearn.Pipeline` with a custom `CreditDataCleaner` transformer.
6. Explored F1-maximizing and cost-based threshold selection (the latter explicitly weighting a missed default more heavily than a false alarm — since the dataset has no loan amount/LGD data, this is illustrative of the *methodology* real credit decisioning uses, not a deployable dollar-optimal cutoff).
7. Cross-validated the final pipeline (`StratifiedKFold`, 5 folds) and ran randomized hyperparameter search on XGBoost.

## Honest scope / limitations

This produces a **PD (probability of default) model only**. A real credit decision requires $EL = PD \times LGD \times EAD$, and this dataset has no loan amount or loss-given-default data — the threshold/cost analysis here demonstrates the methodology, not a deployable cutoff.

## Setup

```
pip install -r requirements.txt
```

Download `cs-training.csv` and `cs-test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them in `data/`.

Then open `credit_risk_model.ipynb`.
