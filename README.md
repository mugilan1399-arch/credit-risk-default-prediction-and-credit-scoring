# Credit Default Risk Prediction

Predicting probability of serious delinquency within 2 years using 2 different models (Logistic Regression and XGBOOST), using the Kaggle ["Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrowers, ~6.7% default rate).

## Results

| Model | AUC |
|---|---|
| Logistic regression (DebtRatio/income and capped utilization fix) | 0.812 |
| Logistic regression (+ late-payment sentinel fix) | **0.854** |
| Logistic regression (all fixes + Cross-Validation) | **0.849** |
| XGBoost (after same fixes) | **0.859** |
| XGBoost (+ Cross-Validation) | **0.859** |
| XGBoost (+ Hyperparamter tuning) | **0.865** |

## Methodology

1. Explored missingness (`MonthlyIncome`: 19.8%, `NumberOfDependents`: 2.6%) and confirmed severe class imbalance, ruling out accuracy as an evaluation metric.
2. Split off a stratified validation set from `cs-training.csv` (Kaggle's `cs-test.csv` has no usable labels — confirmed early on).
3. Realised that:
   - `RevolvingUtilizationOfUnsecuredLines` had extreme outliers (max 50,708) --> Cap at the train-derived 99th percentile.
   - `DebtRatio`'s extreme values (max 329,664) were shown to correlate almost perfectly (92.6%) with missing `MonthlyIncome` — strong evidence the raw debt            amount was stored in place of a ratio when income was unreported. Fixing this added a `MonthlyIncome_was_missing` flag and recovered a per-row ratio estimate.
4. Built a `SimpleImputer` (median) → `StandardScaler` → `LogisticRegression` --> obtained **AUC = 0.812**.
5. Built an `XGBClassifier` baseline (median-imputed only, no scaling needed): **AUC = 0.859** — slightly over logistic regression, consistent with XGBoost's         ability to capture non-linear feature interactions.
5. Investigated *why* there is a gap, rather than accepting it:
   - The three late-payment count columns showed suspiciously high pairwise correlation (0.98–0.99), traced to 214 rows sharing an identical sentinel value (98 or      96) across all three columns — a data-encoding artifact, not real payment history. Flagging and nulling these rows fixed both a counterintuitive negative          coefficient (multicollinearity-driven sign flip) and pushed logistic regression to **AUC = 0.854** — nearly matching XGBoost's 0.859 through data cleaning         alone, no hyperparameter tuning.
6. Wrapped the full cleaning → impute → scale → model sequence into a single `sklearn.Pipeline` with a custom `CreditDataCleaner` transformer, enforcing the fit-     on-train-only discipline structurally rather than manually.
7. Explored threshold selection beyond the default 0.5: F1-maximizing threshold (found by direct search over `precision_recall_curve`, since precision/recall are     step functions of a finite sample — not differentiable), and a cost-based threshold that explicitly weights missing a defaulter more heavily than a false alarm    (assumed 5:1 ratio, since the dataset lacks loan amount/LGD data needed for a real cost figure).
8. Used Cross-Validation to find average AUCs for XGB and L.R
9. Did Hyperparamter tuning on xgb to improve AUC score

## Honest scope / limitations

This produces a **PD (probability of default) model only**. A real credit decision requires $EL = PD \times LGD \times EAD$, and this dataset has no loan amount or loss-given-default data — the threshold/cost analysis here demonstrates the methodology, not a deployable cutoff.

## Setup

```
pip install -r requirements.txt
```

Download `cs-training.csv` and `cs-test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them in `data/`.

Then open `credit_risk_model.ipynb`.
