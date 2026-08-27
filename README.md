# Credit Default Risk Prediction + Credit Scoring

**Part 1: Credit Default Prediction Model**
   - Predicting probability of serious delinquency within 2 years using 2 different models (Logistic Regression and XGBOOST), using the Kaggle ["Give Me Some           Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrowers, ~6.7% default rate).

**Part 2: Segmentation and Credit Scoring (Logistic Regression)**
   - Bucket borrowers into risk tiers (A–E) using the *already-trained* logistic regression pipeline, then validate, explain, and translate those tiers into            something a lender could actually act on.

**Part 3: Regulatory Capital (Basel IRB risk-weight function)**
   - Feed the PD from Part 1 into the Basel IRB risk-weight function to obtain a capital number, and use that capital number and RAROC formula to derive an             approve/decline cutoff.

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

**Part 1**
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

**Part 2**
1. Examined model's predicted probability distribution before bucketing, heavily right-skewed (median ≈2.7%, 99th percentile ≈6.7%), then split into 5 equal-count    percentile tiers via `pd.qcut`.
2. Validated the tiers against true outcomes: actual default rate increases monotonically A→E, and each tier's actual rate tracks its predicted mean probability      closely, confirming the tiers reflect real risk separation.
3. Explained tier placement using decomposition of the logit into additive feature contributions for each borrower, then used high-beta features to directly          compare between tiers A, C and E --> showed consistent trends. 
4. Built a points-based scorecard:
   - For Logistic Regression, `Score = Offset − Factor × logit` is used which preserves additivity across features (probability itself doesn't decompose                additively).
   - Used illustrative values (base score 600 at 50:1 good:bad odds, 20 points to double the odds) --> does not reflect real data.
   - Built the points table by binning each feature and averaging bin-level contributions, then caught and fixed a real defect: the zero-inflated late-payment          columns (~90% zero) caused `qcut`'s percentiles to collapse onto one bin --> fixed with fixed cutoffs (0/1/2/3+) instead of percentiles.
   - Found a second defect: rare extreme outliers (e.g. a borrower earning $1.56M/month) get diluted toward their bin's average, a tradeoff of any binned               scorecard, not something more bins can fix.
   - Added a review flag (reconstruction error > one PDO cycle) to manually isolate borrowers the table misrepresents instead of trusting blindly — 1.16% of            borrowers flagged, concentrated in tier E.

**Part 3**
 1. Defined asset correlations, risk-weight function and stated the PD and LGD input floors.
 2. Plotted the 3 curves corresponding to other retail, QRRE and mortgage.
 3. Derived the approve/decline cutoff by solving for break-even interest rate and taking maximum of the boxplots.
   
## Honest scope / limitations

**Part 1**:
   - This produces a **PD (probability of default) model only**. A real credit decision requires $EL = PD \times LGD \times EAD$, and this dataset has no loan          amount   or loss-given-default data — the threshold/cost analysis here demonstrates the methodology, not a deployable cutoff.

**Part 2**:
   - It scores and tiers borrowers only, the decision-making process is not illustrated here because it differs between lenders.

**Part 3**:
   - **The rows are borrowers, not loans.** IRB assigns capital per *loan*; a row here is a *borrower*. A borrower can requests for different loans at once -->         leading to different sub-classes, different correlations, different LGDs. The row-to-curve mapping is structurally undefined.
   - **The PD is not a Basel PD.** `SeriousDlqin2yrs` is a 90+ DPD flag over **2 years**; IRB consumes a **1-year long-run-average** default rate for at least          **5 years**. (CRE 36.81, 36.82)
   - **No LGD and EAD.** The LGDs below correspond to regulatory input floors; EAD is expressed per unit of credit limit, no actual currency.

## Setup

```
pip install -r requirements.txt
```

Download `cs-training.csv` and `cs-test.csv` from the [Kaggle competition page](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place them in `data/`.

Then open `credit_default_risk_prediction_+_credit_scoring_+_basel.ipynb`.
