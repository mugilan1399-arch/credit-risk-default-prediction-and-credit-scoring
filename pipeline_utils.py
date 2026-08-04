import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

LATE_PAYMENT_COLS = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate']


class CreditDataCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.utilization_cap_ = X['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
        self.income_median_ = X['MonthlyIncome'].median()
        return self

    def transform(self, X):
        X = X.copy()

        X['RevolvingUtilizationOfUnsecuredLines'] = X['RevolvingUtilizationOfUnsecuredLines'].clip(upper=self.utilization_cap_)

        income_missing = X['MonthlyIncome'].isnull()
        X.loc[income_missing, 'DebtRatio'] = X.loc[income_missing, 'DebtRatio'] / self.income_median_

        sentinel_mask = (X[LATE_PAYMENT_COLS] >= 96).any(axis=1)
        X.loc[sentinel_mask, LATE_PAYMENT_COLS] = np.nan

        return X
