"""
Class representing a financial portfolio for a given period.
Contains weights and returns of the portfolio.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from periods import Period

@dataclass
class Portfolio:
    period: Period
    weights: list[float]

    change_series: pd.Series = field(init=False)

    def __post_init__(self):
        self.change_series = pd.Series(self.period.change_series_matrix @ self.weights,
                                       index=self.period.change_series_matrix.index)

    @property
    def variance(self) -> float:
        return self.change_series.var()

    @property
    def std_dev(self) -> float:
        return self.change_series.std()

    def daily_changes(self, weights = None) -> pd.Series:
        if weights is None:
            weights = self.weights
        return pd.Series(self.period.change_series_matrix @ weights,
                         index=self.period.change_series_matrix.index)

    def total_change(self, weights = None) -> float:
        if weights is None:
            weights = self.weights
        return self.period.total_changes @ weights


def get_min_var_weights(cov_matrix_inv: pd.DataFrame) -> list[float]:
    u = np.array([1 for _ in range(len(cov_matrix_inv))])
    return (u @ cov_matrix_inv) / (u @ cov_matrix_inv @ u.T)
