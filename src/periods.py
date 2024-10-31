"""
Module representing the periods of data series.
"""

from datetime import date
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

from asset_class import AssetClass, AssetNames

def divide_in_periods(assets: list[AssetClass]):
    price_series_dict = {asset.name: asset.prices_series for asset in assets}

    prices_df = pd.DataFrame(price_series_dict).ffill()

    prices_data = [group for _, group in prices_df.groupby(prices_df.index.year)]
    return [Period(prices) for prices in prices_data]


@dataclass
class Period:
    price_series_matrix: pd.DataFrame

    @cached_property
    def change_series_matrix(self) -> pd.DataFrame:
        return self.price_series_matrix.pct_change(fill_method=None)[1:]

    @property
    def start_date(self) -> date:
        return self.price_series_matrix.index[0].date()

    @property
    def end_date(self) -> date:
        return self.price_series_matrix.index[-1].date()

    @property
    def year(self) -> int:
        return self.start_date.year

    @property
    def total_changes(self) -> list[float]:
        return [self.asset_total_change(asset) for asset in AssetNames]

    @property
    def cov_matrix(self) -> pd.DataFrame:
        return self.change_series_matrix.cov()

    @property
    def cov_matrix_inv(self) -> np.ndarray:
        return np.linalg.inv(self.cov_matrix)

    @property
    def corr_matrix(self) -> pd.DataFrame:
        return self.change_series_matrix.corr()

    def asset_changes_series(self, asset_name: str) -> pd.Series:
        return self.change_series_matrix[asset_name]

    def asset_total_change(self, asset_name: str) -> float:
        return self.price_series_matrix[asset_name].iloc[-1]\
            / self.price_series_matrix[asset_name].iloc[0] - 1

    def asset_var(self, asset_name: str) -> float:
        return self.asset_changes_series(asset_name).var()

    def asset_std_dev(self, asset_name: str) -> float:
        return self.asset_changes_series(asset_name).std()

    def __str__(self) -> str:
        return str(self.year)


def std_dev_by_year(asset_name: str, periods: list[Period]):
    data = {str(period): period.asset_std_dev(asset_name) for period in periods}
    return pd.Series(data)

def var_by_year(asset_name: str, periods: list[Period]):
    data = {str(period): period.asset_var(asset_name) for period in periods}
    return pd.Series(data)

def change_by_year(asset_name: str, periods: list[Period]):
    data = {str(period): period.asset_total_change(asset_name) for period in periods}
    return pd.Series(data)
