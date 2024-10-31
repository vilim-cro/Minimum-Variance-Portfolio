"""
Class representing a financial asset class. Contains price and change series of the asset class.
Calculates mean return, variance and standard deviation of the asset class.
"""
# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

from enum import Enum
from dataclasses import dataclass, field

import pandas as pd

class AssetClassType(Enum):
    COMMODITIES = 'Commodities'
    EQUITY = 'Equities'
    BONDS = 'Bonds'

class AssetNames(Enum):
    OIL = 'Oil'
    GOLD = 'Gold'
    CORN = 'Corn'
    CHINA_EQ = 'China eq.'
    US_EQ = 'US eq.'
    EUROPE_EQ = 'Europe eq.'
    JAPAN_EQ = 'Japan eq.'
    US_10_YEAR_BOND = 'US bonds'

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.ffill()
    data['Date'] = pd.to_datetime(data.index)
    data = data.set_index('Date')
    data['Price'] = data['Price'].replace(',', '', regex=True).astype(float)
    return data

def create_asset_class(path: str, *args, **kwargs):
    data = pd.read_csv(path, index_col=0)
    data = clean_data(data)
    return AssetClass(data, *args, **kwargs)

@dataclass
class AssetClass:
    data: pd.DataFrame
    name: str
    asset_type: AssetClassType

    prices_series: pd.Series = field(init=False)
    change_series: pd.Series = field(init=False)

    def __post_init__(self):
        self.prices_series = self.data['Price'].iloc[::-1]
        self.change_series = self.prices_series.pct_change()

    @property
    def variance(self):
        return self.change_series.var()

    @property
    def std_dev(self):
        return self.change_series.std()


def rank_assets_by_std_dev(assets: list[AssetClass]) -> pd.Series:
    return pd.Series({asset.name: asset.std_dev for asset in assets}).sort_values()
