"""
A class representing a time sequence of financial portfolios.
"""

from dataclasses import dataclass

from portfolio import Portfolio

@dataclass
class PortfolioSeries:
    portfolios: list[Portfolio]

    @property
    def variance(self) -> list[float]:
        return [portfolio.variance for portfolio in self.portfolios]

    @property
    def std_dev(self) -> list[float]:
        return [portfolio.std_dev for portfolio in self.portfolios]

    @property
    def prev_portfolios(self):
        return [Portfolio(p2.period, p1.weights) for p1, p2 in zip(
            self.portfolios[:-1], self.portfolios[1:])]
