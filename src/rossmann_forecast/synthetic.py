"""Small synthetic Rossmann-shaped data for tests and CI. Never used for reported results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel, build_panel


def make_synthetic(
    n_stores: int = 12, n_days: int = 700, seed: int = 0, promo_effect: float = 0.25
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (train, store) frames with the Kaggle column layout."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-01", periods=n_days, freq="D")
    base = rng.uniform(3000, 9000, n_stores)
    dow_factor = np.array([1.1, 1.0, 0.95, 1.0, 1.1, 1.25, 0.0])  # Monday .. Sunday
    rows = []
    for s in range(n_stores):
        week = (dates - dates[0]).days // 7
        promo = ((week + s) % 2 == 0) & (dates.dayofweek < 5)
        season = 1 + 0.15 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
        noise = rng.lognormal(0, 0.06, n_days)
        sales = base[s] * dow_factor[dates.dayofweek] * season * (1 + promo_effect * promo) * noise
        opened = (dates.dayofweek != 6).astype(int)
        sales = np.where(opened == 1, sales, 0).round()
        rows.append(
            pd.DataFrame(
                {
                    "Store": s + 1,
                    "DayOfWeek": dates.dayofweek + 1,
                    "Date": dates,
                    "Sales": sales,
                    "Customers": (sales / 9).round(),
                    "Open": opened,
                    "Promo": promo.astype(int),
                    "StateHoliday": "0",
                    "SchoolHoliday": ((dates.month == 8) | (dates.month == 12)).astype(int),
                }
            )
        )
    train = pd.concat(rows, ignore_index=True)
    store = pd.DataFrame(
        {
            "Store": np.arange(1, n_stores + 1),
            "StoreType": rng.choice(list("abcd"), n_stores),
            "Assortment": rng.choice(list("abc"), n_stores),
            "CompetitionDistance": rng.uniform(100, 10000, n_stores),
            "CompetitionOpenSinceMonth": rng.integers(1, 13, n_stores).astype(float),
            "CompetitionOpenSinceYear": rng.integers(2005, 2012, n_stores).astype(float),
            "Promo2": rng.integers(0, 2, n_stores),
            "Promo2SinceWeek": np.where(rng.random(n_stores) < 0.5, 10.0, np.nan),
            "Promo2SinceYear": np.where(rng.random(n_stores) < 0.5, 2012.0, np.nan),
            "PromoInterval": rng.choice(["Jan,Apr,Jul,Oct", "Feb,May,Aug,Nov", ""], n_stores),
        }
    )
    return train, store


def synthetic_panel(**kwargs) -> Panel:
    train, store = make_synthetic(**kwargs)
    return build_panel(train, store)
