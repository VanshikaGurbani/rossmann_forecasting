"""Rolling-origin (expanding window) fold definitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    test_start: pd.Timestamp  # forecast origin: first date being predicted
    test_end: pd.Timestamp  # last date being predicted, inclusive

    @property
    def train_end(self) -> pd.Timestamp:
        """Last date whose sales may be used for fitting or features."""
        return self.test_start - pd.Timedelta(days=1)


def rolling_origin_folds(origins, horizon: int, data_end) -> list[Fold]:
    """One fold per origin; the model sees everything strictly before the origin.

    Test windows must not overlap and must fit inside the data.
    """
    starts = sorted(pd.Timestamp(o) for o in origins)
    data_end = pd.Timestamp(data_end)
    folds = []
    for i, start in enumerate(starts, start=1):
        end = start + pd.Timedelta(days=horizon - 1)
        if end > data_end:
            raise ValueError(f"fold {i} ends {end.date()} after the data ends {data_end.date()}")
        if folds and start <= folds[-1].test_end:
            raise ValueError(f"fold {i} overlaps the previous test window")
        folds.append(Fold(index=i, test_start=start, test_end=end))
    return folds
