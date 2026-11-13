"""Naive baselines every model has to beat. They only read sales strictly before the origin."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .data import Panel
from .splits import Fold

BASELINES = ["naive_last", "snaive", "snaive_mean4", "last_year"]


def _first_valid(candidates: list[np.ndarray]) -> np.ndarray:
    out = candidates[0].copy()
    for cand in candidates[1:]:
        out = np.where(np.isnan(out), cand, out)
    return out


def baseline_forecasts(panel: Panel, fold: Fold) -> pd.DataFrame:
    """Long frame (date, store_id, <baseline columns>) for every date of the fold's window.

    naive_last    last observed (open-day) sales of the store before the origin
    snaive        sales on the most recent same weekday before the origin
    snaive_mean4  mean of the four most recent same-weekday sales before the origin
    last_year     sales 364 days earlier (same weekday last year), falling back to snaive_mean4
    """
    dates = panel.dates
    origin = fold.test_start
    pos0 = int(dates.get_loc(origin))
    horizon = (fold.test_end - fold.test_start).days + 1
    values = panel.sales.to_numpy().copy()
    values[pos0:] = np.nan  # nothing at or after the origin is visible
    ns = values.shape[1]

    history = values[:pos0]
    last_idx = pd.DataFrame(history).ffill().to_numpy()[-1]  # last non-NaN per store

    rows = []
    for j in range(horizon):
        t = pos0 + j
        k = 7 * math.ceil((j + 1) / 7)  # smallest multiple of 7 that lands before the origin
        same_dow = [values[t - k - 7 * i] if t - k - 7 * i >= 0 else np.full(ns, np.nan)
                    for i in range(8)]  # fmt: skip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean4 = np.nanmean(np.stack(same_dow[:4]), axis=0)
        snaive = _first_valid(same_dow)
        mean4 = np.where(np.isnan(mean4), snaive, mean4)
        ly = values[t - 364] if t - 364 >= 0 else np.full(ns, np.nan)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates[t],
                    "store_id": panel.stores,
                    "naive_last": last_idx,
                    "snaive": snaive,
                    "snaive_mean4": mean4,
                    "last_year": _first_valid([ly, mean4]),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)
