"""Per-store Prophet baselines, fitted separately for each store on a fixed random sample."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .config import SEED
from .data import Panel
from .splits import Fold


def sample_stores(stores, n: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stores = np.asarray(stores)
    return np.sort(rng.choice(stores, size=min(n, len(stores)), replace=False))


def _fit_one(
    store: int,
    train_df: pd.DataFrame,
    future_df: pd.DataFrame,
    use_regressors: bool,
) -> tuple[int, np.ndarray, float]:
    from prophet import Prophet  # imported here so worker processes load it themselves

    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    start = time.perf_counter()
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    if use_regressors:
        for reg in ("promo", "state_holiday", "school_holiday"):
            model.add_regressor(reg)
    model.fit(train_df)
    yhat = model.predict(future_df)["yhat"].to_numpy()
    return store, np.clip(yhat, 0, None), time.perf_counter() - start


def prophet_forecasts(
    panel: Panel, fold: Fold, stores, n_jobs: int | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit Prophet per store on history before the origin; forecast the fold window.

    Two variants: `prophet_plain` (no regressors) and `prophet_promo` (same planned promo and
    holiday flags the global model receives). Returns predictions and mean seconds per fit.
    """
    dates = panel.dates
    pos0 = int(dates.get_loc(fold.test_start))
    window = pd.date_range(fold.test_start, fold.test_end)
    wpos = [int(dates.get_loc(d)) for d in window]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 2)

    jobs = []
    for variant, use_reg in (("prophet_plain", False), ("prophet_promo", True)):
        for store in stores:
            j = int(np.where(panel.stores == store)[0][0])
            hist_mask = panel.sales.iloc[:pos0, j].notna().to_numpy()
            reg_cols = {
                "promo": panel.promo.iloc[:, j].to_numpy().astype(float),
                "state_holiday": (panel.state_holiday.iloc[:, j].to_numpy() > 0).astype(float),
                "school_holiday": panel.school_holiday.iloc[:, j].to_numpy().astype(float),
            }
            train_df = pd.DataFrame(
                {
                    "ds": dates[:pos0][hist_mask],
                    "y": panel.sales.iloc[:pos0, j].to_numpy()[hist_mask],
                }
            )
            future_df = pd.DataFrame({"ds": window})
            if use_reg:
                for name, arr in reg_cols.items():
                    train_df[name] = arr[:pos0][hist_mask]
                    future_df[name] = arr[wpos]
            jobs.append((variant, store, train_df, future_df, use_reg))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one)(store, tr, fu, reg) for _, store, tr, fu, reg in jobs
    )
    frames = {}
    seconds: dict[str, list[float]] = {"prophet_plain": [], "prophet_promo": []}
    for (variant, _, _, future_df, _), (store, yhat, secs) in zip(jobs, results, strict=True):
        frames.setdefault(variant, []).append(
            pd.DataFrame({"date": future_df["ds"], "store_id": store, variant: yhat})
        )
        seconds[variant].append(secs)
    plain = pd.concat(frames["prophet_plain"], ignore_index=True)
    promo = pd.concat(frames["prophet_promo"], ignore_index=True)
    merged = plain.merge(promo, on=["date", "store_id"])
    return merged, {k: float(np.mean(v)) for k, v in seconds.items()}
