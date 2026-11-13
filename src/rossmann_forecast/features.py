"""Feature construction with an explicit minimum lag on every sales-derived feature.

Rule enforced here and tested in tests/test_features.py: a feature for target date t may read
realised sales only up to t - min_lag. With min_lag equal to the forecast horizon, no feature
for any date in a forecast window depends on sales inside that window.
Calendar, promo, holiday and open-day columns are planned in advance and are used as known
covariates for the target date.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .data import Panel

CATEGORICAL = ["store", "store_type", "assortment", "state_holiday"]
MAX_MIN_LAG = 357  # the last-year features read t-357 at the latest


def _nanmean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    stack = np.stack([f.to_numpy() for f in frames])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
    return pd.DataFrame(mean, index=frames[0].index, columns=frames[0].columns)


def sales_features(
    sales: pd.DataFrame, promo: pd.DataFrame, min_lag: int
) -> dict[str, pd.DataFrame]:
    """Sales-derived features. Row t only reads rows <= t - min_lag of `sales`."""
    if not 1 <= min_lag <= MAX_MIN_LAG:
        raise ValueError(f"min_lag must be in [1, {MAX_MIN_LAG}]")
    lagged = sales.shift(min_lag)
    out: dict[str, pd.DataFrame] = {}
    out["roll28_mean"] = lagged.rolling(28, min_periods=7).mean()
    out["roll91_mean"] = lagged.rolling(91, min_periods=14).mean()
    out["roll91_std"] = lagged.rolling(91, min_periods=14).std()
    promo_sales = sales.where(promo).shift(min_lag).rolling(91, min_periods=5).mean()
    base_sales = sales.where(~promo).shift(min_lag).rolling(91, min_periods=5).mean()
    out["roll91_promo_mean"] = promo_sales
    out["roll91_nopromo_mean"] = base_sales
    out["promo_ratio_91"] = promo_sales / base_sales
    # same weekday, most recent occurrences that are at least min_lag days old
    k0 = 7 * math.ceil(min_lag / 7)
    dow_frames = [sales.shift(k0 + 7 * i) for i in range(8)]
    out["dow_last"] = dow_frames[0]
    out["dow_mean4"] = _nanmean(dow_frames[:4])
    out["dow_mean8"] = _nanmean(dow_frames)
    # last year, same weekday
    out["ly_364"] = sales.shift(364)
    out["ly_mean3"] = _nanmean([sales.shift(357), sales.shift(364), sales.shift(371)])
    return out


def _promo2_active(panel: Panel) -> np.ndarray:
    static = panel.static
    dates = panel.dates
    iso = dates.isocalendar()
    year = iso["year"].to_numpy().astype(int)[:, None]
    week = iso["week"].to_numpy().astype(int)[:, None]
    since_year = static["Promo2SinceYear"].fillna(9999).to_numpy()[None, :]
    since_week = static["Promo2SinceWeek"].fillna(99).to_numpy()[None, :]
    started = (year > since_year) | ((year == since_year) & (week >= since_week))
    month_names = np.array(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    )
    intervals = static["PromoInterval"].fillna("").astype(str).to_numpy()
    month_ok = np.zeros((len(dates), len(intervals)), dtype=bool)
    for j, text in enumerate(intervals):
        if not text:
            continue
        months = {m.strip() for m in text.split(",")}
        flags = np.array([m in months for m in month_names])
        month_ok[:, j] = flags[dates.month.to_numpy() - 1]
    promo2 = static["Promo2"].fillna(0).to_numpy()[None, :] == 1
    return (started & month_ok & promo2).astype(int)


def build_features(panel: Panel, min_lag: int, known_until=None) -> pd.DataFrame:
    """Long frame with one row per (date, store) and all model features.

    known_until: sales on dates >= this timestamp are hidden before any feature is built,
    so nothing from a forecast window can reach the features even by accident.
    """
    sales = panel.sales
    if known_until is not None:
        sales = sales.copy()
        sales.loc[sales.index >= pd.Timestamp(known_until)] = np.nan
    dates, stores = panel.dates, panel.stores
    nd, ns = len(dates), len(stores)

    def flat(matrix) -> np.ndarray:
        return np.asarray(matrix, dtype=np.float64).reshape(-1)

    cols: dict[str, np.ndarray] = {}
    cols["date"] = np.repeat(dates.to_numpy(), ns)
    cols["store_id"] = np.tile(stores, nd)
    for name, frame in sales_features(sales, panel.promo, min_lag).items():
        cols[name] = flat(frame.to_numpy())

    promo = panel.promo.astype(int)
    opened = panel.open.astype(int)
    holiday = (panel.state_holiday > 0).astype(int)
    cols["promo"] = flat(promo.to_numpy())
    cols["promo_prev"] = flat(promo.shift(1).fillna(0).to_numpy())
    cols["promo_next"] = flat(promo.shift(-1).fillna(0).to_numpy())
    cols["promo_7d"] = flat(promo.rolling(7, min_periods=1).sum().to_numpy())
    cols["school_holiday"] = flat(panel.school_holiday.astype(int).to_numpy())
    cols["state_holiday"] = flat(panel.state_holiday.to_numpy())
    cols["holiday_prev"] = flat(holiday.shift(1).fillna(0).to_numpy())
    cols["holiday_next"] = flat(holiday.shift(-1).fillna(0).to_numpy())
    cols["open_prev"] = flat(opened.shift(1).fillna(1).to_numpy())
    cols["open_next"] = flat(opened.shift(-1).fillna(1).to_numpy())
    cols["promo2_active"] = flat(_promo2_active(panel))

    cols["dow"] = np.repeat(dates.dayofweek.to_numpy(), ns).astype(float)
    cols["day"] = np.repeat(dates.day.to_numpy(), ns).astype(float)
    cols["month"] = np.repeat(dates.month.to_numpy(), ns).astype(float)
    cols["week"] = np.repeat(dates.isocalendar()["week"].to_numpy().astype(int), ns).astype(float)
    cols["doy"] = np.repeat(dates.dayofyear.to_numpy(), ns).astype(float)

    static = panel.static
    cols["store"] = np.tile(np.arange(ns), nd).astype(float)
    cols["store_type"] = np.tile(
        static["StoreType"].astype("category").cat.codes.to_numpy(), nd
    ).astype(float)
    cols["assortment"] = np.tile(
        static["Assortment"].astype("category").cat.codes.to_numpy(), nd
    ).astype(float)
    cols["comp_distance"] = np.tile(static["CompetitionDistance"].to_numpy(dtype=float), nd)
    comp_year = static["CompetitionOpenSinceYear"].to_numpy(dtype=float)[None, :]
    comp_month = static["CompetitionOpenSinceMonth"].to_numpy(dtype=float)[None, :]
    months_open = (dates.year.to_numpy()[:, None] - comp_year) * 12 + (
        dates.month.to_numpy()[:, None] - comp_month
    )
    cols["comp_open_months"] = flat(months_open)

    cols["is_open"] = flat(opened.to_numpy())
    feats = pd.DataFrame(cols)
    float_cols = [c for c in feats.select_dtypes("float64").columns if c != "store_id"]
    feats[float_cols] = feats[float_cols].astype("float32")
    # actual sales stay attached for scoring and for building training targets
    feats["sales"] = flat(panel.sales.to_numpy())
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    drop = {"date", "store_id", "sales", "is_open"}
    return [c for c in feats.columns if c not in drop]
