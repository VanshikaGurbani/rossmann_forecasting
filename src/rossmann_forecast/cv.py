"""One rolling-origin fold: build leak-safe features, fit models, forecast, calibrate intervals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import baseline_forecasts
from .config import CONFORMAL_LEVELS, HORIZON, MIN_LAG
from .conformal import conformal_quantile, interval, log_scores
from .data import Panel
from .features import build_features
from .models import FittedModel, fit_lightgbm, fit_random_forest, forecast_rows, training_rows
from .splits import Fold
from .statistical import prophet_forecasts


@dataclass
class FoldResult:
    fold: Fold
    predictions: pd.DataFrame  # one row per scored (open, sales > 0) store-day
    timings: dict[str, float] = field(default_factory=dict)
    conformal_q: dict[float, float] = field(default_factory=dict)
    n_calibration: int = 0
    lightgbm: FittedModel | None = None
    test_features: pd.DataFrame | None = None


def run_fold(
    panel: Panel,
    fold: Fold,
    *,
    horizon: int = HORIZON,
    min_lag: int = MIN_LAG,
    prophet_stores=None,
    include_rf: bool = True,
    levels=CONFORMAL_LEVELS,
    keep_model: bool = False,
    log=print,
) -> FoldResult:
    origin = fold.test_start
    timings: dict[str, float] = {}

    log(f"[fold {fold.index}] features (origin {origin.date()})")
    feats = build_features(panel, min_lag, known_until=origin)
    train = training_rows(feats, origin)
    test = forecast_rows(feats, fold.test_start, fold.test_end)

    log(f"[fold {fold.index}] LightGBM on {len(train):,} rows")
    lgb_model = fit_lightgbm(train)
    timings["lightgbm_fit"] = lgb_model.fit_seconds
    scored = test[test["sales"].notna()]
    preds = scored[["date", "store_id", "sales"]].rename(columns={"sales": "y"}).copy()
    lgb_log = lgb_model.predict_log(scored)
    preds["lightgbm"] = np.clip(np.expm1(lgb_log), 0, None)
    preds["lightgbm_log"] = lgb_log

    if include_rf:
        log(f"[fold {fold.index}] random forest")
        rf = fit_random_forest(train)
        timings["random_forest_fit"] = rf.fit_seconds
        preds["random_forest"] = rf.predict(scored)

    # calibration: refit one horizon earlier, score on the window just before the origin
    cal_origin = origin - pd.Timedelta(days=horizon)
    log(f"[fold {fold.index}] conformal calibration (origin {cal_origin.date()})")
    feats_c = build_features(panel, min_lag, known_until=cal_origin)
    train_c = training_rows(feats_c, cal_origin)
    model_c = fit_lightgbm(train_c)
    timings["calibration_fit"] = model_c.fit_seconds
    cal = forecast_rows(feats_c, cal_origin, origin - pd.Timedelta(days=1))
    cal = cal[cal["sales"].notna()]
    cal_scores = log_scores(cal["sales"].to_numpy(), model_c.predict_log(cal))
    q_by_level = {}
    for level in levels:
        q = conformal_quantile(cal_scores, level)
        lo, hi = interval(lgb_log, q)
        tag = int(round(level * 100))
        preds[f"lo{tag}"], preds[f"hi{tag}"] = lo, hi
        q_by_level[level] = q

    base = baseline_forecasts(panel, fold)
    preds = preds.merge(base, on=["date", "store_id"], how="left")

    if prophet_stores is not None and len(prophet_stores):
        log(f"[fold {fold.index}] Prophet on {len(prophet_stores)} stores x 2 variants")
        proph, secs = prophet_forecasts(panel, fold, prophet_stores)
        preds = preds.merge(proph, on=["date", "store_id"], how="left")
        timings["prophet_plain_fit_per_store"] = secs["prophet_plain"]
        timings["prophet_promo_fit_per_store"] = secs["prophet_promo"]

    preds.insert(0, "fold", fold.index)
    return FoldResult(
        fold=fold,
        predictions=preds,
        timings=timings,
        conformal_q=q_by_level,
        n_calibration=len(cal_scores),
        lightgbm=lgb_model if keep_model else None,
        test_features=scored if keep_model else None,
    )
