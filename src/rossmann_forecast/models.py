"""Global models: one fit over all stores, store identity and store history as features."""

from __future__ import annotations

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import LIGHTGBM_PARAMS, SEED
from .features import CATEGORICAL, feature_columns


@dataclass
class FittedModel:
    name: str
    model: object
    columns: list[str]
    fit_seconds: float

    def predict_log(self, feats: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(feats[self.columns]), dtype=float)

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return np.clip(np.expm1(self.predict_log(feats)), 0, None)


def training_rows(feats: pd.DataFrame, origin) -> pd.DataFrame:
    """Open days with positive sales strictly before the origin."""
    mask = (feats["date"] < pd.Timestamp(origin)) & (feats["is_open"] == 1) & feats["sales"].notna()
    return feats.loc[mask]


def forecast_rows(feats: pd.DataFrame, start, end) -> pd.DataFrame:
    """Open days inside [start, end]; `sales` is kept only so the caller can score."""
    mask = (
        (feats["date"] >= pd.Timestamp(start))
        & (feats["date"] <= pd.Timestamp(end))
        & (feats["is_open"] == 1)
    )
    return feats.loc[mask]


def fit_lightgbm(train: pd.DataFrame, params: dict | None = None) -> FittedModel:
    cols = feature_columns(train)
    cfg = {**LIGHTGBM_PARAMS, **(params or {})}
    start = time.perf_counter()
    model = lgb.LGBMRegressor(**cfg)
    model.fit(
        train[cols],
        np.log1p(train["sales"].to_numpy()),
        categorical_feature=[c for c in CATEGORICAL if c in cols],
    )
    return FittedModel("lightgbm", model, cols, time.perf_counter() - start)


def fit_random_forest(train: pd.DataFrame, n_estimators: int = 100) -> FittedModel:
    cols = feature_columns(train)
    start = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=14,
        min_samples_leaf=20,
        max_features=0.5,
        max_samples=0.3,
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(train[cols], np.log1p(train["sales"].to_numpy()))
    return FittedModel("random_forest", model, cols, time.perf_counter() - start)
