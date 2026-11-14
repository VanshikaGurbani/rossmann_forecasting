"""Compact data files the visualization notebook reads, so it runs without the Kaggle files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import Panel
from .metrics import wape

VIZ_DIR = "reports/viz_data"
FORECAST_COLUMNS = [
    "fold", "date", "store_id", "y", "lightgbm", "lo80", "hi80", "lo90", "hi90",
    "naive_last", "snaive", "snaive_mean4", "last_year", "random_forest",
    "prophet_plain", "prophet_promo",
]  # fmt: skip
HISTORY_DAYS = 70


def export_viz_data(
    panel: Panel,
    preds: pd.DataFrame,
    folds: list[dict],
    out_dir: str | Path = VIZ_DIR,
    history_days: int = HISTORY_DAYS,
) -> None:
    """Write forecasts (all folds) and the sales history before each forecast origin."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [c for c in FORECAST_COLUMNS if c in preds.columns]
    forecasts = preds[cols].copy()
    num = [c for c in cols if c not in ("fold", "date", "store_id")]
    forecasts[num] = forecasts[num].astype("float32").round(0)
    forecasts["store_id"] = forecasts["store_id"].astype("int16")
    forecasts["fold"] = forecasts["fold"].astype("int8")
    forecasts.to_parquet(out / "forecasts.parquet", index=False)

    frames = []
    for fold in folds:
        origin = pd.Timestamp(fold["test_start"])
        window = panel.sales.loc[
            origin - pd.Timedelta(days=history_days) : origin - pd.Timedelta(days=1)
        ]
        long = window.stack().dropna().rename("sales").reset_index()
        long.columns = ["date", "store_id", "sales"]
        long["fold"] = fold["fold"]
        frames.append(long)
    history = pd.concat(frames, ignore_index=True)
    history["store_id"] = history["store_id"].astype("int16")
    history["fold"] = history["fold"].astype("int8")
    history["sales"] = history["sales"].astype("float32").round(0)
    history.to_parquet(out / "history.parquet", index=False)


def load_forecasts(out_dir: str | Path = VIZ_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / "forecasts.parquet")


def load_history(out_dir: str | Path = VIZ_DIR) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / "history.parquet")


def load_results(path: str | Path = "reports/results.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def store_wape_table(forecasts: pd.DataFrame, model: str = "lightgbm") -> pd.DataFrame:
    """Store-level WAPE per (fold, store) with the store's mean daily sales."""
    rows = []
    for (fold, store), grp in forecasts.dropna(subset=[model]).groupby(["fold", "store_id"]):
        rows.append(
            {
                "fold": int(fold),
                "store_id": int(store),
                "mean_sales": float(grp["y"].mean()),
                "wape": wape(grp["y"], grp[model]),
                "mae": float((grp["y"] - grp[model]).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def running_wape(grp: pd.DataFrame, model: str, upto_day: int) -> float:
    """WAPE using only the first `upto_day` calendar days of the forecast window."""
    start = grp["date"].min()
    seen = grp[(grp["date"] - start).dt.days < upto_day].dropna(subset=[model])
    return wape(seen["y"], seen[model]) if len(seen) else float("nan")
