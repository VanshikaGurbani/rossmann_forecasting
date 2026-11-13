"""Aggregate fold predictions into the metrics tables reported in docs/RESULTS.md."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import all_metrics, wape

POINT_MODELS = ["naive_last", "snaive", "snaive_mean4", "last_year", "random_forest", "lightgbm"]
SAMPLE_MODELS = POINT_MODELS + ["prophet_plain", "prophet_promo"]
METRICS = ["wape", "rmse", "mae", "mape", "rmspe"]


def _summarise(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(per_fold)
    return {
        "mean": frame.mean().to_dict(),
        "std": frame.std(ddof=1).to_dict() if len(frame) > 1 else {c: float("nan") for c in frame},
    }


def model_table(preds: pd.DataFrame, models: list[str]) -> dict:
    out = {}
    for model in models:
        if model not in preds.columns:
            continue
        per_fold = []
        for fold, grp in preds.groupby("fold"):
            valid = grp.dropna(subset=[model])
            m = all_metrics(valid["y"], valid[model])
            m["fold"] = int(fold)
            m["n_rows"] = int(len(valid))
            per_fold.append(m)
        summary = _summarise([{k: v for k, v in m.items() if k in METRICS} for m in per_fold])
        out[model] = {"per_fold": per_fold, **summary}
    return out


def store_error_table(
    preds: pd.DataFrame, models=("lightgbm", "last_year", "snaive_mean4")
) -> dict:
    """Distribution over (fold, store) pairs of store-level WAPE."""
    rows = []
    for (fold, store), grp in preds.groupby(["fold", "store_id"]):
        row = {"fold": int(fold), "store_id": int(store), "mean_sales": float(grp["y"].mean())}
        for model in models:
            row[model] = wape(grp["y"], grp[model])
        row["lightgbm_mae"] = float(np.mean(np.abs(grp["y"] - grp["lightgbm"])))
        rows.append(row)
    frame = pd.DataFrame(rows)
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    dist = {m: {f"p{int(q * 100)}": float(frame[m].quantile(q)) for q in quantiles} for m in models}
    for m in models:
        dist[m]["max"] = float(frame[m].max())
    beat = {m: float((frame["lightgbm"] < frame[m]).mean()) for m in models if m != "lightgbm"}
    by_store = frame.groupby("store_id").agg(
        wape=("lightgbm", "mean"), mean_sales=("mean_sales", "mean")
    )
    worst = by_store.sort_values("wape", ascending=False).head(5)
    return {
        "n_store_folds": int(len(frame)),
        "wape_quantiles": dist,
        "share_store_folds_lightgbm_beats": beat,
        "worst_stores": [
            {"store": int(s), "wape": float(r["wape"]), "mean_sales": float(r["mean_sales"])}
            for s, r in worst.iterrows()
        ],
        "store_fold_frame": frame,
    }


def interval_table(preds: pd.DataFrame, levels=(80, 90), origins=None) -> dict:
    out = {}
    for tag in levels:
        lo, hi = preds[f"lo{tag}"], preds[f"hi{tag}"]
        inside = (preds["y"] >= lo) & (preds["y"] <= hi)
        per_fold = []
        for fold, grp in preds.groupby("fold"):
            g_in = (grp["y"] >= grp[f"lo{tag}"]) & (grp["y"] <= grp[f"hi{tag}"])
            width = grp[f"hi{tag}"] - grp[f"lo{tag}"]
            per_fold.append(
                {
                    "fold": int(fold),
                    "coverage": float(g_in.mean()),
                    "mean_width": float(width.mean()),
                    "median_relative_width": float((width / grp["lightgbm"]).median()),
                }
            )
        horizon_day = preds.groupby("fold")["date"].transform(lambda s: (s - s.min()).dt.days)
        bucket = pd.cut(
            horizon_day, [-1, 13, 27, 41], labels=["days 1-14", "days 15-28", "days 29-42"]
        )
        by_bucket = inside.groupby(bucket, observed=True).mean().to_dict()
        per_store = inside.groupby([preds["fold"], preds["store_id"]]).mean()
        out[str(tag)] = {
            "nominal": tag / 100,
            "per_fold": per_fold,
            "overall_coverage": float(inside.mean()),
            "coverage_by_horizon": {str(k): float(v) for k, v in by_bucket.items()},
            "per_store_fold_coverage_quantiles": {
                f"p{int(q * 100)}": float(per_store.quantile(q)) for q in (0.1, 0.5, 0.9)
            },
            "share_store_folds_within_10pts": float((np.abs(per_store - tag / 100) <= 0.10).mean()),
        }
    return out
