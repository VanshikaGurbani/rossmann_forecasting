"""Point-forecast metrics. All take arrays of actuals and predictions in original sales units."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    return y, p


def rmse(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y_true, y_pred) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(y - p)))


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error: sum|error| / sum|actual|."""
    y, p = _arrays(y_true, y_pred)
    denom = np.sum(np.abs(y))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y - p)) / denom)


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error over rows with a non-zero actual."""
    y, p = _arrays(y_true, y_pred)
    keep = y != 0
    return float(np.mean(np.abs((y[keep] - p[keep]) / y[keep])))


def rmspe(y_true, y_pred) -> float:
    """Root mean squared percentage error, the Kaggle metric; rows with zero actual are ignored."""
    y, p = _arrays(y_true, y_pred)
    keep = y != 0
    return float(np.sqrt(np.mean(((y[keep] - p[keep]) / y[keep]) ** 2)))


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "rmspe": rmspe(y_true, y_pred),
    }
