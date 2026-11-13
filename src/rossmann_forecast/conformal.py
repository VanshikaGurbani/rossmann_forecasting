"""Split-conformal prediction intervals on the log1p scale.

Calibration residuals come from a model that was fitted only on data before the calibration
window and scored on that window, which in turn ends before the forecast origin. This keeps
time order intact; the guarantee is approximate because forecast errors are not exchangeable
across seasons (the per-fold coverage table in docs/RESULTS.md shows how far off it is).
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores, level: float) -> float:
    """The ceil((n+1) * level)-th smallest score (finite-sample corrected quantile)."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * level)
    return float(scores[min(k, n) - 1])


def log_scores(y_true, log_pred) -> np.ndarray:
    return np.abs(np.log1p(np.asarray(y_true, dtype=float)) - np.asarray(log_pred, dtype=float))


def interval(log_pred, q: float) -> tuple[np.ndarray, np.ndarray]:
    log_pred = np.asarray(log_pred, dtype=float)
    lower = np.clip(np.expm1(log_pred - q), 0, None)
    upper = np.expm1(log_pred + q)
    return lower, upper


def coverage(y_true, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))
