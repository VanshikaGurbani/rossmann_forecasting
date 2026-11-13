"""SHAP attributions for the global LightGBM model (predictive attribution, not causal)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SEED
from .models import FittedModel


def shap_importance(
    model: FittedModel, rows: pd.DataFrame, n: int = 5000, seed: int = SEED, out_dir=None
) -> pd.Series:
    """Mean |SHAP| per feature (log1p-sales units) on a random sample of forecast rows."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sample = rows.sample(min(n, len(rows)), random_state=seed)[model.columns]
    values = shap.TreeExplainer(model.model).shap_values(sample)
    importance = pd.Series(np.abs(values).mean(axis=0), index=model.columns).sort_values(
        ascending=False
    )
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(values, sample, show=False, max_display=15)
        plt.title("SHAP summary, global LightGBM (final fold)")
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=130)
        plt.close()
    return importance
