"""End-to-end run: rolling-origin CV, intervals, promo estimates, SHAP, report files."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import DEFAULT_ORIGINS, HORIZON, PROPHET_SAMPLE_STORES, SEED
from .cv import run_fold
from .data import Panel
from .evaluation import POINT_MODELS, SAMPLE_MODELS, interval_table, model_table, store_error_table
from .explain import shap_importance
from .promo import fixed_effects_lift, model_counterfactual
from .splits import rolling_origin_folds
from .statistical import sample_stores


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _versions() -> dict[str, str]:
    import lightgbm
    import sklearn

    out = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lightgbm.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        import prophet

        out["prophet"] = prophet.__version__
    except Exception:
        pass
    return out


def run_all(
    panel: Panel,
    *,
    out_dir: str | Path = "reports",
    docs_dir: str | Path = "docs",
    readme: str | Path | None = "README.md",
    cache_dir: str | Path = "data/processed",
    origins=DEFAULT_ORIGINS,
    horizon: int = HORIZON,
    n_prophet_stores: int = PROPHET_SAMPLE_STORES,
    include_rf: bool = True,
    include_prophet: bool = True,
    n_perm: int = 100,
    log=print,
) -> dict:
    out_dir, cache_dir = Path(out_dir), Path(cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    folds = rolling_origin_folds(origins, horizon, panel.dates[-1])
    sample = sample_stores(panel.stores, n_prophet_stores) if include_prophet else None
    results = [
        run_fold(
            panel,
            fold,
            horizon=horizon,
            min_lag=horizon,
            prophet_stores=sample,
            include_rf=include_rf,
            keep_model=True,
            log=log,
        )
        for fold in folds
    ]
    preds = pd.concat([r.predictions for r in results], ignore_index=True)
    preds.to_parquet(cache_dir / "predictions.parquet", index=False)

    point_models = [m for m in POINT_MODELS if m in preds.columns]
    summary = {
        "meta": {
            "package_version": __version__,
            "git_commit": _git_commit(),
            "versions": _versions(),
            "seed": SEED,
            "horizon_days": horizon,
            "min_lag_days": horizon,
            "n_stores": int(len(panel.stores)),
            "data_start": str(panel.dates[0].date()),
            "data_end": str(panel.dates[-1].date()),
            "n_prophet_stores": int(len(sample)) if sample is not None else 0,
            "prophet_store_ids": [int(s) for s in sample] if sample is not None else [],
        },
        "folds": [
            {
                "fold": r.fold.index,
                "test_start": str(r.fold.test_start.date()),
                "test_end": str(r.fold.test_end.date()),
                "n_scored_rows": int(len(r.predictions)),
                "n_stores": int(r.predictions["store_id"].nunique()),
                "mean_daily_sales": float(r.predictions["y"].mean()),
                "n_calibration_rows": r.n_calibration,
                "conformal_q_log": {str(k): v for k, v in r.conformal_q.items()},
            }
            for r in results
        ],
        "all_stores": model_table(preds, point_models),
        "timing_seconds": {f"fold{r.fold.index}": r.timings for r in results},
    }
    if sample is not None:
        on_sample = preds[preds["store_id"].isin(sample)]
        summary["sample_stores"] = model_table(on_sample, SAMPLE_MODELS)

    errors = store_error_table(preds)
    store_frame = errors.pop("store_fold_frame")
    summary["store_error"] = errors
    summary["intervals"] = interval_table(preds)

    log("promo: model counterfactual per fold")
    summary["promo_model_counterfactual"] = [
        {"fold": r.fold.index, **model_counterfactual(r.lightgbm, r.test_features)} for r in results
    ]
    log("promo: fixed-effects regression")
    summary["promo_fixed_effects"] = fixed_effects_lift(panel, n_perm=n_perm)

    log("SHAP")
    last = results[-1]
    importance = shap_importance(last.lightgbm, last.test_features, out_dir=out_dir / "figures")
    summary["shap_top"] = {k: float(v) for k, v in importance.head(12).items()}
    summary["total_runtime_seconds"] = time.perf_counter() - started

    store_frame.to_csv(cache_dir / "store_fold_errors.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    from .report import write_reports
    from .viz_data import export_viz_data

    export_viz_data(panel, preds, summary["folds"], out_dir=out_dir / "viz_data")
    write_reports(
        panel, preds, store_frame, summary, out_dir=out_dir, docs_dir=docs_dir, readme=readme
    )
    return summary
