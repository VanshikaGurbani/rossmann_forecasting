import pandas as pd
import pytest

from rossmann_forecast.cv import run_fold
from rossmann_forecast.splits import rolling_origin_folds
from rossmann_forecast.synthetic import synthetic_panel
from rossmann_forecast.viz_data import (
    export_viz_data,
    load_forecasts,
    load_history,
    running_wape,
    store_wape_table,
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    panel = synthetic_panel(n_stores=6, n_days=520, seed=4)
    origin = panel.dates[-1] - pd.Timedelta(days=13)
    (fold,) = rolling_origin_folds([origin], 14, panel.dates[-1])
    result = run_fold(panel, fold, horizon=14, min_lag=14, include_rf=False, log=lambda *_: None)
    out = tmp_path_factory.mktemp("dash")
    meta = [{"fold": 1, "test_start": str(fold.test_start.date())}]
    export_viz_data(panel, result.predictions, meta, out_dir=out, history_days=30)
    return out, fold


def test_export_round_trip(exported):
    out, fold = exported
    forecasts, history = load_forecasts(out), load_history(out)
    assert {"fold", "date", "store_id", "y", "lightgbm", "lo90", "hi90"} <= set(forecasts.columns)
    assert forecasts["date"].min() >= fold.test_start
    assert history["date"].max() < fold.test_start  # history never reaches into the forecast window


def test_store_table_and_running_wape(exported):
    forecasts = load_forecasts(exported[0])
    table = store_wape_table(forecasts)
    assert len(table) == forecasts["store_id"].nunique()
    one = forecasts[forecasts["store_id"] == forecasts["store_id"].iloc[0]]
    assert 0 <= running_wape(one, "lightgbm", 14) < 1
    assert running_wape(one, "lightgbm", 1) != running_wape(one, "lightgbm", 14)
