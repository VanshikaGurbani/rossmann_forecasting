"""End to end: scrambling sales inside the forecast window must not change any forecast."""

import numpy as np
import pandas as pd

from rossmann_forecast.cv import run_fold
from rossmann_forecast.features import build_features
from rossmann_forecast.models import forecast_rows, training_rows
from rossmann_forecast.splits import rolling_origin_folds
from rossmann_forecast.synthetic import synthetic_panel

HORIZON = 14


def _quiet(*_args):
    return None


def test_forecasts_do_not_depend_on_sales_inside_the_window():
    panel = synthetic_panel(n_stores=6, n_days=520, seed=3)
    origin = panel.dates[-1] - pd.Timedelta(days=HORIZON - 1)
    (fold,) = rolling_origin_folds([origin], HORIZON, panel.dates[-1])
    kwargs = {"horizon": HORIZON, "min_lag": HORIZON, "include_rf": False, "log": _quiet}
    baseline = run_fold(panel, fold, **kwargs)

    scrambled = synthetic_panel(n_stores=6, n_days=520, seed=3)
    window = (scrambled.sales.index >= fold.test_start) & (scrambled.sales.index <= fold.test_end)
    noise = np.random.default_rng(0).uniform(0.2, 5.0, scrambled.sales.loc[window].shape)
    scrambled.sales.loc[window] = scrambled.sales.loc[window].to_numpy() * noise
    changed = run_fold(scrambled, fold, **kwargs)

    a, b = baseline.predictions, changed.predictions
    assert not np.allclose(a["y"], b["y"]), "the actuals themselves should differ"
    forecast_cols = [c for c in a.columns if c != "y"]
    pd.testing.assert_frame_equal(a[forecast_cols], b[forecast_cols], rtol=1e-9)


def test_training_rows_are_strictly_before_origin():
    panel = synthetic_panel(n_stores=4, n_days=450, seed=5)
    origin = panel.dates[400]
    feats = build_features(panel, 14, known_until=origin)
    train = training_rows(feats, origin)
    test = forecast_rows(feats, origin, origin + pd.Timedelta(days=13))
    assert train["date"].max() < origin <= test["date"].min()
