import dataclasses

import numpy as np
import pandas as pd

from rossmann_forecast.baselines import baseline_forecasts
from rossmann_forecast.splits import rolling_origin_folds


def test_snaive_reproduces_a_purely_weekly_series(panel):
    weekly = np.array([10.0, 12, 14, 16, 18, 20, np.nan])  # Monday .. Sunday
    weekday = panel.sales.index.dayofweek.to_numpy()
    panel.sales.loc[:, :] = np.repeat(weekly[weekday][:, None], panel.sales.shape[1], axis=1)
    origin = panel.dates[-30]
    (fold,) = rolling_origin_folds([origin], 14, panel.dates[-1])
    out = baseline_forecasts(panel, fold)
    open_rows = out[out["date"].dt.dayofweek != 6]
    expected = weekly[open_rows["date"].dt.dayofweek.to_numpy()]
    np.testing.assert_allclose(open_rows["snaive"], expected)
    np.testing.assert_allclose(open_rows["snaive_mean4"], expected)


def test_baselines_only_use_data_before_origin(panel):
    origin = panel.dates[-40]
    (fold,) = rolling_origin_folds([origin], 14, panel.dates[-1])
    base = baseline_forecasts(panel, fold)
    sales = panel.sales.copy()
    sales.loc[sales.index >= origin] = 1e9
    again = baseline_forecasts(dataclasses.replace(panel, sales=sales), fold)
    pd.testing.assert_frame_equal(base, again)
