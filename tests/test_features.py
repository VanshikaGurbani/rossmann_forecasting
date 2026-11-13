"""The no-look-ahead guarantee: features for date t read sales only up to t - min_lag."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from rossmann_forecast.features import build_features, feature_columns, sales_features


def _features_on(frame: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    sub = frame[frame["date"] == date].sort_values("store_id")
    return sub[feature_columns(frame)].reset_index(drop=True)


def _scramble(panel, after_date, before_date=None):
    """Copy of the panel with sales scrambled for dates in (after_date, before_date]."""
    sales = panel.sales.copy()
    mask = sales.index > after_date
    if before_date is not None:
        mask &= sales.index <= before_date
    sales.loc[mask] = sales.loc[mask] * 7.0 + 1234.0
    return dataclasses.replace(panel, sales=sales)


@pytest.mark.parametrize("min_lag", [1, 14, 42])
def test_features_at_t_ignore_sales_after_t_minus_lag(panel, min_lag):
    base = build_features(panel, min_lag)
    for pos in (400, 450, 500):
        target = panel.dates[pos]
        cutoff = target - pd.Timedelta(days=min_lag)
        changed = build_features(_scramble(panel, cutoff), min_lag)
        pd.testing.assert_frame_equal(
            _features_on(base, target), _features_on(changed, target), check_exact=True
        )


def test_the_test_has_power_usable_history_does_change_features(panel):
    min_lag, target = 14, panel.dates[450]
    cutoff = target - pd.Timedelta(days=min_lag)
    base = build_features(panel, min_lag)
    changed = build_features(_scramble(panel, cutoff - pd.Timedelta(days=60), cutoff), min_lag)
    assert not _features_on(base, target).equals(_features_on(changed, target)), (
        "scrambling usable history must change features, otherwise the leak test is vacuous"
    )


def test_known_until_hides_the_forecast_window(panel):
    origin = panel.dates[480]
    garbage = _scramble(panel, origin - pd.Timedelta(days=1))
    a = build_features(panel, 14, known_until=origin)
    b = build_features(garbage, 14, known_until=origin)
    cols = feature_columns(a)
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)


def test_min_lag_bounds():
    frame = pd.DataFrame(np.ones((400, 2)))
    with pytest.raises(ValueError):
        sales_features(frame, frame.astype(bool), 0)
    with pytest.raises(ValueError):
        sales_features(frame, frame.astype(bool), 400)


def test_feature_columns_exclude_target_and_ids(panel):
    cols = feature_columns(build_features(panel, 14))
    assert not {"sales", "date", "store_id", "is_open"} & set(cols)
