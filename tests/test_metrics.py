import numpy as np
import pytest

from rossmann_forecast import metrics


def test_known_values():
    y = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 180.0, 330.0])
    assert metrics.mae(y, p) == pytest.approx(20.0)
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((100 + 400 + 900) / 3))
    assert metrics.wape(y, p) == pytest.approx(60 / 600)
    assert metrics.mape(y, p) == pytest.approx(np.mean([0.1, 0.1, 0.1]))
    assert metrics.rmspe(y, p) == pytest.approx(0.1)


def test_perfect_forecast_is_zero():
    y = np.array([5.0, 7.0, 9.0])
    assert all(v == 0 for v in metrics.all_metrics(y, y).values())


def test_zero_actuals_are_ignored_for_percentage_metrics():
    y = np.array([0.0, 100.0])
    p = np.array([50.0, 90.0])
    assert metrics.rmspe(y, p) == pytest.approx(0.1)
    assert metrics.mape(y, p) == pytest.approx(0.1)


def test_wape_weights_large_stores_more_than_mape():
    y = np.array([10.0, 1000.0])
    p = np.array([20.0, 1000.0])  # 100% error on the small row only
    assert metrics.mape(y, p) == pytest.approx(0.5)
    assert metrics.wape(y, p) == pytest.approx(10 / 1010)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.rmse([1, 2], [1])
