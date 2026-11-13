import numpy as np
import pytest

from rossmann_forecast.conformal import conformal_quantile, coverage, interval, log_scores


def test_quantile_uses_finite_sample_correction():
    scores = np.arange(1.0, 10.0)  # n = 9
    assert conformal_quantile(scores, 0.8) == 8.0  # ceil(10 * 0.8) = 8th smallest
    assert conformal_quantile(scores, 0.99) == 9.0  # capped at the maximum


def test_empty_calibration_raises():
    with pytest.raises(ValueError):
        conformal_quantile([], 0.9)


def test_interval_is_ordered_and_non_negative():
    lo, hi = interval(np.log1p([0.0, 100.0, 5000.0]), 0.3)
    assert np.all(lo >= 0) and np.all(lo <= hi)


def test_coverage_is_close_to_nominal_on_exchangeable_data():
    rng = np.random.default_rng(0)
    truth = rng.uniform(1000, 9000, 40000)
    pred_log = np.log1p(truth) + rng.normal(0, 0.1, truth.size)
    cal, test = slice(0, 20000), slice(20000, None)
    q = conformal_quantile(log_scores(truth[cal], pred_log[cal]), 0.9)
    lo, hi = interval(pred_log[test], q)
    assert coverage(truth[test], lo, hi) == pytest.approx(0.9, abs=0.01)
