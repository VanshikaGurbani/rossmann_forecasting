import pandas as pd
import pytest

from rossmann_forecast.splits import rolling_origin_folds


def test_folds_are_ordered_non_overlapping_and_train_precedes_test():
    folds = rolling_origin_folds(["2014-12-01", "2014-08-01", "2015-06-20"], 42, "2015-07-31")
    assert [f.test_start for f in folds] == sorted(f.test_start for f in folds)
    for prev, nxt in zip(folds, folds[1:], strict=False):
        assert prev.test_end < nxt.test_start
    for f in folds:
        assert f.train_end < f.test_start
        assert (f.test_end - f.test_start).days + 1 == 42


def test_training_window_expands():
    folds = rolling_origin_folds(["2014-08-01", "2014-12-01"], 42, "2015-07-31")
    assert folds[0].train_end < folds[1].train_end


def test_rejects_overlap_and_overrun():
    with pytest.raises(ValueError):
        rolling_origin_folds(["2015-01-01", "2015-01-20"], 42, "2015-07-31")
    with pytest.raises(ValueError):
        rolling_origin_folds(["2015-07-01"], 42, "2015-07-31")


def test_train_end_is_day_before_origin():
    (fold,) = rolling_origin_folds(["2015-06-20"], 42, "2015-07-31")
    assert fold.train_end == pd.Timestamp("2015-06-19")
    assert fold.test_end == pd.Timestamp("2015-07-31")
