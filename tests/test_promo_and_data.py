import pandas as pd
import pytest

from rossmann_forecast.data import build_panel, check_data
from rossmann_forecast.promo import fixed_effects_lift
from rossmann_forecast.synthetic import make_synthetic, synthetic_panel


def test_fixed_effects_recovers_a_known_promo_effect_and_permutations_do_not():
    panel = synthetic_panel(n_stores=30, n_days=400, seed=2, promo_effect=0.25)
    out = fixed_effects_lift(panel, n_perm=20)
    assert out["lift_pct"] == pytest.approx(25.0, abs=3.0)
    assert out["ci95_low_pct"] < out["lift_pct"] < out["ci95_high_pct"]
    assert out["permutation_beta_abs_max"] < abs(out["beta_log"])
    assert out["permutation_p_value"] < 0.1


def test_check_data_explains_the_kaggle_step(tmp_path):
    ok, msg = check_data(tmp_path)
    assert not ok and "kaggle" in msg.lower()


def test_check_data_accepts_valid_files(tmp_path):
    train, store = make_synthetic(n_stores=2, n_days=10)
    train.to_csv(tmp_path / "train.csv", index=False)
    store.to_csv(tmp_path / "store.csv", index=False)
    assert check_data(tmp_path)[0]


def test_panel_drops_closed_days_from_sales():
    train, store = make_synthetic(n_stores=2, n_days=14)
    panel = build_panel(train, store)
    sunday = panel.sales.index.dayofweek == 6
    assert panel.sales.loc[sunday].isna().all().all()
    assert panel.sales.shape == (14, 2)
    assert panel.dates[0] == pd.Timestamp("2013-01-01")
