import pytest

from rossmann_forecast.synthetic import synthetic_panel


@pytest.fixture
def panel():
    return synthetic_panel(n_stores=6, n_days=520, seed=1)
