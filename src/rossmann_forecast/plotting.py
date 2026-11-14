"""Shared plotting style and names for the notebooks."""

from __future__ import annotations

from pathlib import Path

NAMES = {
    "naive_last": "Naive (last day)",
    "snaive": "Seasonal naive (last week)",
    "snaive_mean4": "Mean of last 4 same weekdays",
    "last_year": "Same weekday last year",
    "random_forest": "Global Random Forest",
    "lightgbm": "Global LightGBM",
    "prophet_plain": "Prophet per store",
    "prophet_promo": "Prophet per store + promo",
}
COLORS = {
    "lightgbm": "#c0392b",
    "last_year": "#2e8b57",
    "snaive_mean4": "#8e6bbf",
    "random_forest": "#d98324",
    "prophet_promo": "#1f77b4",
    "prophet_plain": "#17becf",
    "snaive": "#7f7f7f",
    "naive_last": "#bcbd22",
}


def apply_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titleweight": "bold",
        }
    )


def find_root(start: str | Path | None = None) -> Path:
    """Repository root (the folder containing pyproject.toml), searching upwards."""
    here = Path(start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    return here
