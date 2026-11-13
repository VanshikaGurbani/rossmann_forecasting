"""Loading the Kaggle Rossmann files into a store x date panel."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_COLS = {
    "Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo",
    "StateHoliday", "SchoolHoliday",
}  # fmt: skip
STORE_COLS = {
    "Store", "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
}  # fmt: skip

KAGGLE_HELP = """\
Rossmann data not found in {data_dir}.

The Kaggle competition data cannot be downloaded without accepting its rules, so do this once:
  1. Sign in at https://www.kaggle.com and accept the rules at
     https://www.kaggle.com/competitions/rossmann-store-sales/rules
  2. Either download train.csv and store.csv from the Data tab into {data_dir}/
     or, with the kaggle CLI configured:
        kaggle competitions download -c rossmann-store-sales -p {data_dir}
        (then unzip into {data_dir}/)
"""


@dataclass
class Panel:
    """Dense date x store matrices. `sales` is NaN on closed days and on zero-sales days."""

    dates: pd.DatetimeIndex
    stores: np.ndarray
    sales: pd.DataFrame
    open: pd.DataFrame
    promo: pd.DataFrame
    school_holiday: pd.DataFrame
    state_holiday: pd.DataFrame  # integer codes: 0 none, 1 public, 2 easter, 3 christmas
    static: pd.DataFrame  # one row per store, index = store id


def check_data(data_dir: str | Path) -> tuple[bool, str]:
    data_dir = Path(data_dir)
    missing = [f for f in ("train.csv", "store.csv") if not (data_dir / f).exists()]
    if missing:
        return False, KAGGLE_HELP.format(data_dir=data_dir)
    train_head = pd.read_csv(data_dir / "train.csv", nrows=5)
    store_head = pd.read_csv(data_dir / "store.csv", nrows=5)
    problems = []
    if not TRAIN_COLS <= set(train_head.columns):
        problems.append(f"train.csv is missing {sorted(TRAIN_COLS - set(train_head.columns))}")
    if not STORE_COLS <= set(store_head.columns):
        problems.append(f"store.csv is missing {sorted(STORE_COLS - set(store_head.columns))}")
    if problems:
        return False, "; ".join(problems)
    return True, f"OK: found train.csv and store.csv in {data_dir}"


def build_panel(train: pd.DataFrame, store: pd.DataFrame) -> Panel:
    train = train.copy()
    train["Date"] = pd.to_datetime(train["Date"])
    holiday_codes = {"0": 0, "a": 1, "b": 2, "c": 3}
    train["state_code"] = train["StateHoliday"].astype(str).map(holiday_codes).fillna(0).astype(int)
    train["_sales"] = train["Sales"].where((train["Open"] == 1) & (train["Sales"] > 0))

    def pivot(col: str) -> pd.DataFrame:
        wide = train.pivot(index="Date", columns="Store", values=col)
        full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
        return wide.reindex(full_idx).sort_index(axis=1)

    sales_w = pivot("_sales").astype("float64")
    open_w = pivot("Open").fillna(0).astype(bool)
    promo_w = pivot("Promo").fillna(0).astype(bool)
    school_w = pivot("SchoolHoliday").fillna(0).astype(bool)
    state_w = pivot("state_code").fillna(0).astype(int)
    stores = sales_w.columns.to_numpy()
    static = store.set_index("Store").reindex(stores)
    return Panel(
        dates=sales_w.index,
        stores=stores,
        sales=sales_w,
        open=open_w,
        promo=promo_w,
        school_holiday=school_w,
        state_holiday=state_w,
        static=static,
    )


def load_panel(data_dir: str | Path = "data") -> Panel:
    ok, msg = check_data(data_dir)
    if not ok:
        raise FileNotFoundError(msg)
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"StateHoliday": str})
    store = pd.read_csv(data_dir / "store.csv")
    return build_panel(train, store)
