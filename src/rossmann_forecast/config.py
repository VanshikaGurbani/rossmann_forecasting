"""Constants shared by the pipeline. Everything that changes a reported number lives here."""

SEED = 42
HORIZON = 42  # days ahead to forecast, the horizon of the Kaggle competition
MIN_LAG = HORIZON  # sales-derived features may only read data at least this many days old

# First target date of each evaluation fold. Chosen once, before any model was run, to cover
# the Kaggle test season (Aug-Sep), the Christmas peak, and the most recent window of data.
DEFAULT_ORIGINS = ("2014-08-01", "2014-12-01", "2015-06-20")

PROPHET_SAMPLE_STORES = 100
LIGHTGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
CONFORMAL_LEVELS = (0.8, 0.9)
