# Upgrade plan and audit findings

Written before any code changes, from the repo at commit `e3bece5` ("Clean start").
Everything below was checked by re-running the original notebook logic on the
`data/train.csv` + `data/store.csv` that ship in the repo (1,017,209 rows,
1,115 stores, 2013-01-01 to 2015-07-31).

## 0. Verdict on the headline number

README / GitHub description: **"Random Forest RMSE 357 vs Prophet ~780, a ~50% reduction"**.

The two numbers are real in the sense that the notebooks print them and I can
reproduce them (Prophet 780.71, Random Forest 357 to 371 depending on exact
dummy columns). But the claim built on them does **not** hold up:

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Only one store was modelled.** Every model notebook filters `Store == 1`. The README says the project "forecasts daily sales for 1,115 stores". | `store_id = 1` in notebooks 02-05; train shape `(754, 27)`, test `(27, 27)` |
| 2 | **The test set is 27 rows** (July 2015, store 1 only). One RMSE on 27 points has no meaningful uncertainty and says nothing about the chain. | notebook 03 output |
| 3 | **Prophet was scored on the wrong dates.** `make_future_dataframe(periods=27)` yields 27 *calendar* days (Jul 1-27) but the 27 test rows are *open* days (Jul 1-31, Sundays missing). The first 27 forecasts are compared positionally to 27 actuals that sit on different dates. | forecast dates 2015-07-01..07-27 vs test dates 2015-07-01..07-31. With dates aligned Prophet's RMSE is 734, not 781. |
| 4 | **The comparison is not like for like.** Random Forest receives `Promo` (a planned, known-in-advance flag) and Prophet does not. Adding the same flag to Prophet as a regressor takes it from 734 to **411** on the same 27 rows, so most of the "50% reduction" is *information given to one model and withheld from the other*, not model quality. | Prophet + `Promo` regressor, date-aligned: RMSE 410.7 |
| 5 | **The lag features do almost nothing**, and are also not available in a multi-step forecast. A RF trained *without any lag or rolling features* scores 364 vs 371 with them. Meanwhile `Sales_Lag_7` / rolling means for test rows after the 7th use *realised July sales*, which would not exist when forecasting July from June 30. The RF score is therefore a mix of one-step-ahead and multi-step; Prophet is a pure 27-day-ahead forecast. | 20 of 27 test rows have a lag-7 feature that falls inside the test window |
| 6 | **SARIMAX (1285) is mis-specified, not a real baseline.** It is fitted to a series with closed days dropped (so the "weekly" period `s=7` is really 6 or 7 depending on the week) and its output is index-aligned positionally. A trivial same-weekday-last-week baseline scores about the same (1285). | seasonal-naive RMSE 1284.7 |
| 7 | **No naive baseline was reported.** Neither model was compared to anything a retail analyst would try first. | - |

Conclusion: the 357-vs-780 comparison is a single-store, 27-row, misaligned,
information-asymmetric result. It must not be used as a headline.
The README is rewritten from measured numbers only (see commits).

### Feature-engineering leakage check (the thing the task asked to verify)

* Lags: `groupby('Store')['Sales'].shift(k)` is per-store and only looks back.
  No look-ahead. But it shifts by *rows* after closed days / zero-sales days
  were deleted, so "lag 7" means "7 open days ago", not "7 days ago".
* Rolling means: `groupby('Store')['Sales'].shift(1).rolling(w).mean()` rolls over
  the *whole* concatenated column, not per store. I checked all 844,338 rows
  against a properly grouped version: **0 differences**, because the NaN created
  by the per-store shift propagates through the window, so the code is accidentally
  correct. It is, however, only correct when the frame is sorted by (Store, Date).
  `train.csv` ships in descending date order and `src/feature_engineering.py`
  does not sort, so the `src/` version silently computes *forward-looking*
  features on raw input. The notebooks sort first; the `src/` module does not.
* `Promo_Duration = groupby(Store).Promo.cumsum()` is a monotone counter since
  2013 (a time index in disguise), not "running promo days". Tree models cannot
  extrapolate it, so test rows land beyond the training range. Misnamed, and not
  a promo-duration feature. Removed in the rewrite.
* `Customers` is correctly dropped from the features (it is contemporaneous with
  `Sales` and would leak). It is used as a regressor in the causal notebook, see below.
* The split itself (`Date < 2015-07-01` vs `>=`) is time-based and has no overlap.

### Causal "promo lift" (+0.90%, $39.96) is not a promo-lift estimate

`04_CausalImpact_Analysis` fits `Sales ~ Customers + DayOfWeek + Month` on
Mar-Apr 2015 and reports the average residual on May-Jun 2015 as "promotional lift".

* The model contains **no promo variable** and the "post period" is not a promo
  period: store 1 has Promo=1 on 46.0% of open days in the pre period and 43.8% in
  the post period. There is no treatment contrast.
* `Customers` is a **mediator** (promos work by bringing in customers). Conditioning
  on it removes most of the effect by construction; the result is a forecast residual.
* `Month` is entered as a number with only two distinct training values, then extrapolated.
* One store, no confidence interval. README says "significantly higher"; no test was run.

A naive within-store comparison shows the opposite order of magnitude: store 1 sells
5,300 on promo days vs 4,320 off promo (+22.7%); averaged over all stores the naive
promo/non-promo ratio is +41%. The honest answer needs a design with controls,
not this residual. See plan item 5.

### Other issues

* Notebooks hard-code `/Users/ronitgandhi/...` paths (not runnable by anyone else)
  and write large intermediate CSVs (`cleaned_rossmann.csv`, `feature_engineered_rossmann.csv`) into `data/`.
* `requirements.txt` is unpinned (`>=`), contains `causalimpact` which the code never
  imports, and omits `joblib`. No seeds on Prophet. No tests, no CI, no LICENSE file
  (the README says MIT).
* `outputs/` PNGs come from the single-store models and are regenerated.
* The Kaggle `train.csv` (39 MB) is committed. Kaggle's rules restrict redistribution;
  see "Data" below.

### Authorship and provenance (for the owner, not a code issue)

* `git remote -v` lists `origin = RonitGandhi/rossmann_forecasting` and `mine = VanshikaGurbani/rossmann_forecasting`.
* The single upstream commit `e3bece5` ("Clean start", 2025-04-26) is authored by "RonitaGandhi".
  The first commit of the VanshikaGurbani copy (`2d2413b`, "Initial commit", 2026-10-11) has a
  **byte-identical tree** (`git diff e3bece5 2d2413b` is empty) and no shared history.
* All five notebooks hard-code `/Users/ronitgandhi/...` paths and their notebook metadata carries
  no author field.
* Conclusion: every file present before this branch (notebooks, `src/*.py`, first README, the
  outputs PNGs) was written by Ronit Gandhi, not by the owner of this repository. The README now
  credits that source and does not claim the original analysis. Only the work on this branch is
  new. The owner must confirm the relationship (collaborator, fork she should have forked
  properly, permission to reuse) before a resume bullet describes the project as her own.

## 1. What the rebuilt project does

Task definition, used everywhere: **forecast daily sales of every open store for the next
H = 42 days (6 weeks, the Kaggle horizon) from information available at the forecast origin**.
Planned covariates (promo calendar, holidays, open flag, store attributes) may be used for the
target dates; realised sales only up to the day before the origin.

| Step | Deliverable | Notes |
|------|-------------|-------|
| Reproducibility | installable package `rossmann_forecast`, CLI `rossmann`, Makefile, pinned requirements, fixed seed | Kaggle download is manual (rules acceptance); `rossmann check-data` explains it |
| Validation | rolling-origin expanding window, 3 folds x 42 days (origins fixed before any run: 2014-08-01, 2014-12-01, 2015-06-20) | WAPE, RMSE, MAE, MAPE and Kaggle RMSPE, mean +/- std over folds |
| Baselines | last value, seasonal naive (same weekday), mean of last 4 same weekdays, last year same weekday (t-364) | every model must beat these |
| Models | one **global LightGBM** over all 1,115 stores, global Random Forest, per-store **Prophet** (plain and with promo/holiday regressors) on a fixed 100-store random sample | per-store SARIMAX from the original README is not rerun; the original was mis-specified and Prophet covers the per-store case. All models are also scored on the same sample rows |
| Uncertainty | split conformal on log1p sales, calibrated on the window before each origin, 80% and 90% | empirical coverage per fold, by horizon bucket and per store |
| Promo | model counterfactual (flip the promo flag) and store x weekday + date fixed-effects regression with a permutation placebo | labelled an association, not a validated causal estimate; the old +0.90% is withdrawn |
| Explainability | SHAP on the final-fold LightGBM | |
| Quality | pytest (no-look-ahead perturbation tests, fold leakage test, metrics, splits, conformal), ruff, GitHub Actions on synthetic data | |

Not built: a dashboard, TFT/NeuralProphet-style models, statsforecast, per-store SARIMAX.
Nothing in the README claims them.

### Leakage design

All sales-derived features carry a minimum lag equal to the forecast horizon (`min_lag = 42`),
so a feature for target date `t` only reads sales up to `t - 42`, always before any origin that
could predict `t`. Sales on or after the origin are additionally masked before features are
built. The tests scramble sales after `t - min_lag` and assert the features at `t` are
bit-identical, and scramble the whole forecast window and assert no forecast changes.

### Data

`data/train.csv`, `data/store.csv`, `data/test.csv` were already tracked in git history before this
branch; no new large files were added. Recommended follow-up for the owner: `git rm --cached
data/*.csv` and use `rossmann check-data`. Merging that deletion would remove the local copies, so
copy them first. `data/test.csv` has no `Sales` column (Kaggle hold-out) and is unused.
