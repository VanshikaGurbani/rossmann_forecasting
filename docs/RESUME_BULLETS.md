# Resume bullets (Rossmann forecasting)

Every number below is read from `reports/results.json` / `docs/RESULTS.md`, produced by
`rossmann run-all` at the commit named in that file. Re-run it and confirm the numbers before
sending anything out.

## Two bullets

* Built a leakage-safe rolling-origin forecasting pipeline in Python for daily sales of all 1,115
  Rossmann stores (3 test windows, 42 days ahead); one global LightGBM model cut weighted error
  (WAPE) to 9.2% versus 18.9% for the best seasonal-naive baseline (51% lower) and beat per-store
  Prophet (12.5% with promo inputs, 17.9% without) on a 100-store sample.
* Added 90% conformal prediction intervals with measured coverage (86.4% overall, 76.7% in the
  Christmas-season window), a 30-test pytest suite that scrambles the forecast window's data to
  prove no future leakage, and ruff plus GitHub Actions CI; reported the promo effect (+39%,
  95% CI 36.6% to 41.9%) as an association, not a causal estimate.

## What she must verify before using these

1. **Authorship (blocking).** Before this branch, every file in the repository (notebooks, `src/`,
   first README, plots) was written by Ronit Gandhi: the upstream commit `e3bece5` is authored by
   "RonitaGandhi", the notebooks hard-code `/Users/ronitgandhi/...`, and the `mine` repository's
   first commit has a byte-identical tree. Only the work on branch
   `claude/rigorous-forecasting-ec76af` is new. Confirm the relationship and permission to reuse,
   and decide whether the README credit is acceptable. Do not describe the original analysis as
   hers.
2. **Reproduce the numbers.** Run `rossmann check-data` then `rossmann run-all` on her own
   machine. Two full runs here gave identical metrics. Timings will differ.
3. **CI status.** The branch has not been pushed. Push it (her call), open the Actions tab and
   confirm the run is green before mentioning CI.
4. **Be able to explain the terms:** WAPE (sum of absolute errors over sum of sales), rolling-origin
   cross-validation, why sales features are lagged by at least the horizon, what split conformal
   prediction does and why coverage falls short in the Christmas fold.
5. **Scope of the Prophet claim.** Prophet ran on a 100-store random sample (seed 42), not all
   1,115 stores. The bullet says so; keep it.
6. **Do not say causal.** Promos follow one chain-wide calendar, so the +39% is an association.
7. **Do not quote the old headline.** "Random Forest RMSE 357 vs Prophet 780" came from one store
   and 27 test rows with misaligned Prophet dates and is withdrawn (`docs/UPGRADE_PLAN.md`). The
   GitHub repository description still contains it and should be updated.
8. **Data and licence.** `data/*.csv` (Kaggle files) are still tracked from before this work, and
   the repository has no `LICENSE`. Check the Kaggle competition rules on redistribution.
9. **Fold choice.** The three origins were fixed before running any model, but they are a small
   sample (three windows); say "3 rolling-origin windows", not "extensively validated".
10. Python 3.13.5 was used for every pin; 3.14 is not supported.
