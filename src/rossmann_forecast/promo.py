"""Promotion effect estimates. These are associations under stated assumptions, not validated
causal effects: promo timing is chosen by the retailer and nothing here is randomised.

A fact that shapes everything below, found by inspecting the data: Rossmann ran ONE promo
calendar for the whole chain (alternate full Monday-to-Friday weeks). Every store has the same
promo flag on a given date, so a date fixed effect absorbs the promo completely and "treated
versus control stores on the same day" does not exist. The promo effect is identified only from
week-to-week alternation within a store, so it cannot be separated from anything else that
differs between promo and non-promo weeks.

1. model_counterfactual: flip the promo flag in the fitted global model's forecast window.
2. fixed_effects_lift: log(sales) on promo with store x weekday and year-month fixed effects and
   holiday controls, standard errors clustered by calendar week (the level at which the promo
   varies), plus a randomisation test that permutes promo weeks within a month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED
from .data import Panel
from .models import FittedModel


def model_counterfactual(model: FittedModel, rows: pd.DataFrame) -> dict[str, float]:
    """Predicted sales with promo=1 versus promo=0 on the promoted days of a forecast window.

    Only the `promo` column is flipped; features derived from neighbouring days are held fixed.
    No interval is given: every store shares the promo calendar, so a window holds only a few
    independent promo weeks and a store-level bootstrap would badly understate the uncertainty.
    """
    promoted = rows[rows["promo"] == 1]
    on = promoted.copy()
    off = promoted.copy()
    on["promo"] = 1.0
    off["promo"] = 0.0
    p_on = model.predict(on)
    p_off = model.predict(off)
    weeks = promoted["date"].dt.isocalendar()
    return {
        "lift_pct": float((p_on.sum() / p_off.sum() - 1) * 100),
        "mean_extra_sales_per_promo_day": float((p_on - p_off).mean()),
        "n_promo_days": int(len(promoted)),
        "n_promo_weeks": int(weeks.groupby(["year", "week"]).ngroups),
    }


def _demean(values: np.ndarray, groups: list[np.ndarray], n_groups: list[int], iters: int = 100):
    """Alternating projections: remove every group mean until nothing changes."""
    out = np.array(values, dtype=float, copy=True)
    for _ in range(iters):
        before = out.copy()
        for g, k in zip(groups, n_groups, strict=True):
            sums = np.bincount(g, weights=out, minlength=k)
            counts = np.bincount(g, minlength=k)
            out -= (sums / np.maximum(counts, 1))[g]
        if np.max(np.abs(out - before)) < 1e-10:
            break
    return out


def _week_permutation(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Map each date to the date whose promo flag it receives after permuting full weeks
    among the weeks of the same calendar month (week assigned to the month of its Monday)."""
    mapping = np.arange(len(dates))
    iso = dates.isocalendar()
    week_id = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    by_week: dict[int, list[int]] = {}
    for pos, wid in enumerate(week_id):
        by_week.setdefault(int(wid), []).append(pos)
    by_month: dict[tuple[int, int], list[int]] = {}
    for wid, positions in by_week.items():
        if len(positions) != 7:  # skip partial weeks at the edges of the sample
            continue
        monday = dates[positions[0]]
        by_month.setdefault((monday.year, monday.month), []).append(wid)
    for weeks in by_month.values():
        shuffled = rng.permutation(weeks)
        for src, dst in zip(weeks, shuffled, strict=True):
            mapping[by_week[src]] = by_week[int(dst)]
    return mapping


def fixed_effects_lift(panel: Panel, n_perm: int = 100, seed: int = SEED) -> dict[str, float | int]:
    """log(sales) ~ promo + school/state holiday + store x weekday FE + year-month FE.

    Lift = exp(beta) - 1. Standard errors are clustered by ISO calendar week.
    """
    sales = panel.sales
    dates = sales.index
    mask = sales.notna().to_numpy()
    d_idx, s_idx = np.nonzero(mask)
    n_stores = sales.shape[1]
    y = np.log(sales.to_numpy()[d_idx, s_idx])
    promo = panel.promo.to_numpy()
    x_promo = promo[d_idx, s_idx].astype(float)
    controls = np.column_stack(
        [
            (panel.state_holiday.to_numpy()[d_idx, s_idx] > 0).astype(float),
            panel.school_holiday.to_numpy()[d_idx, s_idx].astype(float),
        ]
    )
    month_id = ((dates.year - dates.year.min()) * 12 + dates.month - 1).to_numpy()
    groups = [s_idx * 7 + dates.dayofweek.to_numpy()[d_idx], month_id[d_idx]]
    n_groups = [n_stores * 7, int(month_id.max()) + 1]
    iso = dates.isocalendar()
    week_codes = pd.factorize(iso["year"].astype(int) * 100 + iso["week"].astype(int))[0]
    cluster = week_codes[d_idx]
    n_clusters = int(cluster.max()) + 1

    y_t = _demean(y, groups, n_groups)
    ctrl_t = np.column_stack([_demean(controls[:, j], groups, n_groups) for j in range(2)])
    ctrl_t = ctrl_t[:, np.sum(ctrl_t**2, axis=0) > 1e-9]  # drop controls with no variation

    def fit(x_t: np.ndarray) -> tuple[float, float]:
        design = np.column_stack([x_t, ctrl_t])
        gram = design.T @ design
        beta = np.linalg.solve(gram, design.T @ y_t)
        resid = y_t - design @ beta
        scores = np.column_stack(
            [
                np.bincount(cluster, weights=design[:, j] * resid, minlength=n_clusters)
                for j in range(design.shape[1])
            ]
        )
        bread = np.linalg.inv(gram)
        meat = scores.T @ scores
        var = bread @ meat @ bread * n_clusters / max(n_clusters - 1, 1)
        return float(beta[0]), float(np.sqrt(var[0, 0]))

    x_t = _demean(x_promo, groups, n_groups)
    beta, se = fit(x_t)
    raw_promo = float(np.mean(np.exp(y[x_promo == 1])) / np.mean(np.exp(y[x_promo == 0])) - 1)

    rng = np.random.default_rng(seed)
    perm_betas = []
    for _ in range(n_perm):
        mapping = _week_permutation(dates, rng)
        x_perm = promo[mapping[d_idx], s_idx].astype(float)
        perm_betas.append(fit(_demean(x_perm, groups, n_groups))[0])
    perm = np.asarray(perm_betas)
    p_value = float((1 + np.sum(np.abs(perm) >= abs(beta))) / (1 + n_perm))
    return {
        "lift_pct": float((np.exp(beta) - 1) * 100),
        "ci95_low_pct": float((np.exp(beta - 1.96 * se) - 1) * 100),
        "ci95_high_pct": float((np.exp(beta + 1.96 * se) - 1) * 100),
        "beta_log": beta,
        "se_cluster_week": se,
        "n_week_clusters": n_clusters,
        "naive_ratio_lift_pct": raw_promo * 100,
        "permutation_beta_sd": float(perm.std(ddof=1)) if n_perm > 1 else float("nan"),
        "permutation_beta_abs_max": float(np.abs(perm).max()) if n_perm else float("nan"),
        "permutation_p_value": p_value,
        "n_permutations": n_perm,
        "n_obs": int(len(y)),
    }
