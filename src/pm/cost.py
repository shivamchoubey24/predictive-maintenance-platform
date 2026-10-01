"""Maintenance cost simulation: reactive vs. fixed schedule vs. RUL-triggered predictive."""
import numpy as np
import pandas as pd

from .config import COSTS

SMOOTH = 5


def _per_unit(oof: pd.DataFrame):
    for u, g in oof.sort_values(["unit", "cycle"]).groupby("unit"):
        sm = g["pred_rul"].rolling(SMOOTH, min_periods=1).median().to_numpy()
        yield u, g["cycle"].to_numpy(), sm


def simulate_predictive(oof, trigger, costs=COSTS):
    """Maintain at the first cycle where smoothed predicted RUL <= trigger.
    If that never happens before failure, the engine fails in service."""
    total, unplanned, wasted = 0.0, 0, []
    for _, cycles, sm in _per_unit(oof):
        life = int(cycles.max())
        hit = np.where(sm <= trigger)[0]
        c = int(cycles[hit[0]]) if len(hit) else None
        if c is None or c >= life:
            total += costs["unplanned"]; unplanned += 1
        else:
            total += costs["planned"] + costs["wasted_per_cycle"] * (life - c)
            wasted.append(life - c)
    n = oof["unit"].nunique()
    return dict(total=total, per_engine=total / n, unplanned_failures=unplanned,
                avg_wasted_cycles=float(np.mean(wasted)) if wasted else 0.0)


def simulate_reactive(oof, costs=COSTS):
    n = oof["unit"].nunique()
    return dict(total=n * costs["unplanned"], per_engine=float(costs["unplanned"]),
                unplanned_failures=n, avg_wasted_cycles=0.0)


def simulate_fixed(oof, interval, costs=COSTS):
    total, unplanned, wasted = 0.0, 0, []
    for _, g in oof.groupby("unit"):
        life = int(g["cycle"].max())
        if interval >= life:
            total += costs["unplanned"]; unplanned += 1
        else:
            total += costs["planned"] + costs["wasted_per_cycle"] * (life - interval)
            wasted.append(life - interval)
    n = oof["unit"].nunique()
    return dict(total=total, per_engine=total / n, unplanned_failures=unplanned,
                avg_wasted_cycles=float(np.mean(wasted)) if wasted else 0.0)


def optimise(oof, costs=COSTS):
    """Best trigger / best fixed interval by grid search, plus the comparison table."""
    trig = {t: simulate_predictive(oof, t, costs) for t in range(5, 101, 5)}
    best_t = min(trig, key=lambda t: trig[t]["total"])
    fixed = {i: simulate_fixed(oof, i, costs) for i in range(100, 260, 5)}
    best_i = min(fixed, key=lambda i: fixed[i]["total"])
    reactive = simulate_reactive(oof, costs)
    pred = trig[best_t]
    return dict(
        reactive=reactive, best_fixed_interval=best_i, fixed=fixed[best_i],
        best_trigger=best_t, predictive=pred,
        savings_vs_reactive_pct=100 * (1 - pred["total"] / reactive["total"]),
        savings_vs_fixed_pct=100 * (1 - pred["total"] / fixed[best_i]["total"]),
        trigger_curve=[[t, trig[t]["per_engine"]] for t in trig])
