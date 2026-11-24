"""Run a policy through an evaluation episode and report cost metrics.

A "policy" is any object with `act(obs, env) -> action` and, optionally, `reset(env)`.
Baselines use it directly; trained RL agents are wrapped in a small adapter later.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def run_policy(env, policy, record: bool = False) -> dict:
    """Roll the policy through one full evaluation episode.

    total_cost is the economic cost of the year: money paid to the grid
    + degradation cost - value of the energy left in the battery relative to the start.
    The last term is the same terminal credit the reward uses and the LP oracle optimises,
    so every policy is compared on the same footing.
    """
    obs, info = env.reset(seed=0)
    if hasattr(policy, "reset"):
        policy.reset(env)
    rec = {k: [] for k in ("index", "soc_kwh", "p_batt_kw", "price", "grid_import_kwh")}
    done = False
    while not done:
        action = policy.act(obs, env)
        obs, _, _, done, info = env.step(action)
        if record:
            for k in rec:
                rec[k].append(info[k])
    total = info["cum_cost_won"] + info["cum_degradation_won"] - info["terminal_credit_won"]
    out = dict(house_id=info["house_id"], total_cost=total, nothing_cost=info["cum_nothing_won"],
               saving_pct=100.0 * (info["cum_nothing_won"] - total) / info["cum_nothing_won"])
    if record:
        out["trajectory"] = {k: np.asarray(v) for k, v in rec.items()}
    return out


def mean_ci95(values) -> tuple[float, float]:
    """Mean and half-width of the 95 % Student-t confidence interval."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan")
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(half)
