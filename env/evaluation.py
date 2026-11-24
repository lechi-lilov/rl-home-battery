"""Run a policy through the evaluation episodes of a split and report cost metrics.

A "policy" is any object with `act(obs, env) -> action` and, optionally, `reset(env)`.
Baselines use it directly; trained RL agents are wrapped in a small adapter later.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

RECORD_KEYS = ("index", "soc_kwh", "p_batt_kw", "price", "grid_import_kwh", "grid_export_kwh")


def run_policy(env, policy, record: bool = False) -> dict:
    """Roll the policy through every segment of the env's split (env must be in eval mode).

    Each segment is an independent episode that starts with the battery at 50 %.
    total_cost = money paid to the grid + degradation cost - value of the energy left in the battery
    relative to the segment start, summed over segments. The last term is the same terminal credit the
    reward uses and the LP oracle optimises, so every policy is compared on the same footing.
    """
    if env.mode != "eval":
        raise ValueError("run_policy needs an environment in eval mode")
    house = env.houses[env.fixed_house if env.fixed_house is not None else env.house_ids[0]]
    total = nothing = 0.0
    rec = {k: [] for k in RECORD_KEYS}
    for k in range(len(house.segments)):
        obs, info = env.reset(seed=0, options={"segment": k})
        if hasattr(policy, "reset"):
            policy.reset(env)
        done = False
        while not done:
            obs, _, _, done, info = env.step(policy.act(obs, env))
            if record:
                for key in RECORD_KEYS:
                    rec[key].append(info[key])
        total += info["cum_cost_won"] + info["cum_degradation_won"] - info["terminal_credit_won"]
        nothing += info["cum_nothing_won"]
    out = dict(house_id=house.house_id, total_cost=total, nothing_cost=nothing,
               saving_pct=100.0 * (nothing - total) / nothing, n_segments=len(house.segments))
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
