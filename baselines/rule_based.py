"""Baseline 3: time-of-use rule. Charge in the cheapest hours, discharge in the most expensive.

The cheap / expensive hours of a day are read from the tariff schedule (known in advance).
Load and PV are only used from the PREVIOUS hour, so the rule has exactly the information the RL agent has.
The four rule parameters are tuned on the VALIDATION split, never on the test split.
"""
from __future__ import annotations

import itertools

import numpy as np

from env.evaluation import run_policy

GRID = dict(
    soc_target=[0.0, 0.15, 0.3, 0.5, 0.7, 1.0],   # stop grid charging above this usable-SoC fraction (0 = never)
    charge_frac=[0.1, 0.25, 0.5, 1.0],            # charge power as a fraction of p_max
    discharge_frac=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0],  # max discharge power as a fraction of p_max
    mid_selfcons=[False, True],           # outside cheap/expensive hours: causal self-consumption (baseline 2)
)


class RuleBased:
    name = "rule_based"

    def __init__(self, soc_target=0.7, charge_frac=0.5, discharge_frac=0.5, mid_selfcons=True):
        self.params = dict(soc_target=soc_target, charge_frac=charge_frac,
                           discharge_frac=discharge_frac, mid_selfcons=mid_selfcons)

    def act(self, obs, env):
        p, h, t = self.params, env.house, env.t
        prev = max(t - 1, 0)                                  # last fully observed hour
        day_start = t - int(h.hour[t])                       # index of today's 00:00
        day_prices = h.price[day_start:day_start + 24]       # tariff schedule of today
        price = h.price[t]
        soc_frac = float(obs[5])                              # usable-SoC fraction, observation entry 5
        if price == day_prices.min() and price < day_prices.max():       # cheapest hours
            a = p["charge_frac"] if soc_frac < p["soc_target"] else 0.0
        elif price == day_prices.max() and price > day_prices.min():     # most expensive hours
            # discharge at most what the home consumed in the previous hour: energy pushed into the
            # grid earns nothing, so discharging beyond the load would be wasted
            a = -min(p["discharge_frac"], h.load[prev] / env.cfg.p_max_kw)
        elif p["mid_selfcons"]:
            a = float(np.clip(-(h.load[prev] - h.pv[prev]) / env.cfg.p_max_kw, -1.0, 1.0))
        else:
            a = 0.0
        return np.array([a], dtype=np.float32)


def tune(envs: dict) -> tuple[dict, float]:
    """Grid search on the given (validation) environments. Returns (best params, mean cost)."""
    best, best_cost = None, np.inf
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        policy = RuleBased(**params)
        cost = np.mean([run_policy(env, policy)["total_cost"] for env in envs.values()])
        if cost < best_cost:
            best, best_cost = params, float(cost)
    return best, best_cost
