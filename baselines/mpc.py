"""Baseline 5: model predictive control (MPC) with a simple data-driven forecast.

Every midnight the controller forecasts load and PV for the coming hours as the AVERAGE of the same
clock hour over the previous `n_days` days (n_days = 1 is the classic persistence forecast
"tomorrow looks like yesterday"). With this forecast and the known tariff it solves the same LP as the
oracle (baselines/lp_optimal.py), applies the first 24 hours of the plan, and re-plans the next midnight.
The horizon and n_days are chosen on the validation split only.

It only uses information from before the decision time, exactly like the RL agent, which makes it a
FAIR benchmark. Daily re-planning with a persistence forecast is the scheme of Hussain et al.,
arXiv:2609.14776; the multi-day average is our addition to give MPC a less noisy forecast.

The battery's state of charge depends only on the battery power, not on the load, so the planned SoC
trajectory is always feasible and the plan is executed as is. Forecast errors only change how much
power is bought from or exported to the grid.
"""
from __future__ import annotations

import itertools

import numpy as np

from baselines.lp_optimal import solve_lp

REPLAN_EVERY = 24  # hours
HORIZONS = (24, 48)
N_DAYS = (1, 7, 14)


def forecast_indices(t: int, horizon: int, n_days: int) -> np.ndarray:
    """Data indices averaged to forecast hours t .. t+horizon-1, shape (n_days, horizon).

    Row m-1 holds the same clock hour m days before the forecast day of that hour (day k//24 after t),
    so every index is strictly < t.
    """
    k = np.arange(horizon)
    m = np.arange(1, n_days + 1)[:, None]
    return t + k[None, :] - 24 * (k[None, :] // 24 + m)


def persistence_indices(t: int, horizon: int) -> np.ndarray:
    """The n_days = 1 case: the same clock hour one forecast-day earlier."""
    return forecast_indices(t, horizon, 1)[0]


class MPCForecast:
    name = "mpc"

    def __init__(self, horizon_hours: int = 24, n_days: int = 1):
        if horizon_hours % 24 != 0 or horizon_hours < 24 or n_days < 1:
            raise ValueError("horizon_hours must be a positive multiple of 24 and n_days >= 1")
        self.horizon, self.n_days = horizon_hours, n_days

    def reset(self, env):
        self.plan = None
        self.plan_start = None

    def _replan(self, env):
        h, t = env.house, env.t
        H = min(self.horizon, env.horizon - env.steps)
        idx = np.clip(forecast_indices(t, H, self.n_days), 0, None)   # clip only matters before the data start
        load_fc, pv_fc = h.load[idx].mean(axis=0), h.pv[idx].mean(axis=0)
        v = env.terminal_price if env.cfg.terminal_value else 0.0
        plan = solve_lp(load_fc, pv_fc, h.price[t:t + H], env.cfg, env.soc, v)
        self.plan, self.plan_start = plan["p_batt"], t

    def act(self, obs, env):
        if self.plan is None or env.t - self.plan_start >= REPLAN_EVERY:
            self._replan(env)
        p = self.plan[env.t - self.plan_start] / env.cfg.p_max_kw
        return np.array([np.clip(p, -1.0, 1.0)], dtype=np.float32)


class MPCPersistence(MPCForecast):
    """Literature variant (n_days = 1), reported separately."""
    name = "mpc_persistence"

    def __init__(self, horizon_hours: int = 24):
        super().__init__(horizon_hours, n_days=1)


def tune(envs: dict, run_policy) -> tuple[dict, dict]:
    """Grid search over (horizon, n_days) on the given (validation) environments."""
    costs = {}
    for H, n in itertools.product(HORIZONS, N_DAYS):
        costs[(H, n)] = float(np.mean([run_policy(env, MPCForecast(H, n))["total_cost"] for env in envs.values()]))
    H, n = min(costs, key=costs.get)
    return dict(horizon_hours=H, n_days=n), {f"H{k[0]}_n{k[1]}": v for k, v in costs.items()}
