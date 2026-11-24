"""Baseline 4: optimal oracle. Linear program with perfect foresight of load, PV and price.

It uses exactly the same physics as the environment, so its cost is a true lower bound on what
any policy can reach in that environment. Replaying its plan through the environment gives
the same cost as the LP objective (checked in tests/test_baselines.py).

Variables per hour t (T hours):
  pc[t]   AC power drawn by the battery (>= 0)       pd[t]  AC power delivered by it (>= 0)
  g[t]    grid import (>= 0)                          e[t]   grid export (>= 0)
  soc[t]  stored energy, T+1 values, soc[0] is fixed
Constraints:
  soc[t+1] = soc[t] + eta_c * pc[t] - pd[t] / eta_d              (battery dynamics)
  g[t] - e[t] = load[t] - pv[t] + pc[t] - pd[t]                  (energy balance)
Objective:
  sum price*g - export_price*e + degradation*(pc+pd) - v * soc[T]   (v = value of leftover energy)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

EPS = 1e-4  # won per kWh of battery throughput: breaks ties so the plan never charges and discharges together


def solve_lp(load, pv, price, cfg, soc0: float, terminal_price: float) -> dict:
    T = len(load)
    I = sp.identity(T, format="csr")
    Z = sp.csr_matrix((T, T))
    dsoc = sp.diags([-np.ones(T), np.ones(T)], [0, 1], shape=(T, T + 1), format="csr")  # soc[t+1] - soc[t]
    # variable order: pc, pd, g, e, soc
    dynamics = sp.hstack([-cfg.eta_charge * I, I / cfg.eta_discharge, Z, Z, dsoc])
    balance = sp.hstack([-I, I, I, -I, sp.csr_matrix((T, T + 1))])
    A = sp.vstack([dynamics, balance], format="csr")
    b = np.concatenate([np.zeros(T), np.asarray(load) - np.asarray(pv)])

    cost = np.zeros(5 * T + 1)
    cost[:T] = cost[T:2 * T] = cfg.degradation_cost + EPS
    cost[2 * T:3 * T] = price
    cost[3 * T:4 * T] = -cfg.export_price
    cost[-1] = -terminal_price                       # reward for leftover energy at the end
    lo, hi = cfg.soc_min_frac * cfg.capacity_kwh, cfg.soc_max_frac * cfg.capacity_kwh
    bounds = ([(0, cfg.p_max_kw)] * (2 * T) + [(0, None)] * (2 * T) + [(lo, hi)] * (T + 1))
    bounds[4 * T] = (soc0, soc0)                     # soc[0] is given

    res = linprog(cost, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    objective = res.fun + terminal_price * soc0      # == money + degradation - terminal credit (EPS aside)
    return dict(p_batt=x[:T] - x[T:2 * T], soc=x[4 * T:], objective=float(objective))


class LPOracle:
    name = "lp_oracle"

    def reset(self, env):
        h, s, T = env.house, env.t, env.horizon
        v = env.terminal_price if env.cfg.terminal_value else 0.0
        plan = solve_lp(h.load[s:s + T], h.pv[s:s + T], h.price[s:s + T], env.cfg, env.soc, v)
        self.start, self.p_batt, self.objective = s, plan["p_batt"], plan["objective"]

    def act(self, obs, env):
        p = self.p_batt[env.t - self.start] / env.cfg.p_max_kw
        return np.array([np.clip(p, -1.0, 1.0)], dtype=np.float32)
