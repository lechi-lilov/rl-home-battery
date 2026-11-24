"""Baseline 2: classic PV self-consumption.

Store PV surplus, cover the home's deficit from the battery, never charge from the grid.
This is how a typical hybrid inverter behaves. It reacts to the CURRENT hour's load and PV,
information the RL agent does not have, so it is a deliberately strong baseline.
"""
import numpy as np


class SelfConsumption:
    name = "self_consumption"

    def act(self, obs, env):
        h, t = env.house, env.t
        net = h.load[t] - h.pv[t]  # > 0: deficit (discharge), < 0: PV surplus (charge)
        return np.array([np.clip(-net / env.cfg.p_max_kw, -1.0, 1.0)], dtype=np.float32)
