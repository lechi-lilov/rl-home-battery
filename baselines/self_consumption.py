"""Baseline 2: PV self-consumption. Store PV surplus, cover the home's deficit, never plan grid charging.

Two variants, because they differ in what they know:
* SelfConsumptionCausal uses the PREVIOUS hour's surplus/deficit, i.e. exactly the information the
  RL agent has. This is the fair comparison for the RL agents.
* SelfConsumption (reactive) uses the CURRENT hour's surplus/deficit, like a hybrid inverter that
  measures in real time. It has an information advantage over the RL agent and is reported as a
  stronger reference, not as a fair competitor.
"""
import numpy as np


def _follow(net_kw: float, p_max_kw: float) -> np.ndarray:
    # net > 0: deficit -> discharge; net < 0: PV surplus -> charge
    return np.array([np.clip(-net_kw / p_max_kw, -1.0, 1.0)], dtype=np.float32)


class SelfConsumption:
    name = "self_consumption"

    def act(self, obs, env):
        h, t = env.house, env.t
        return _follow(h.load[t] - h.pv[t], env.cfg.p_max_kw)


class SelfConsumptionCausal:
    name = "self_consumption_causal"

    def act(self, obs, env):
        h, prev = env.house, max(env.t - 1, 0)
        return _follow(h.load[prev] - h.pv[prev], env.cfg.p_max_kw)
