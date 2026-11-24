"""Make a trained Stable-Baselines3 model look like a baseline policy, so `run_policy` can evaluate it."""
from __future__ import annotations

import numpy as np


class SB3Policy:
    """Deterministic policy from an SB3 model.

    For DQN the model outputs a discrete index; `levels` (from DiscreteActionWrapper) maps it to the
    battery set-point in [-1, 1], so the evaluation environment stays the plain continuous one.
    """

    def __init__(self, model, name: str, levels: np.ndarray | None = None):
        self.model, self.name, self.levels = model, name, levels

    def act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        if self.levels is not None:
            return np.array([self.levels[int(action)]], dtype=np.float32)
        return np.asarray(action, dtype=np.float32).reshape(1)
