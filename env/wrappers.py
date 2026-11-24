"""Discrete-action version of the environment, used for DQN."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DiscreteActionWrapper(gym.ActionWrapper):
    """Action k in {0..n-1} -> battery set-point linspace(-1, 1, n)[k].

    n must be odd so that "do nothing" (0.0) is one of the actions.
    """

    def __init__(self, env: gym.Env, n_levels: int = 5):
        super().__init__(env)
        if n_levels % 2 == 0 or n_levels < 3:
            raise ValueError("n_levels must be odd and >= 3")
        self.levels = np.linspace(-1.0, 1.0, n_levels, dtype=np.float32)
        self.action_space = spaces.Discrete(n_levels)

    def action(self, act):
        return np.array([self.levels[int(act)]], dtype=np.float32)
