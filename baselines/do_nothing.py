"""Baseline 1: the battery is never used (the home behaves as if it had no battery)."""
import numpy as np


class DoNothing:
    name = "do_nothing"

    def act(self, obs, env):
        return np.zeros(1, dtype=np.float32)
