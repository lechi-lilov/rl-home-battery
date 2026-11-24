"""Home battery + rooftop PV under a time-of-use tariff, as a Gymnasium environment.

One step = one hour. At the start of hour t the agent picks a battery power set-point
a in [-1, 1] (positive = charge, negative = discharge, scaled by p_max). The hour's load and PV
are then revealed, the grid import/export follows from energy balance, and the reward is the
negative electricity cost.

What the agent can observe (nothing from the future):
  calendar (hour, day of year, weekend), battery SoC, tariff price now and for the next 5 hours
  (the tariff is a known schedule), and the load/PV of the PREVIOUS hour. Optionally a
  persistence forecast: load/PV of the same hours one day earlier.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from env.data import PRICE_LOOKAHEAD, HouseData
from env.tariff import Tariff

DT = 1.0  # hours per step
FORECAST_HOURS = 3  # persistence forecast covers hours t, t+1, t+2
LOOKBACK_HOURS = 24  # the oldest data an observation can contain is from t - 24


@dataclass
class EnvConfig:
    capacity_kwh: float = 10.0
    p_max_kw: float = 5.0
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.90
    episode_days: int = 7
    export_price: float = 0.0
    degradation_cost: float = 0.0
    reward_scale: float = 100.0
    reward_mode: str = "cost"
    terminal_value: bool = True
    use_forecast: bool = False
    load_scale_kw: float = 5.0
    pv_scale_kw: float = 5.0

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/env.yaml", **overrides) -> "EnvConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        flat = {**raw.pop("battery"), **raw}
        flat.update(overrides)
        return cls(**flat)

    def __post_init__(self) -> None:
        if self.reward_mode not in ("cost", "savings"):
            raise ValueError("reward_mode must be 'cost' or 'savings'")


class EnergyEnv(gym.Env):
    """mode="train": random house and start day, `episode_days` long, random initial SoC; the episode
    always lies inside one contiguous segment of the split, so it never touches validation days.
    mode="eval": one segment of one house (reset option "segment"), SoC at 50 % of the usable range.
    `env.evaluation.run_policy` loops over all segments."""

    metadata = {"render_modes": []}

    def __init__(self, houses: dict[int, HouseData], tariff: Tariff, config: EnvConfig | None = None,
                 mode: str = "train", house_id: int | None = None):
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")
        self.cfg = config or EnvConfig()
        self.tariff = tariff
        self.houses = houses
        self.house_ids = sorted(houses)
        self._start_cache: dict[tuple[int, int], np.ndarray] = {}
        self.mode = mode
        self.fixed_house = house_id
        if mode == "eval" and house_id is None and len(houses) > 1:
            raise ValueError("eval mode needs house_id when several houses are loaded")

        c = self.cfg
        self.soc_lo = c.soc_min_frac * c.capacity_kwh
        self.soc_hi = c.soc_max_frac * c.capacity_kwh
        # value of one kWh left in the battery at the end of an episode (energy comes back with eta_discharge)
        self.terminal_price = tariff.mean_price * c.eta_discharge
        self.max_price = tariff.max_price

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        n_fc = 2 * FORECAST_HOURS if c.use_forecast else 0
        # [sin h, cos h, sin doy, cos doy, weekend, soc, price x6, prev load, prev pv, (forecast)]
        low = np.array([-1, -1, -1, -1, 0, 0] + [0] * PRICE_LOOKAHEAD + [0, 0] + [0] * n_fc, dtype=np.float32)
        # load/pv are scaled by ~5 kW, real values stay below ~3; 100 is a loose finite upper bound
        high = np.array([1, 1, 1, 1, 1, 1] + [1] * PRICE_LOOKAHEAD + [100.0, 100.0] + [100.0] * n_fc,
                        dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random: same seed -> same episode
        options = options or {}
        hid = options.get("house_id", self.fixed_house)
        if hid is None:
            hid = self.house_ids[int(self.np_random.integers(len(self.house_ids)))]
        self.house = self.houses[hid]
        h = self.house

        if self.mode == "eval":
            # one evaluation episode = one segment of the split (the test year is a single segment,
            # validation has one segment per month)
            seg_start, seg_end = h.segments[options.get("segment", 0)]
            start = options.get("start", seg_start)
            self.horizon = options.get("horizon", seg_end - start)
        else:
            self.horizon = options.get("horizon", self.cfg.episode_days * 24)
            starts = self._train_starts(h, self.horizon)
            start = options.get("start", int(starts[self.np_random.integers(len(starts))]))
        if not any(s <= start and start + self.horizon <= e for s, e in h.segments):
            raise ValueError("episode must lie inside one segment of the split")

        frac = options.get("init_soc_frac")
        if frac is None:
            frac = float(self.np_random.uniform()) if self.mode == "train" else 0.5
        self.soc = self.soc_lo + frac * (self.soc_hi - self.soc_lo)
        self.soc_start = self.soc
        self.t = start
        self.steps = 0
        self.cum_cost = 0.0          # money actually paid to the grid (won)
        self.cum_nothing = 0.0       # what the same home would have paid without a battery
        self.cum_degradation = 0.0
        return self._obs(), self._info(None)

    def step(self, action):
        c, h, t = self.cfg, self.house, self.t
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        p_req = a * c.p_max_kw

        # --- battery physics: clip the request so SoC stays inside [soc_lo, soc_hi]
        if p_req >= 0:  # charge: p kW from the AC side puts eta*p kW into the cells
            p_ac = min(p_req, (self.soc_hi - self.soc) / (c.eta_charge * DT))
            self.soc += c.eta_charge * p_ac * DT
            p_batt = p_ac            # AC-side power drawn by the battery (> 0)
        else:           # discharge: the cells give up p/eta to deliver p kW on the AC side
            p_ac = min(-p_req, (self.soc - self.soc_lo) * c.eta_discharge / DT)
            self.soc -= p_ac / c.eta_discharge * DT
            p_batt = -p_ac           # AC-side power delivered by the battery (< 0)

        # --- energy balance for this hour (load and PV are revealed only now)
        net = h.load[t] - h.pv[t] + p_batt          # > 0: import from grid, < 0: export
        grid_import = max(net, 0.0) * DT
        grid_export = max(-net, 0.0) * DT
        price = h.price[t]
        cost = price * grid_import - c.export_price * grid_export
        net0 = h.load[t] - h.pv[t]                  # same hour without a battery
        cost_nothing = price * max(net0, 0.0) * DT - c.export_price * max(-net0, 0.0) * DT
        degradation = c.degradation_cost * abs(p_batt) * DT

        money = -cost if c.reward_mode == "cost" else cost_nothing - cost
        reward = (money - degradation) / c.reward_scale

        self.cum_cost += cost
        self.cum_nothing += cost_nothing
        self.cum_degradation += degradation
        self.t += 1
        self.steps += 1

        truncated = self.steps >= self.horizon       # time limit, the problem itself never ends
        terminal_credit = 0.0
        if truncated and c.terminal_value:
            terminal_credit = self.terminal_price * (self.soc - self.soc_start)
            reward += terminal_credit / c.reward_scale
        info = self._info(dict(p_batt_kw=p_batt, grid_import_kwh=grid_import, grid_export_kwh=grid_export,
                               price=price, cost_won=cost, terminal_credit_won=terminal_credit,
                               index=t))
        return self._obs(), float(reward), False, truncated, info

    # --------------------------------------------------------------- helpers
    def _train_starts(self, h: HouseData, horizon: int) -> np.ndarray:
        """All midnights from which an episode of `horizon` hours fits inside one segment (cached).

        Episodes start at least LOOKBACK_HOURS after the segment start, because observations look back
        up to 24 h. Otherwise the first observations of an episode could contain data from the days
        just before the segment, i.e. validation days.
        """
        key = (h.house_id, horizon)
        if key not in self._start_cache:
            starts = [np.arange(s + LOOKBACK_HOURS, e - horizon + 1, 24) for s, e in h.segments]
            starts = np.concatenate(starts) if starts else np.array([], dtype=int)
            if len(starts) == 0:
                raise ValueError("episode is longer than every segment of the split")
            self._start_cache[key] = starts
        return self._start_cache[key]

    def _obs(self) -> np.ndarray:
        c, h, t = self.cfg, self.house, self.t
        ang_h = 2 * np.pi * h.hour[t] / 24
        ang_d = 2 * np.pi * (h.doy[t] - 1) / 365.25
        soc_frac = (self.soc - self.soc_lo) / (self.soc_hi - self.soc_lo)
        prev = max(t - 1, 0)  # the hour that has just been fully observed
        parts = [np.sin(ang_h), np.cos(ang_h), np.sin(ang_d), np.cos(ang_d), h.weekend[t],
                 np.clip(soc_frac, 0.0, 1.0)]
        parts += list(h.price[t:t + PRICE_LOOKAHEAD] / self.max_price)
        parts += [h.load[prev] / c.load_scale_kw, h.pv[prev] / c.pv_scale_kw]
        if c.use_forecast:  # persistence: the same hours one day earlier (all in the past)
            idx = np.clip(t - 24 + np.arange(FORECAST_HOURS), 0, None)
            parts += list(h.load[idx] / c.load_scale_kw) + list(h.pv[idx] / c.pv_scale_kw)
        return np.asarray(parts, dtype=np.float32)

    def _info(self, step_info: dict | None) -> dict:
        info = dict(house_id=self.house.house_id, soc_kwh=self.soc, cum_cost_won=self.cum_cost,
                    cum_nothing_won=self.cum_nothing, cum_degradation_won=self.cum_degradation)
        if step_info:
            info.update(step_info)
        return info
