"""Environment tests. Each test checks one physical or logical property independently of the env code."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from env import DiscreteActionWrapper, EnergyEnv, EnvConfig, Tariff, load_data_config, load_houses

DATA_CFG = load_data_config()
POOL = DATA_CFG["pool_houses"]
HELDOUT = DATA_CFG["heldout_houses"]


@pytest.fixture(scope="module")
def tariff():
    return Tariff.from_yaml("configs/tariff_jeju.yaml")


@pytest.fixture(scope="module")
def train_houses(tariff):
    return load_houses(POOL[:3], "train", tariff)


@pytest.fixture(scope="module")
def test_houses(tariff):
    return load_houses(POOL[:1], "test", tariff)


def make(houses, tariff, mode="train", house_id=None, **cfg):
    return EnergyEnv(houses, tariff, EnvConfig.from_yaml(**cfg), mode=mode, house_id=house_id)


# ----------------------------------------------------------------- tariff
def test_tariff_prices_match_official_table(tariff):
    ts = pd.DatetimeIndex(["2013-01-15 23:00", "2013-01-15 03:00", "2012-10-03 18:00", "2012-04-10 10:00",
                           "2012-07-20 21:59", "2012-07-20 08:00", "2012-07-20 16:00"])
    got = tariff.price_array(ts)
    # winter light, winter light (wraps midnight), spring/fall peak, spring/fall mid,
    # summer peak, summer mid (08:00 starts mid), summer peak (16:00 starts peak)
    assert got.tolist() == [138.7, 138.7, 172.4, 153.8, 220.5, 184.7, 220.5]
    assert tariff.max_price == 220.5
    # light 10 h, mid 8 h, peak 6 h per day; 5 spring/fall months (153 days), 7 other months (212 days)
    sf = (10 * 125.8 + 8 * 153.8 + 6 * 172.4) / 24
    sw = (10 * 138.7 + 8 * 184.7 + 6 * 220.5) / 24
    assert tariff.mean_price == pytest.approx((sf * 153 + sw * 212) / 365)


# ------------------------------------------------------------- gym contract
def test_check_env_continuous(train_houses, tariff):
    check_env(make(train_houses, tariff), skip_render_check=True)


def test_check_env_with_forecast_features(train_houses, tariff):
    env = make(train_houses, tariff, use_forecast=True)
    assert env.observation_space.shape == (20,)
    check_env(env, skip_render_check=True)


def test_check_env_discrete(train_houses, tariff):
    check_env(DiscreteActionWrapper(make(train_houses, tariff), 5), skip_render_check=True)


def test_discrete_levels_include_idle(train_houses, tariff):
    w = DiscreteActionWrapper(make(train_houses, tariff), 5)
    assert w.levels.tolist() == [-1.0, -0.5, 0.0, 0.5, 1.0]
    with pytest.raises(ValueError):
        DiscreteActionWrapper(make(train_houses, tariff), 4)


def test_same_seed_same_trajectory(train_houses, tariff):
    def run():
        env = make(train_houses, tariff)
        obs, info = env.reset(seed=7)
        out = [info["house_id"]]
        rng = np.random.default_rng(0)
        for _ in range(50):
            obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=1))
            out.append((r, info["soc_kwh"]))
        return out
    assert run() == run()


# ----------------------------------------------------------------- physics
def test_soc_always_inside_bounds_and_energy_balance(train_houses, tariff):
    env = make(train_houses, tariff)
    c = env.cfg
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(env.horizon):
        soc_before = env.soc
        _, _, _, _, info = env.step(rng.uniform(-1, 1, size=1))
        assert env.soc_lo - 1e-9 <= env.soc <= env.soc_hi + 1e-9
        # the change in stored energy follows from the AC-side battery power and the efficiencies
        p = info["p_batt_kw"]
        expected = c.eta_charge * p if p >= 0 else p / c.eta_discharge
        assert env.soc - soc_before == pytest.approx(expected, abs=1e-9)
        # grid balance: import - export == load - pv + battery
        t = info["index"]
        h = env.house
        assert info["grid_import_kwh"] - info["grid_export_kwh"] == pytest.approx(h.load[t] - h.pv[t] + p)


def test_round_trip_efficiency(train_houses, tariff):
    env = make(train_houses, tariff)
    c = env.cfg
    env.reset(seed=0, options={"init_soc_frac": 0.0})   # battery at its lower bound
    start = env.soc
    env.step([0.2])                                       # charge 0.2 * 5 kW for one hour = 1 kWh (AC)
    stored = env.soc - start
    assert stored == pytest.approx(c.eta_charge * 1.0)
    _, _, _, _, info = env.step([-stored * c.eta_discharge / c.p_max_kw])  # deliver everything stored
    assert env.soc == pytest.approx(start)
    assert -info["p_batt_kw"] == pytest.approx(c.eta_charge * c.eta_discharge)  # 0.9025 kWh back per 1 kWh in


def test_full_battery_cannot_charge_and_empty_cannot_discharge(train_houses, tariff):
    env = make(train_houses, tariff)
    env.reset(seed=0, options={"init_soc_frac": 1.0})
    _, _, _, _, info = env.step([1.0])
    assert info["p_batt_kw"] == pytest.approx(0.0)
    env.reset(seed=0, options={"init_soc_frac": 0.0})
    _, _, _, _, info = env.step([-1.0])
    assert info["p_batt_kw"] == pytest.approx(0.0)


# ------------------------------------------------------------ reward / cost
def test_idle_policy_cost_equals_independent_calculation(test_houses, tariff):
    env = make(test_houses, tariff, mode="eval", house_id=POOL[0], terminal_value=False)
    env.reset(seed=0)
    h = env.house
    done = False
    while not done:
        _, _, _, done, info = env.step([0.0])
    expected = (h.price[h.i0:h.i1] * np.maximum(h.load[h.i0:h.i1] - h.pv[h.i0:h.i1], 0)).sum()
    assert info["cum_cost_won"] == pytest.approx(expected)
    assert info["cum_nothing_won"] == pytest.approx(expected)   # idle battery == no battery


def test_reward_is_negative_scaled_cost(train_houses, tariff):
    env = make(train_houses, tariff, terminal_value=False)
    env.reset(seed=3)
    _, r, _, _, info = env.step([0.0])
    assert r == pytest.approx(-info["cost_won"] / env.cfg.reward_scale)


def test_savings_reward_is_zero_when_idle_and_shifts_policy_ranking_not_at_all(train_houses, tariff):
    env = make(train_houses, tariff, reward_mode="savings", terminal_value=False)
    env.reset(seed=3)
    _, r, _, _, _ = env.step([0.0])
    assert r == pytest.approx(0.0)
    # "savings" = "cost" reward + (cost without battery)/scale, and that term does not depend on the action
    a = make(train_houses, tariff, reward_mode="cost", terminal_value=False)
    b = make(train_houses, tariff, reward_mode="savings", terminal_value=False)
    a.reset(seed=5)
    b.reset(seed=5)
    for act in ([0.7], [-0.4], [0.0]):
        _, ra, *_ = a.step(act)
        _, rb, *_ = b.step(act)
        nothing_step = (b.cum_nothing - getattr(b, "_prev", 0.0)) / b.cfg.reward_scale
        b._prev = b.cum_nothing
        assert rb - ra == pytest.approx(nothing_step)


def test_terminal_value_charges_for_energy_removed(train_houses, tariff):
    env = make(train_houses, tariff, terminal_value=True)
    env.reset(seed=0, options={"horizon": 3, "init_soc_frac": 1.0})
    for _ in range(2):
        env.step([-1.0])
    _, r, _, trunc, info = env.step([-1.0])
    assert trunc
    assert info["terminal_credit_won"] < 0           # battery ended emptier than it started
    assert info["terminal_credit_won"] == pytest.approx(env.terminal_price * (env.soc - env.soc_start))


# ---------------------------------------------- no leakage / split discipline
def test_train_episodes_stay_inside_train_split(train_houses, tariff):
    env = make(train_houses, tariff)
    for seed in range(60):
        env.reset(seed=seed)
        h = env.house
        assert h.i0 <= env.t and env.t + env.horizon <= h.i1
        assert h.hour[env.t] == 0                      # episodes start at midnight
    # the train split really ends before the validation / test periods
    ts_end = pd.Timestamp(DATA_CFG["splits"]["train"][1])
    assert ts_end < pd.Timestamp(DATA_CFG["splits"]["val"][0]) < pd.Timestamp(DATA_CFG["splits"]["test"][0])


def test_eval_covers_exactly_the_test_year(test_houses, tariff):
    env = make(test_houses, tariff, mode="eval", house_id=POOL[0])
    env.reset(seed=0)
    assert env.horizon == 365 * 24
    assert env.t == env.house.i0


def test_observation_uses_only_past_load_and_pv(train_houses, tariff):
    env = make(train_houses, tariff, use_forecast=True)
    obs, _ = env.reset(seed=2)
    h, c, t = env.house, env.cfg, env.t
    # previous-hour load/pv sit right after the soc and the 6 price entries
    assert obs[12] == pytest.approx(h.load[t - 1] / c.load_scale_kw, rel=1e-6)
    assert obs[13] == pytest.approx(h.pv[t - 1] / c.pv_scale_kw, rel=1e-6)
    # forecast features are the same hours one day earlier: indices t-24 .. t-22
    assert obs[14:17] == pytest.approx(h.load[t - 24:t - 21] / c.load_scale_kw, rel=1e-6)
    assert obs[17:20] == pytest.approx(h.pv[t - 24:t - 21] / c.pv_scale_kw, rel=1e-6)
    # changing the CURRENT hour's load must not change the observation
    # (the loaded arrays are read-only on purpose, so tamper with a copy)
    tampered = h.load.copy()
    tampered[t] += 100.0
    env.house = dataclasses.replace(h, load=tampered)
    assert np.array_equal(env._obs(), obs)
    # ...but changing the PREVIOUS hour must change it (the test can actually detect a dependency)
    tampered = h.load.copy()
    tampered[t - 1] += 100.0
    env.house = dataclasses.replace(h, load=tampered)
    assert not np.array_equal(env._obs(), obs)


def test_price_lookahead_matches_tariff(train_houses, tariff):
    env = make(train_houses, tariff)
    obs, _ = env.reset(seed=4)
    h, t = env.house, env.t
    assert obs[6:12] == pytest.approx(h.price[t:t + 6] / tariff.max_price, rel=1e-6)
