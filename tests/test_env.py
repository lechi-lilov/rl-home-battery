"""Environment tests. Each test checks one physical or logical property independently of the env code."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from env import DiscreteActionWrapper, EnergyEnv, EnvConfig, Tariff, load_data_config, load_houses
from env.data import split_days

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
    env = make(test_houses, tariff, mode="eval", house_id=POOL[0], terminal_value=False, reward_shaping="none")
    env.reset(seed=0)
    h = env.house
    done = False
    while not done:
        _, _, _, done, info = env.step([0.0])
    s, e = h.segments[0]
    expected = (h.price[s:e] * np.maximum(h.load[s:e] - h.pv[s:e], 0)).sum()
    assert info["cum_cost_won"] == pytest.approx(expected)
    assert info["cum_nothing_won"] == pytest.approx(expected)   # idle battery == no battery


def test_reward_is_negative_scaled_cost(train_houses, tariff):
    env = make(train_houses, tariff, terminal_value=False, reward_shaping="none", reward_mode="cost")
    env.reset(seed=3)
    _, r, _, _, info = env.step([0.0])
    assert r == pytest.approx(-info["cost_won"] / env.cfg.reward_scale)


def test_savings_reward_is_zero_when_idle_and_shifts_policy_ranking_not_at_all(train_houses, tariff):
    env = make(train_houses, tariff, reward_mode="savings", terminal_value=False, reward_shaping="none")
    env.reset(seed=3)
    _, r, _, _, _ = env.step([0.0])
    assert r == pytest.approx(0.0)
    # "savings" = "cost" reward + (cost without battery)/scale, and that term does not depend on the action
    a = make(train_houses, tariff, reward_mode="cost", terminal_value=False, reward_shaping="none")
    b = make(train_houses, tariff, reward_mode="savings", terminal_value=False, reward_shaping="none")
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
def test_splits_are_disjoint_and_validation_covers_every_month():
    train, val, test = (split_days(s, DATA_CFG) for s in ("train", "val", "test"))
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert train.max() < test.min() and val.max() < test.min()       # test comes after all development data
    assert sorted(set(val.month)) == list(range(1, 13))              # every season is validated
    assert set(val.day) == set(range(8, 15))                         # days 8..14 of each month
    assert len(val) == 24 * 7 and len(test) == 365                   # 24 months x 1 week, one full year


def test_segments_match_split_days(tariff):
    h = load_houses(POOL[:1], "val", tariff)[POOL[0]]
    hours = np.concatenate([np.arange(s, e) for s, e in h.segments])
    days = pd.DatetimeIndex(h.timestamps(hours).normalize().unique())
    assert days.equals(split_days("val", DATA_CFG))
    assert all(h.hour[s] == 0 for s, _ in h.segments)               # every segment starts at midnight


def test_train_episodes_never_touch_validation_or_test_days(train_houses, tariff):
    """Not even as INPUT: observations look back up to 24 h (previous hour, persistence forecast),
    so the look-back window must also consist of training days only."""
    env = make(train_houses, tariff, use_forecast=True)
    allowed = set(split_days("train", DATA_CFG))
    for seed in range(300):
        env.reset(seed=seed)
        h = env.house
        assert h.hour[env.t] == 0                                     # episodes start at midnight
        seen_hours = np.arange(env.t - 24, env.t + env.horizon)       # look-back window + episode
        assert set(h.timestamps(seen_hours).normalize()) <= allowed


def test_eval_covers_exactly_the_test_year(test_houses, tariff):
    env = make(test_houses, tariff, mode="eval", house_id=POOL[0])
    env.reset(seed=0)
    assert len(env.house.segments) == 1
    assert env.horizon == 365 * 24
    assert env.house.timestamps([env.t])[0] == pd.Timestamp("2012-07-01")


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


def test_potential_shaping_with_gamma_one_equals_terminal_credit(train_houses, tariff):
    """With gamma = 1 the shaping terms telescope to terminal_price * (SoC_end - SoC_start), i.e. exactly the
    terminal credit, so both reward variants give the same episode return for the same actions."""
    plain = make(train_houses, tariff, reward_shaping="none")
    shaped = make(train_houses, tariff, reward_shaping="potential", shaping_gamma=1.0)
    plain.reset(seed=11)
    shaped.reset(seed=11)
    rng = np.random.default_rng(3)
    ret_plain = ret_shaped = 0.0
    done = False
    while not done:
        a = rng.uniform(-1, 1, size=1)
        _, r1, _, done, info = plain.step(a)
        _, r2, *_ = shaped.step(a)
        ret_plain += r1
        ret_shaped += r2
    assert info["terminal_credit_won"] != 0.0
    assert ret_shaped == pytest.approx(ret_plain, rel=1e-9)


def test_potential_shaping_step_value(train_houses, tariff):
    env = make(train_houses, tariff, reward_shaping="potential", shaping_gamma=0.99, terminal_value=True, reward_mode="cost")
    env.reset(seed=2, options={"init_soc_frac": 0.5})
    soc0 = env.soc
    _, r, _, _, info = env.step([0.4])
    expected = (-info["cost_won"] + 0.99 * env.terminal_price * env.soc - env.terminal_price * soc0) / env.cfg.reward_scale
    assert r == pytest.approx(expected)
    with pytest.raises(ValueError):
        EnvConfig.from_yaml(reward_shaping="potential", terminal_value=False)


@pytest.mark.parametrize("mode", ["cost", "savings"])
@pytest.mark.parametrize("shaping", ["none", "potential"])
def test_episode_rewards_add_up_to_the_money_metric(train_houses, tariff, mode, shaping):
    """Regression test for a real bug: with shaping_gamma < 1 a money-losing policy got a HIGHER reward than doing
    nothing. With the default config the undiscounted episode return must equal the evaluation metric exactly."""
    env = make(train_houses, tariff, reward_mode=mode, reward_shaping=shaping)
    assert env.cfg.shaping_gamma == 1.0
    rng = np.random.default_rng(7)
    for seed in range(5):
        env.reset(seed=seed)
        ret, done = 0.0, False
        while not done:
            _, r, _, done, info = env.step(rng.uniform(-1, 1, size=1))
            ret += r
        total_cost = info["cum_cost_won"] + info["cum_degradation_won"] - info["terminal_credit_won"]
        money = -total_cost if mode == "cost" else info["cum_nothing_won"] - total_cost
        assert ret == pytest.approx(money / env.cfg.reward_scale, rel=1e-9, abs=1e-9)


def test_replacement_value_makes_buy_and_hold_worth_nothing(train_houses, tariff):
    """With stored_energy_value=replacement, charging at the cheapest price and holding earns zero reward."""
    env = make(train_houses, tariff, reward_mode="savings", reward_shaping="potential",
               stored_energy_value="replacement")
    assert env.terminal_price == pytest.approx(tariff.min_price / env.cfg.eta_charge)
    env.reset(seed=0, options={"init_soc_frac": 0.0})
    h = env.house
    # move to an hour with the cheapest price and no PV surplus
    while not (h.price[env.t] == tariff.min_price and h.pv[env.t] <= h.load[env.t]):
        env.step([0.0])
    _, r, *_ = env.step([0.2])            # buy 1 kWh from the grid, store 0.95 kWh
    assert r == pytest.approx(0.0, abs=1e-9)
