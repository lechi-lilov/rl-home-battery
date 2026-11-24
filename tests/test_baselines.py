"""Baseline tests: the LP oracle must agree with the environment, and no policy may beat it."""
import dataclasses

import numpy as np
import pytest

from baselines import DoNothing, LPOracle, MPCForecast, MPCPersistence, RuleBased, SelfConsumption, SelfConsumptionCausal
from baselines.mpc import forecast_indices, persistence_indices
from env import EnergyEnv, EnvConfig, Tariff, load_data_config, load_houses
from env.evaluation import mean_ci95, run_policy

CFG = load_data_config()
HOUSE = CFG["pool_houses"][0]


@pytest.fixture(scope="module")
def tariff():
    return Tariff.from_yaml("configs/tariff_jeju.yaml")


@pytest.fixture(scope="module")
def val_env(tariff):
    houses = load_houses([HOUSE], "val", tariff)
    return EnergyEnv(houses, tariff, EnvConfig.from_yaml(), mode="eval", house_id=HOUSE)


def test_lp_plan_replayed_in_env_matches_lp_objective(val_env):
    """Independent check that LP physics == environment physics (one LP per validation week)."""
    lp = LPOracle()
    res = run_policy(val_env, lp)
    assert res["n_segments"] == len(lp.objectives) == 24
    assert res["total_cost"] == pytest.approx(sum(lp.objectives), rel=1e-4)


def test_lp_matches_env_with_nonzero_degradation_and_export_price(tariff):
    houses = load_houses([HOUSE], "val", tariff)
    cfg = EnvConfig.from_yaml(degradation_cost=15.0, export_price=40.0)
    env = EnergyEnv(houses, tariff, cfg, mode="eval", house_id=HOUSE)
    lp = LPOracle()
    assert run_policy(env, lp)["total_cost"] == pytest.approx(sum(lp.objectives), rel=1e-4)


def test_nobody_beats_the_oracle(val_env):
    # untuned default rule parameters may even lose to doing nothing; only the oracle bound is guaranteed
    costs = {p.name: run_policy(val_env, p)["total_cost"]
             for p in (DoNothing(), SelfConsumption(), SelfConsumptionCausal(), RuleBased(), MPCPersistence(), LPOracle())}
    assert costs["lp_oracle"] <= min(costs.values()) + 1e-6
    assert costs["lp_oracle"] < costs["do_nothing"]
    assert costs["self_consumption"] <= costs["do_nothing"] + 1e-6


@pytest.mark.parametrize("policy", [SelfConsumptionCausal(), RuleBased(mid_selfcons=True)], ids=lambda p: p.name)
def test_causal_baselines_ignore_the_current_hour(val_env, policy):
    """The causal baselines must have the same information as the RL agent: nothing from hour t itself."""
    obs, _ = val_env.reset(seed=0, options={"segment": 3})
    original = val_env.house
    for _ in range(10):                      # move into the day so mid/peak hours are covered too
        obs, *_ = val_env.step(policy.act(obs, val_env))
    try:
        for _ in range(14):
            t = val_env.t
            a = policy.act(obs, val_env)
            load, pv = original.load.copy(), original.pv.copy()
            load[t] += 3.0
            pv[t] += 2.0                     # change the CURRENT hour only
            val_env.house = dataclasses.replace(original, load=load, pv=pv)
            assert np.array_equal(policy.act(obs, val_env), a)
            val_env.house = original
            obs, *_ = val_env.step(a)
    finally:
        val_env.house = original


def test_do_nothing_saves_nothing(val_env):
    assert run_policy(val_env, DoNothing())["saving_pct"] == pytest.approx(0.0)


def test_self_consumption_never_imports_to_charge(val_env):
    res = run_policy(val_env, SelfConsumption(), record=True)
    tr = res["trajectory"]
    h = val_env.house
    net0 = np.maximum(h.load[tr["index"]] - h.pv[tr["index"]], 0.0)
    charging = tr["p_batt_kw"] > 1e-9
    assert charging.any()
    # While charging the home has a PV surplus, so the grid import must be zero.
    # The action is a float32, so allow rounding noise of ~1e-7 relative to the 5 kW power scale.
    assert np.all(tr["grid_import_kwh"][charging] <= net0[charging] + 1e-6)
    assert tr["grid_import_kwh"][charging].max() < 1e-6


def test_rule_based_only_charges_when_cheap_and_discharges_when_expensive(val_env):
    res = run_policy(val_env, RuleBased(mid_selfcons=False), record=True)
    tr = res["trajectory"]
    h = val_env.house
    for t, p in zip(tr["index"], tr["p_batt_kw"]):
        day = h.price[t - int(h.hour[t]):t - int(h.hour[t]) + 24]
        if p > 1e-9:
            assert h.price[t] == day.min()
        if p < -1e-9:
            assert h.price[t] == day.max()


def test_mean_ci95():
    m, half = mean_ci95([1.0, 2.0, 3.0])
    assert m == 2.0
    assert half == pytest.approx(4.302653 * 1.0 / np.sqrt(3), rel=1e-5)  # t(0.975, 2) * s / sqrt(n)


def test_persistence_forecast_uses_only_past_hours():
    for H in (24, 48, 72):
        idx = persistence_indices(1000, H)
        assert idx.max() < 1000                                   # strictly the past
        assert np.all((1000 + np.arange(H) - idx) % 24 == 0)      # same clock hour
        many = forecast_indices(1000, H, 7)
        assert many.shape == (7, H) and many.max() < 1000 and np.all((1000 + np.arange(H) - many) % 24 == 0)
        assert np.array_equal(many[0], idx)


@pytest.mark.parametrize("horizon,n_days", [(24, 1), (48, 1), (48, 7)])
def test_mpc_first_day_does_not_depend_on_the_future(val_env, horizon, n_days):
    """Change every load/PV value from the decision time on: the first day's plan must not change."""
    def first_day_actions():
        mpc = MPCForecast(horizon, n_days)
        obs, _ = val_env.reset(seed=0, options={"segment": 5})
        mpc.reset(val_env)
        out = []
        for _ in range(24):
            a = mpc.act(obs, val_env)
            out.append(a)
            obs, *_ = val_env.step(a)
        return np.array(out)

    original = val_env.house
    t0 = original.segments[5][0]
    before = first_day_actions()
    load, pv = original.load.copy(), original.pv.copy()
    load[t0:] *= 3.0
    pv[t0:] = 0.0
    try:
        val_env.houses[original.house_id] = dataclasses.replace(original, load=load, pv=pv)
        after = first_day_actions()
    finally:
        val_env.houses[original.house_id] = original
    assert np.array_equal(before, after)


def test_mpc_never_beats_the_oracle(val_env):
    costs = {p.name: run_policy(val_env, p)["total_cost"] for p in (MPCForecast(48, 7), LPOracle())}
    assert costs["lp_oracle"] <= costs["mpc"] + 1e-6
