"""Baseline tests: the LP oracle must agree with the environment, and no policy may beat it."""
import numpy as np
import pytest

from baselines import DoNothing, LPOracle, RuleBased, SelfConsumption
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
    """Independent check that LP physics == environment physics."""
    lp = LPOracle()
    res = run_policy(val_env, lp)
    assert res["total_cost"] == pytest.approx(lp.objective, rel=1e-4)


def test_lp_matches_env_with_nonzero_degradation_and_export_price(tariff):
    houses = load_houses([HOUSE], "val", tariff)
    cfg = EnvConfig.from_yaml(degradation_cost=15.0, export_price=40.0)
    env = EnergyEnv(houses, tariff, cfg, mode="eval", house_id=HOUSE)
    lp = LPOracle()
    assert run_policy(env, lp)["total_cost"] == pytest.approx(lp.objective, rel=1e-4)


def test_nobody_beats_the_oracle(val_env):
    # untuned default rule parameters may even lose to doing nothing; only the oracle bound is guaranteed
    costs = {p.name: run_policy(val_env, p)["total_cost"]
             for p in (DoNothing(), SelfConsumption(), RuleBased(), LPOracle())}
    assert costs["lp_oracle"] <= min(costs.values()) + 1e-6
    assert costs["lp_oracle"] < costs["do_nothing"]
    assert costs["self_consumption"] <= costs["do_nothing"] + 1e-6


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
