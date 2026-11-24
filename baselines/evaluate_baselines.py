"""Evaluate all baselines on every household and split, and save tables + one example trajectory.

Usage:  python -m baselines.evaluate_baselines
Outputs: results/baselines.csv, results/baselines_summary.csv, results/rule_based_params.json,
         results/baseline_trajectories.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import DoNothing, LPOracle, RuleBased, SelfConsumption
from baselines.rule_based import tune
from env import EnergyEnv, EnvConfig, Tariff, load_data_config, load_houses
from env.evaluation import mean_ci95, run_policy

OUT = Path("results")


def make_envs(split, house_ids, tariff, cfg):
    houses = load_houses(house_ids, split, tariff)
    return {h: EnergyEnv(houses, tariff, cfg, mode="eval", house_id=h) for h in house_ids}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data_cfg = load_data_config()
    tariff = Tariff.from_yaml("configs/tariff_jeju.yaml")
    cfg = EnvConfig.from_yaml()
    groups = {"seen": data_cfg["pool_houses"], "heldout": data_cfg["heldout_houses"]}
    envs = {s: make_envs(s, groups["seen"] + groups["heldout"], tariff, cfg) for s in ("val", "test")}

    # 1) tune the rule-based parameters on the validation split of the SEEN houses only
    params, val_cost = tune({h: envs["val"][h] for h in groups["seen"]})
    (OUT / "rule_based_params.json").write_text(json.dumps(params, indent=2))
    print("tuned rule-based parameters:", params, f"(mean val cost {val_cost:.0f} won)")

    # 2) evaluate every policy on every house of both splits
    policies = [DoNothing(), SelfConsumption(), RuleBased(**params), LPOracle()]
    rows, traj = [], {}
    for split in ("val", "test"):
        for group, ids in groups.items():
            for h in ids:
                res = {p.name: run_policy(envs[split][h], p, record=(split == "test" and h == ids[0] and group == "seen"))
                       for p in policies}
                lp_saving = res["lp_oracle"]["nothing_cost"] - res["lp_oracle"]["total_cost"]
                for name, r in res.items():
                    rows.append(dict(split=split, group=group, house=h, policy=name,
                                     total_cost_won=r["total_cost"], nothing_cost_won=r["nothing_cost"],
                                     saving_pct=r["saving_pct"],
                                     captured_pct=100.0 * (r["nothing_cost"] - r["total_cost"]) / lp_saving))
                    if "trajectory" in r:
                        for k, v in r["trajectory"].items():
                            traj[f"{name}/{k}"] = v
                        traj["house"] = np.array(h)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "baselines.csv", index=False)
    np.savez_compressed(OUT / "baseline_trajectories.npz", **traj)

    # 3) summary: mean +- 95 % CI over houses
    summ = []
    for (split, group, policy), g in df.groupby(["split", "group", "policy"], sort=False):
        s_m, s_h = mean_ci95(g["saving_pct"])
        c_m, c_h = mean_ci95(g["captured_pct"])
        summ.append(dict(split=split, group=group, policy=policy, n_houses=len(g), saving_pct=s_m,
                         saving_ci95=s_h, captured_pct=c_m, captured_ci95=c_h))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "baselines_summary.csv", index=False)
    print(summ[summ.split == "test"].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
