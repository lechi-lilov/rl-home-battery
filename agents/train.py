"""Train one RL agent (PPO, SAC or DQN) on the training homes and training days.

Every `eval_every` steps the current policy is evaluated on the VALIDATION weeks of the same homes
(never on the test year). The checkpoint with the best validation saving is kept as best_model.zip.

Usage examples
  python -m agents.train --algo ppo --seed 0 --steps 200000
  python -m agents.train --algo sac --seed 1 --steps 200000 --lr 1e-4 --gamma 0.995
  python -m agents.train --algo ppo --seed 0 --steps 200000 --env-override use_forecast=true

Output: runs/<algo>_<tag>_s<seed>/ with config.json, progress.csv (learning curve),
        best_model.zip, final_model.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import stable_baselines3
import torch
import yaml
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agents.policy_adapter import SB3Policy
from env import DiscreteActionWrapper, EnergyEnv, EnvConfig, Tariff, load_data_config, load_houses
from env.evaluation import run_policy

ALGOS = {"ppo": PPO, "sac": SAC, "dqn": DQN}


def parse_override(items: list[str]) -> dict:
    """'key=value' strings -> dict, values parsed as YAML (so true/1e-4/0.5 get the right type)."""
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        parsed = yaml.safe_load(value)
        if isinstance(parsed, str):          # YAML 1.1 reads "1e-4" as a string; accept it as a number
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        out[key] = parsed
    return out


def git_commit() -> str:
    """Commit hash of the code used for this run (+ '-dirty' if there were uncommitted changes)."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class ValidationCallback(BaseCallback):
    """Evaluate on the validation weeks every `eval_every` steps, log the curve, keep the best model."""

    def __init__(self, val_envs: dict, run_dir: Path, eval_every: int, levels=None, name="agent"):
        super().__init__()
        self.val_envs, self.run_dir, self.eval_every = val_envs, run_dir, eval_every
        self.levels, self.name = levels, name
        self.best = -np.inf
        self.next_eval = 0
        self.t0 = time.time()
        self.file = open(run_dir / "progress.csv", "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timesteps", "val_saving_pct", "val_saving_pct_min", "val_saving_pct_max",
                              "mean_train_episode_reward", "wall_seconds"])

    def evaluate(self):
        policy = SB3Policy(self.model, self.name, self.levels)
        savings = [run_policy(env, policy)["saving_pct"] for env in self.val_envs.values()]
        ep = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else [np.nan]
        mean = float(np.mean(savings))
        self.writer.writerow([self.num_timesteps, round(mean, 4), round(min(savings), 4), round(max(savings), 4),
                              round(float(np.mean(ep)), 4), round(time.time() - self.t0, 1)])
        self.file.flush()
        if mean > self.best:
            self.best = mean
            self.model.save(self.run_dir / "best_model")
        if self.verbose:
            print(f"  step {self.num_timesteps:>8d}  val saving {mean:6.2f} %  (best {self.best:6.2f} %)")

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.evaluate()
            self.next_eval += self.eval_every
        return True

    def _on_training_end(self) -> None:
        self.evaluate()
        self.file.close()


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=list(ALGOS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--lr", type=float, default=None, help="learning rate (default from configs/algos.yaml)")
    ap.add_argument("--gamma", type=float, default=None, help="discount factor (default from configs/algos.yaml)")
    ap.add_argument("--eval-every", type=int, default=20_000)
    ap.add_argument("--tag", default="default", help="name of the configuration, used in the run folder")
    ap.add_argument("--env-override", nargs="*", default=[], help="EnvConfig overrides, e.g. use_forecast=true")
    ap.add_argument("--hp", nargs="*", default=[], help="algorithm hyper-parameter overrides, e.g. use_sde=true n_levels=11")
    ap.add_argument("--policy-kw", nargs="*", default=[], help="policy_kwargs overrides, e.g. squash_output=true")
    ap.add_argument("--norm-reward", action="store_true",
                    help="scale training rewards by a running estimate of the return std (VecNormalize). "
                         "Training only: observations and the evaluation are unchanged")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args(argv)

    algo_cfg = yaml.safe_load(Path("configs/algos.yaml").read_text(encoding="utf-8"))
    hp = {**{k: v for k, v in algo_cfg["common"].items() if k != "net_arch"}, **algo_cfg[args.algo]}
    if args.lr is not None:
        hp["learning_rate"] = args.lr
    if args.gamma is not None:
        hp["gamma"] = args.gamma
    hp.update(parse_override(args.hp))
    n_levels = hp.pop("n_levels", None)
    policy_kwargs = dict(net_arch=algo_cfg["common"]["net_arch"], **parse_override(args.policy_kw))
    env_over = parse_override(args.env_override)

    run_dir = Path(args.runs_dir) / f"{args.algo}_{args.tag}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_data_config()
    tariff = Tariff.from_yaml("configs/tariff_jeju.yaml")
    env_cfg = EnvConfig.from_yaml(**env_over)
    pool = data_cfg["pool_houses"]
    train_houses = load_houses(pool, "train", tariff)
    val_houses = load_houses(pool, "val", tariff)

    env = EnergyEnv(train_houses, tariff, env_cfg, mode="train")
    levels = None
    if n_levels:   # DQN always; PPO too when --hp n_levels=K is given (categorical policy, Tang & Agrawal 2020)
        if args.algo == "sac":
            raise ValueError("SAC needs a continuous action space")
        env = DiscreteActionWrapper(env, n_levels)
        levels = env.levels
    env = Monitor(env)
    env.reset(seed=args.seed)
    if args.norm_reward:
        # Monitor sits inside, so logged episode rewards stay in the original (unscaled) units
        env = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=False, norm_reward=True, gamma=hp["gamma"])
    val_envs = {h: EnergyEnv(val_houses, tariff, env_cfg, mode="eval", house_id=h) for h in pool}

    torch.set_num_threads(1)  # small networks: one thread is fastest and keeps parallel runs independent
    model = ALGOS[args.algo]("MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
                             policy_kwargs=policy_kwargs, **hp)

    config = dict(algo=args.algo, seed=args.seed, steps=args.steps, tag=args.tag, hyperparameters=hp,
                  policy_kwargs=policy_kwargs, n_levels=n_levels, env_overrides=env_over, norm_reward=args.norm_reward,
                  env_config=vars(env_cfg), train_homes=pool, git_commit=git_commit(), versions=dict(
                      python=platform.python_version(), torch=torch.__version__,
                      stable_baselines3=stable_baselines3.__version__, numpy=np.__version__))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    cb = ValidationCallback(val_envs, run_dir, args.eval_every, levels, name=args.algo)
    cb.verbose = 1
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(run_dir / "final_model")
    summary = dict(run=str(run_dir), best_val_saving_pct=cb.best, minutes=round((time.time() - t0) / 60, 1))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)
    return summary


if __name__ == "__main__":
    main()
