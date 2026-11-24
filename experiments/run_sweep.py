"""Hyper-parameter sweep and final multi-seed runs (protocol in README, "Evaluation protocol").

  python -m experiments.run_sweep sweep   --workers 7   # 3 algos x 3 lr x 2 gamma x 2 seeds, validation only
  python -m experiments.run_sweep select                # pick the best config per algorithm (validation)
  python -m experiments.run_sweep final   --workers 7   # selected configs x 5 NEW seeds

Every algorithm gets the same grid and the same number of environment steps (equal tuning budget).
Grid values follow the search ranges of RL Baselines3 Zoo (lr in [1e-5, 2e-3], gamma in [0.97, 0.9999]);
gamma 0.97 / 0.99 = effective horizon of ~33 h (one daily cycle) / ~100 h (several days).
Runs that already finished (summary.json exists) are skipped, so the script can be restarted safely.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ALGOS = ["sac", "ppo", "dqn"]           # SAC first: it is the slowest, so it should start first
LRS = [1e-4, 3e-4, 1e-3]
GAMMAS = [0.97, 0.99]
SWEEP_SEEDS = [0, 1]
FINAL_SEEDS = [100, 101, 102, 103, 104]  # never used during the sweep -> no selection bias in the final numbers
STEPS = 500_000
EVAL_EVERY = 50_000
SWEEP_DIR, FINAL_DIR = Path("runs/sweep"), Path("runs/final")
SELECTED = Path("results/selected_configs.json")


def tag(lr: float, gamma: float) -> str:
    return f"lr{lr:g}_g{gamma:g}"


def run_one(algo: str, lr: float, gamma: float, seed: int, runs_dir: Path) -> str:
    name = f"{algo}_{tag(lr, gamma)}_s{seed}"
    if (runs_dir / name / "summary.json").exists():
        return f"skip {name} (done)"
    runs_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "agents.train", "--algo", algo, "--seed", str(seed), "--steps", str(STEPS),
           "--lr", str(lr), "--gamma", str(gamma), "--tag", tag(lr, gamma), "--runs-dir", str(runs_dir),
           "--eval-every", str(EVAL_EVERY)]
    with open(runs_dir / f"{name}.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return f"{'done' if code == 0 else 'FAILED'} {name}"


def launch(jobs: list[tuple], runs_dir: Path, workers: int) -> None:
    print(f"{len(jobs)} runs, {workers} in parallel -> {runs_dir}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, *job, runs_dir) for job in jobs]
        for i, f in enumerate(as_completed(futures), 1):
            print(f"[{i}/{len(jobs)}] {f.result()}", flush=True)


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for cfg_file in sorted(runs_dir.glob("*/config.json")):
        run = cfg_file.parent
        if not (run / "summary.json").exists():
            continue
        cfg = json.loads(cfg_file.read_text())
        prog = pd.read_csv(run / "progress.csv")
        rows.append(dict(run=run.name, algo=cfg["algo"], lr=cfg["hyperparameters"]["learning_rate"],
                         gamma=cfg["hyperparameters"]["gamma"], seed=cfg["seed"],
                         best_val_saving_pct=prog.val_saving_pct.max(),       # = the kept checkpoint (best_model)
                         final_val_saving_pct=prog.val_saving_pct.iloc[-1],
                         minutes=json.loads((run / "summary.json").read_text())["minutes"]))
    return pd.DataFrame(rows)


def select() -> None:
    df = collect(SWEEP_DIR)
    df.to_csv("results/sweep_runs.csv", index=False)
    per_cfg = (df.groupby(["algo", "lr", "gamma"])
                 .agg(mean_best_val=("best_val_saving_pct", "mean"), min_best_val=("best_val_saving_pct", "min"),
                      n_seeds=("seed", "nunique"))
                 .reset_index().sort_values(["algo", "mean_best_val"], ascending=[True, False]))
    per_cfg.to_csv("results/sweep_summary.csv", index=False)
    print(per_cfg.round(3).to_string(index=False))
    best = {a: per_cfg[per_cfg.algo == a].iloc[0] for a in per_cfg.algo.unique()}
    sel = {a: dict(lr=float(r.lr), gamma=float(r.gamma), mean_best_val_saving_pct=float(r.mean_best_val))
           for a, r in best.items()}
    SELECTED.write_text(json.dumps(sel, indent=2))
    print("selected (validation only):", json.dumps(sel))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["sweep", "select", "final"])
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if args.stage == "sweep":
        jobs = [(a, lr, g, s) for a in ALGOS for lr, g, s in itertools.product(LRS, GAMMAS, SWEEP_SEEDS)]
        launch(jobs, SWEEP_DIR, args.workers)
    elif args.stage == "select":
        select()
    else:
        sel = json.loads(SELECTED.read_text())
        jobs = [(a, sel[a]["lr"], sel[a]["gamma"], s) for a in ALGOS if a in sel for s in FINAL_SEEDS]
        launch(jobs, FINAL_DIR, args.workers)


if __name__ == "__main__":
    main()
