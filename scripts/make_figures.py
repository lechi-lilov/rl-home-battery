"""Create the figures used in the README / report from the data and result files.

Usage:  python -m scripts.make_figures          (needs results/ from baselines.evaluate_baselines)
Colours follow a validated categorical palette (blue, orange, aqua, yellow) on a light surface.
Contrast of aqua/yellow is below 3:1, so every series is also named by a legend AND a direct label.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from env import Tariff, load_data_config
from env.data import DEFAULT_CSV, split_days

OUT = Path("docs/img")
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#898781"
POLICY_STYLE = {   # colour + marker, so identity never relies on colour alone
    "do_nothing": ("Do nothing", GRAY, "o"),
    "self_consumption_causal": ("Self-consumption (causal)", BLUE, "s"),
    "rule_based": ("Rule-based (tuned, causal)", ORANGE, "^"),
    "mpc": ("MPC (7-day forecast, causal)", MAGENTA, "P"),
    "self_consumption": ("Self-consumption (reactive)", YELLOW, "v"),
    "lp_oracle": ("LP oracle (perfect foresight)", AQUA, "D"),
}
BAND_FILL = {"light": "#eef4fb", "mid": "#fbf6ea", "peak": "#fbeeea"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def shade_bands(ax, ymax):
    """Light / mid / peak hour bands of the Jeju residential tariff."""
    for (a, b), key, label in [((0, 8), "light", "light load"), ((8, 16), "mid", "mid load"),
                               ((16, 22), "peak", "peak load"), ((22, 24), "light", None)]:
        ax.axvspan(a, b, color=BAND_FILL[key], zorder=0, lw=0)
        if label:
            ax.text((a + b) / 2, ymax, label, ha="center", va="top", color=INK2, fontsize=9)


# ---------------------------------------------------------------- figure 1: agent-environment loop
def fig_loop():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=SURFACE, ec=color, lw=2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
        ax.text(x + 0.2, y + h - 0.65, "\n".join(lines), ha="left", va="top", fontsize=9.2, color=INK2, linespacing=1.5)

    box(0.2, 1.0, 3.3, 3.2, "RL agent", ["PPO / SAC  (continuous action)", "DQN  (5 discrete levels)", "",
                                          "learns a policy  a = π(s)", "from 21 months of training data"], BLUE)
    box(6.5, 1.0, 3.3, 3.2, "Environment", ["home + 10 kWh / 5 kW battery", "90 % round-trip efficiency",
                                            "PV + household load (Ausgrid)", "KEPCO time-of-use tariff", "",
                                            "1 step = 1 hour"], ORANGE)
    # (start, end, label, y): action goes agent -> environment, reward and state come back
    for (x0, x1, txt, y) in [(3.55, 6.45, "action a_t = battery power", 3.45),
                             (6.45, 3.55, "reward r_t = − cost", 2.65),
                             (6.45, 3.55, "next state s_{t+1}", 1.85)]:
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18, color=INK2, lw=1.8))
        ax.text(5.0, y + 0.14, txt, ha="center", va="bottom", color=INK, fontsize=9.6)
    ax.text(0.2, 0.62, "State:   hour & season (sin/cos), weekend, battery charge, tariff price now + next 5 h, "
            "previous-hour load and PV", color=INK2, fontsize=9)
    ax.text(0.2, 0.28, "Reward:  − (import × price − export × export price) − battery wear, plus a terminal value "
            "for energy left in the battery", color=INK2, fontsize=9)
    ax.set_title("What the agent sees, does and gets", loc="left")
    save(fig, "loop.png")


# ---------------------------------------------------------------- figure 2: tariff
def fig_tariff():
    t = Tariff.from_yaml("configs/tariff_jeju.yaml")
    hours = np.arange(25)
    fig, ax = plt.subplots(figsize=(9, 4))
    ymax = 245
    shade_bands(ax, ymax)
    for month, color, marker, label in [(4, BLUE, "o", "Spring / autumn"), (7, ORANGE, "s", "Summer / winter")]:
        y = np.append(t.table[month], t.table[month][0])
        ax.step(hours, y, where="post", color=color, marker=marker, markevery=4, ms=6, label=label, zorder=3)
    ax.text(24.2, t.table[7][0] + 2, "summer / winter", color=INK, fontsize=9, va="bottom", ha="right")
    ax.text(24.2, t.table[4][0] - 3, "spring / autumn", color=INK, fontsize=9, va="top", ha="right")
    ax.set_xlim(0, 24); ax.set_ylim(100, ymax); ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("hour of day"); ax.set_ylabel("won / kWh")
    ax.set_title("In summer/winter, evening electricity costs 59 % more than night electricity")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), ncol=2)
    save(fig, "tariff.png")


# ---------------------------------------------------------------- figure 3: data profile
def fig_data():
    cfg = load_data_config()
    df = pd.read_csv(DEFAULT_CSV, parse_dates=["timestamp"])
    train_days = split_days("train", cfg)
    df = df[df.customer.isin(cfg["pool_houses"]) & df.timestamp.dt.normalize().isin(train_days)]
    prof = df.assign(h=df.timestamp.dt.hour).groupby("h")[["load_kw", "pv_kw"]].mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ymax = 1.55
    shade_bands(ax, ymax)
    ax.plot(prof.index, prof.load_kw, color=BLUE, marker="o", markevery=3, ms=6, label="Household load", zorder=3)
    ax.plot(prof.index, prof.pv_kw, color=ORANGE, marker="s", markevery=3, ms=6, label="Rooftop PV output", zorder=3)
    ax.text(prof.load_kw.idxmax() + 0.3, prof.load_kw.max() + 0.03, "evening load peak", color=INK, fontsize=9)
    ax.text(prof.pv_kw.idxmax() + 0.5, prof.pv_kw.max() + 0.03, "PV peak at noon", color=INK, fontsize=9)
    ax.set_xlim(0, 23); ax.set_ylim(0, ymax); ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day (local clock time)"); ax.set_ylabel("average power, kW")
    ax.set_title("PV peaks at noon, but the home's load peaks in the evening")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88))
    save(fig, "data_profile.png")


# ---------------------------------------------------------------- figure 4: baseline results
def fig_baselines():
    df = pd.read_csv("results/baselines.csv")
    summ = pd.read_csv("results/baselines_summary.csv")
    order = ["lp_oracle", "self_consumption", "mpc", "rule_based", "self_consumption_causal", "do_nothing"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    n = df[df.split == "test"].groupby("group").house.nunique()
    for ax, group, title in zip(axes, ["seen", "heldout"], [f"{n['seen']} homes used in training",
                                                            f"{n['heldout']} homes never seen in training"]):
        for i, p in enumerate(order):
            label, color, marker = POLICY_STYLE[p]
            d = df[(df.split == "test") & (df.group == group) & (df.policy == p)]
            ax.scatter(d.saving_pct, i + rng.uniform(-0.18, 0.18, len(d)), s=26, color=color, alpha=0.45, marker=marker,
                       zorder=2, edgecolor=SURFACE, linewidth=0.5)
            s = summ[(summ.split == "test") & (summ.group == group) & (summ.policy == p)].iloc[0]
            ax.errorbar(s.saving_pct, i, xerr=s.saving_ci95 if np.isfinite(s.saving_ci95) else 0, fmt=marker, color=color,
                        ms=9, capsize=4, lw=2, zorder=3, mec=SURFACE)
            ax.text(s.saving_pct + (s.saving_ci95 if np.isfinite(s.saving_ci95) else 0) + 1, i + 0.3,
                    f"{s.saving_pct:.1f} %", color=INK, fontsize=9, va="center")
        ax.set_yticks(range(len(order))); ax.set_yticklabels([POLICY_STYLE[p][0] for p in order])
        ax.tick_params(labelleft=(ax is axes[0]))
        ax.set_title(title, fontsize=11); ax.set_xlabel("cost saving vs. no battery use, % (test year)")
        ax.grid(axis="y", visible=False); ax.set_ylim(-0.6, len(order) - 0.4)
    shown = df[(df.split == "test") & df.policy.isin(order)].saving_pct
    axes[0].set_xlim(min(-3.0, shown.min() - 2), shown.max() + 6)   # every home stays inside the plot
    t = summ[(summ.split == "test") & (summ.group == "seen")].set_index("policy")
    fig.suptitle(f"Perfect foresight saves {t.loc['lp_oracle', 'saving_pct']:.0f} % of the bill; with only the information "
                 f"the agent has, simple rules save {t.loc['self_consumption_causal', 'saving_pct']:.0f} %",
                 x=0.125, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.125, -0.04, "Dots = individual homes, marker + bar = mean ± 95 % CI over homes (Student t). Test year Jul 2012 – Jun 2013. "
             "Causal = uses the previous hour, like the RL agent; reactive = also sees the current hour.", color=INK2, fontsize=9)
    save(fig, "baselines.png")


# ---------------------------------------------------------------- figure 5: three days, battery state of charge
def pick_typical_days(tr, house: int, cfg_export_price: float = 0.0):
    """Selection rule (fixed, not hand-picked): the test day on which the oracle's saving is closest to its
    median daily saving is the MIDDLE day of the 3-day window."""
    df = pd.read_csv(DEFAULT_CSV, parse_dates=["timestamp"])
    g = df[df.customer == house].set_index("timestamp")
    ts = pd.Timestamp("2010-07-01") + pd.to_timedelta(tr["lp_oracle/index"], unit="h")
    net = g.loc[ts, "load_kw"].to_numpy() - g.loc[ts, "pv_kw"].to_numpy()
    price = tr["lp_oracle/price"]
    nothing = price * np.maximum(net, 0) - cfg_export_price * np.maximum(-net, 0)
    lp = price * tr["lp_oracle/grid_import_kwh"] - cfg_export_price * tr["lp_oracle/grid_export_kwh"]
    daily = pd.Series(nothing - lp, index=ts).groupby(ts.normalize()).sum()
    mid = (daily - daily.median()).abs().idxmin()
    first = min(max(mid - pd.Timedelta(days=1), daily.index[0]), daily.index[-3])
    return ts, first


def fig_example():
    tr = np.load("results/baseline_trajectories.npz")
    house = int(tr["house"])
    ts, first = pick_typical_days(tr, house)
    sel = (ts >= first) & (ts < first + pd.Timedelta(days=3))
    x = np.arange(sel.sum())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, gridspec_kw=dict(height_ratios=[1, 2]))
    a1.fill_between(x, tr["lp_oracle/price"][sel], step="post", color=GRID, zorder=1)
    a1.step(x, tr["lp_oracle/price"][sel], where="post", color=MUTED, lw=1.5)
    a1.set_ylabel("price, won/kWh"); a1.set_ylim(100, 240)
    a1.set_title(TITLE_EXAMPLE)
    ends = {}
    for p in ["self_consumption_causal", "rule_based", "lp_oracle"]:
        label, color, marker = POLICY_STYLE[p]
        y = tr[f"{p}/soc_kwh"][sel]
        a2.plot(x, y, color=color, marker=marker, markevery=6, ms=6, label=label, zorder=3)
        ends[p] = y[-1]
    # direct end labels, spread apart when they would overlap
    placed = []
    for p, yv in sorted(ends.items(), key=lambda kv: kv[1]):
        while any(abs(yv - q) < 0.55 for q in placed):
            yv += 0.55
        placed.append(yv)
        a2.text(x[-1] + 0.8, yv, POLICY_STYLE[p][0].split(" (")[0], color=INK, fontsize=9, va="center")
    a2.axhline(1.0, color=MUTED, lw=0.8, ls=":"); a2.axhline(9.0, color=MUTED, lw=0.8, ls=":")
    a2.set_ylabel("battery charge, kWh"); a2.set_ylim(0, 10.5)
    a2.set_xticks(range(0, 72, 12)); a2.set_xticklabels([f"{(h % 24):02d}:00" for h in range(0, 72, 12)])
    last = first + pd.Timedelta(days=2)
    a2.set_xlabel(f"{first:%d %b} – {last:%d %b %Y}, home {house} (hour of day)"); a2.set_xlim(0, 72)
    a2.legend(loc="upper left", ncol=3)
    fig.text(0.125, -0.03, "Days chosen by a fixed rule: the middle day is the test day whose oracle saving is closest to the "
             "median day. Dotted lines: usable range 1 – 9 kWh.", color=INK2, fontsize=8.5)
    save(fig, "example_days.png")


TITLE_EXAMPLE = "The oracle buys cheap energy before 08:00 and uses it during the day; the causal rules barely use the battery"


if __name__ == "__main__":
    fig_loop(); fig_tariff(); fig_data(); fig_baselines(); fig_example()
