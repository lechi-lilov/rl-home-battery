"""Build the hourly dataset used by the environment.

Steps
1. Load the three yearly Ausgrid CSVs (wide format: 48 half-hour columns per row).
2. Pick households with a documented, data-independent rule + fixed seeds.
3. load = GC (+ CL when the household has a controlled-load channel); pv = GG.
4. Aggregate 2 half-hours -> 1 hour. kWh per hour equals mean kW.
5. Write data/processed/hourly.csv.gz and configs/data.yaml (houses + time splits).

Usage:  python scripts/01_preprocess.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAW = Path("data/raw/ausgrid/extracted")
OUT_CSV = Path("data/processed/hourly.csv.gz")
OUT_CFG = Path("configs/data.yaml")
SEED = 42            # draws the 10 training homes and the first 4 held-out homes
SEED_EXTRA = 43      # draws 20 more held-out homes from the remaining eligible ones
N_POOL, N_HELDOUT, N_EXTRA_HELDOUT = 10, 4, 20
FILES = {
    "Solar home 2010-2011.csv": "%d-%b-%y",
    "Solar home 2011-2012.csv": "%d/%m/%Y",
    "Solar home 2012-2013.csv": "%d/%m/%Y",
}
# Time split. The development period is divided into training and validation by a fixed calendar
# rule: days 8-14 of EVERY month are validation, so validation covers all seasons. The test year
# comes after the whole development period and is never used for any choice.
SPLITS = {
    "development": {"start": "2010-07-01", "end": "2012-06-30"},
    "validation_days_of_month": [8, 14],
    "test": {"start": "2012-07-01", "end": "2013-06-30"},
}
FULL_DAYS = pd.date_range("2010-07-01", "2013-06-30", freq="D")  # 1096 days


def load_all() -> pd.DataFrame:
    frames = []
    for fname, fmt in FILES.items():
        d = pd.read_csv(RAW / fname, skiprows=1)  # row 1 is a title note
        d["date"] = pd.to_datetime(d["date"], format=fmt)  # raises on mismatch
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def channel(df: pd.DataFrame, cust: int, cat: str, time_cols: list[str]):
    """Return a (days, 48) array for one channel, or None if the channel is absent."""
    sub = df[(df["Customer"] == cust) & (df["Consumption Category"] == cat)]
    if sub.empty:
        return None
    assert sub["date"].is_unique, f"duplicate dates: customer {cust} {cat}"
    sub = sub.set_index("date")[time_cols].reindex(FULL_DAYS)
    return sub.to_numpy(dtype=float)  # NaN rows = missing days


def to_hourly(a: np.ndarray) -> np.ndarray:
    # column j covers the half hour [j*30min, (j+1)*30min); hour h = columns 2h, 2h+1
    return a.reshape(len(a), 24, 2).sum(axis=2).reshape(-1)


def main() -> None:
    df = load_all()
    time_cols = [c for c in df.columns if ":" in c]
    assert len(time_cols) == 48 and time_cols[0] == "0:30" and time_cols[-1] == "0:00"

    # per-customer day counts and daily load, for the selection rule
    # total household load = GC + CL, so the rule must use both channels
    gc = df[df["Consumption Category"] == "GC"]
    cl = df[df["Consumption Category"] == "CL"]
    days = gc.groupby("Customer")["date"].nunique()
    cl_days = cl.groupby("Customer")["date"].nunique().reindex(days.index)  # NaN = no CL channel
    gc_day = gc.assign(k=gc[time_cols].sum(axis=1)).groupby("Customer")["k"].mean()
    cl_day = cl.assign(k=cl[time_cols].sum(axis=1)).groupby("Customer")["k"].mean().reindex(days.index).fillna(0.0)
    load_day = gc_day + cl_day
    lo, hi = load_day.quantile([0.25, 0.75])
    complete = lambda c: days[c] == len(FULL_DAYS) and (np.isnan(cl_days[c]) or cl_days[c] == len(FULL_DAYS))
    eligible = sorted(c for c in days.index if complete(c) and lo <= load_day[c] <= hi)
    print(f"eligible houses (full {len(FULL_DAYS)} days, GC+CL load {lo:.1f}-{hi:.1f} kWh/day): {len(eligible)}")

    rng = np.random.default_rng(SEED)
    chosen = [int(c) for c in rng.choice(eligible, size=N_POOL + N_HELDOUT, replace=False)]
    pool, heldout = chosen[:N_POOL], chosen[N_POOL:]
    remaining = [c for c in eligible if c not in chosen]
    extra = [int(c) for c in np.random.default_rng(SEED_EXTRA).choice(remaining, size=N_EXTRA_HELDOUT, replace=False)]
    heldout = heldout + extra
    chosen = pool + heldout

    cols = {}
    rows = []
    for c in chosen:
        gcA, ggA, clA = (channel(df, c, k, time_cols) for k in ("GC", "GG", "CL"))
        load = gcA + (clA if clA is not None else 0.0)
        pv = ggA
        assert not np.isnan(load).any() and not np.isnan(pv).any(), f"NaN in customer {c}"
        cap = float(df.loc[df["Customer"] == c, "Generator Capacity"].iloc[0])
        h_load, h_pv = to_hourly(load), to_hourly(pv)
        ts = pd.date_range(FULL_DAYS[0], periods=len(h_load), freq="h")
        rows.append(pd.DataFrame({"timestamp": ts, "customer": c, "load_kw": h_load, "pv_kw": h_pv}))
        cols[c] = {"pv_kwp": cap, "has_controlled_load": clA is not None}

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_CSV, index=False, float_format="%.4f")  # source data has 3 decimals per half hour

    # sanity checks printed for the report
    out["hour"] = out["timestamp"].dt.hour
    prof = out.groupby("hour")["pv_kw"].mean()
    print("PV peak hour:", int(prof.idxmax()), "| night PV (0-4h) mean kW:", round(prof.loc[0:4].mean(), 4))
    print("load peak hour:", int(out.groupby("hour")["load_kw"].mean().idxmax()))
    print("rows:", len(out), "| pool:", pool, "| held-out:", heldout)

    cfg = {
        "source": "Ausgrid Solar Home Electricity Data (CC BY 3.0 AU), 2010-07-01..2013-06-30",
        "selection_rule": (f"all channels complete for {len(FULL_DAYS)} days and daily GC+CL load within the "
                           f"25-75th percentile of all 300 houses ({len(eligible)} eligible). "
                           f"rng seed {SEED}: {N_POOL} training + {N_HELDOUT} held-out homes; "
                           f"rng seed {SEED_EXTRA}: {N_EXTRA_HELDOUT} more held-out homes from the rest"),
        "resolution": "hourly (kWh per hour == mean kW)",
        "load_definition": "GC + CL (CL only where present); pv = GG",
        "pool_houses": pool,
        "heldout_houses": heldout,
        "houses": {int(k): v for k, v in cols.items()},
        "splits": SPLITS,
    }
    OUT_CFG.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("wrote", OUT_CSV, OUT_CFG)


if __name__ == "__main__":
    main()
