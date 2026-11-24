"""Explore the raw Ausgrid CSVs: per-customer coverage, PV size, consumption size."""
import sys
from pathlib import Path
import pandas as pd

RAW = Path("data/raw/ausgrid/extracted")
# the three yearly files use different date formats
FILES = {
    "Solar home 2010-2011.csv": "%d-%b-%y",
    "Solar home 2011-2012.csv": "%d/%m/%Y",
    "Solar home 2012-2013.csv": "%d/%m/%Y",
}


def load_year(fname: str, date_fmt: str) -> pd.DataFrame:
    # line 1 is a title note, line 2 is the real header
    df = pd.read_csv(RAW / fname, skiprows=1)
    df["date"] = pd.to_datetime(df["date"], format=date_fmt)  # raises if any row mismatches
    return df


def main() -> None:
    frames = [load_year(f, fmt) for f, fmt in FILES.items()]
    df = pd.concat(frames, ignore_index=True)
    print("rows:", len(df), "| customers:", df["Customer"].nunique())
    print("categories:", df["Consumption Category"].value_counts().to_dict())
    print("bad dates:", int(df["date"].isna().sum()))
    time_cols = [c for c in df.columns if ":" in c]
    print("time columns:", len(time_cols), time_cols[:2], "...", time_cols[-2:])

    nan_rate = df[time_cols].isna().mean().mean()
    print(f"overall NaN rate in readings: {nan_rate:.4%}")

    # per customer summary using GC (consumption) and GG (PV)
    gc = df[df["Consumption Category"] == "GC"]
    gg = df[df["Consumption Category"] == "GG"]
    days_gc = gc.groupby("Customer")["date"].nunique()
    days_gg = gg.groupby("Customer")["date"].nunique()
    cap = df.groupby("Customer")["Generator Capacity"].first()
    # kWh per day = sum of 48 half-hour kWh readings
    gc_day = gc.assign(kwh=gc[time_cols].sum(axis=1)).groupby("Customer")["kwh"].mean()
    gg_day = gg.assign(kwh=gg[time_cols].sum(axis=1)).groupby("Customer")["kwh"].mean()
    nan_gc = gc.groupby("Customer")[time_cols].apply(lambda x: x.isna().mean().mean())

    summ = pd.DataFrame({"pv_kWp": cap, "days_GC": days_gc, "days_GG": days_gg,
                         "load_kWh_day": gc_day, "pv_kWh_day": gg_day, "nan_GC": nan_gc})
    summ["pv_to_load"] = summ["pv_kWh_day"] / summ["load_kWh_day"]
    out = Path("data/processed/customer_summary.csv")
    summ.to_csv(out)
    print(summ.describe().round(2).to_string())
    print("saved", out)


if __name__ == "__main__":
    main()
