"""Load the hourly household data and attach tariff prices and calendar features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from env.tariff import Tariff

PRICE_LOOKAHEAD = 6  # price now + next 5 hours are part of the observation
PAD_HOURS = 24       # arrays are padded so lookahead past the last data hour stays in range
DEFAULT_CSV = "data/processed/hourly.csv.gz"
SPLIT_NAMES = ("train", "val", "test")


@dataclass
class HouseData:
    """One household. Time series cover all 3 years; `segments` are the hour ranges of the requested split.

    The full series is kept (not only the split) so that features looking back in time
    (previous hour, same hour yesterday) are available at the start of a segment. They only
    use PAST data, so nothing from the future leaks in.
    """

    house_id: int
    start_time: pd.Timestamp   # timestamp of index 0
    load: np.ndarray    # kW (== kWh per hour), length N
    pv: np.ndarray      # kW, length N
    price: np.ndarray   # won/kWh, length N + PAD_HOURS
    hour: np.ndarray    # 0..23, length N + PAD_HOURS
    doy: np.ndarray     # day of year, length N + PAD_HOURS
    weekend: np.ndarray  # 1.0 on Sat/Sun, length N + PAD_HOURS
    segments: list[tuple[int, int]]  # [start, end) hour indices; each starts at midnight

    def timestamps(self, idx) -> pd.DatetimeIndex:
        return self.start_time + pd.to_timedelta(np.asarray(idx), unit="h")


def load_data_config(path: str | Path = "configs/data.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def split_days(split: str, cfg: dict) -> pd.DatetimeIndex:
    """Calendar days belonging to a split.

    train: development period minus the validation days
    val:   days d1..d2 of every month of the development period
    test:  the test period
    """
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    s = cfg["splits"]
    if split == "test":
        return pd.date_range(s["test"]["start"], s["test"]["end"], freq="D")
    dev = pd.date_range(s["development"]["start"], s["development"]["end"], freq="D")
    d1, d2 = s["validation_days_of_month"]
    is_val = (dev.day >= d1) & (dev.day <= d2)
    return dev[is_val] if split == "val" else dev[~is_val]


def days_to_segments(days: pd.DatetimeIndex, start_time: pd.Timestamp) -> list[tuple[int, int]]:
    """Group consecutive days into [start, end) hour-index ranges."""
    d = ((days - start_time) // pd.Timedelta(days=1)).to_numpy()
    breaks = np.flatnonzero(np.diff(d) != 1) + 1
    return [(int(run[0]) * 24, (int(run[-1]) + 1) * 24) for run in np.split(d, breaks)]


def load_houses(
    house_ids: list[int],
    split: str,
    tariff: Tariff,
    csv_path: str | Path = DEFAULT_CSV,
    data_cfg: dict | None = None,
) -> dict[int, HouseData]:
    cfg = data_cfg or load_data_config()
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    days = split_days(split, cfg)

    houses: dict[int, HouseData] = {}
    for hid in house_ids:
        g = df[df["customer"] == hid].sort_values("timestamp")
        if g.empty:
            raise ValueError(f"house {hid} not found in {csv_path}")
        ts = pd.DatetimeIndex(g["timestamp"])
        if not (ts[1:] - ts[:-1] == pd.Timedelta(hours=1)).all() or ts[0].hour != 0:
            raise ValueError(f"house {hid}: not a regular hourly series starting at midnight")
        ts_ext = ts.append(pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=PAD_HOURS, freq="h"))
        segments = days_to_segments(days, ts[0])
        if segments[0][0] < 0 or segments[-1][1] > len(ts):
            raise ValueError("split lies outside the data")
        houses[hid] = HouseData(
            house_id=hid,
            start_time=ts[0],
            load=g["load_kw"].to_numpy(dtype=np.float64),
            pv=g["pv_kw"].to_numpy(dtype=np.float64),
            price=tariff.price_array(ts_ext),
            hour=ts_ext.hour.to_numpy(),
            doy=ts_ext.day_of_year.to_numpy(),
            weekend=(ts_ext.day_of_week.to_numpy() >= 5).astype(np.float64),
            segments=segments,
        )
    return houses
