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


@dataclass
class HouseData:
    """One household. Time series cover the whole 3 years; [i0, i1) is the requested split.

    The full series is kept (not only the split) so that features looking back in time
    (previous hour, same hour yesterday) are available at the start of a split. They only
    use PAST data, so nothing from the future leaks in.
    """

    house_id: int
    load: np.ndarray    # kW (== kWh per hour), length N
    pv: np.ndarray      # kW, length N
    price: np.ndarray   # won/kWh, length N + PAD_HOURS
    hour: np.ndarray    # 0..23, length N + PAD_HOURS
    doy: np.ndarray     # day of year, length N + PAD_HOURS
    weekend: np.ndarray  # 1.0 on Sat/Sun, length N + PAD_HOURS
    i0: int             # first hour index of the split
    i1: int             # one past the last hour index of the split


def load_data_config(path: str | Path = "configs/data.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_houses(
    house_ids: list[int],
    split: str,
    tariff: Tariff,
    csv_path: str | Path = "data/processed/hourly.csv",
    data_cfg: dict | None = None,
) -> dict[int, HouseData]:
    cfg = data_cfg or load_data_config()
    start, end = (pd.Timestamp(d) for d in cfg["splits"][split])
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    houses: dict[int, HouseData] = {}
    for hid in house_ids:
        g = df[df["customer"] == hid].sort_values("timestamp")
        ts = pd.DatetimeIndex(g["timestamp"])
        if not (ts[1:] - ts[:-1] == pd.Timedelta(hours=1)).all():
            raise ValueError(f"house {hid}: timestamps are not a regular hourly series")
        ts_ext = ts.append(pd.date_range(ts[-1] + pd.Timedelta(hours=1), periods=PAD_HOURS, freq="h"))
        houses[hid] = HouseData(
            house_id=hid,
            load=g["load_kw"].to_numpy(dtype=np.float64),
            pv=g["pv_kw"].to_numpy(dtype=np.float64),
            price=tariff.price_array(ts_ext),
            hour=ts_ext.hour.to_numpy(),
            doy=ts_ext.day_of_year.to_numpy(),
            weekend=(ts_ext.day_of_week.to_numpy() >= 5).astype(np.float64),
            i0=int(np.searchsorted(ts, start, side="left")),
            i1=int(np.searchsorted(ts, end + pd.Timedelta(days=1), side="left")),
        )
    return houses
