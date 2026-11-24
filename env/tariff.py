"""Time-of-use tariff: maps a timestamp to an energy price in won/kWh."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DAYS_IN_MONTH = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])  # typical (non-leap) year


@dataclass(frozen=True)
class Tariff:
    name: str
    source: str
    table: np.ndarray  # shape (13, 24): table[month, hour] in won/kWh (row 0 unused)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        # hour -> band, every hour must belong to exactly one band
        band_of_hour: dict[int, str] = {}
        for band, ranges in cfg["bands"].items():
            for start, end in ranges:
                for h in range(start, end):
                    if h in band_of_hour:
                        raise ValueError(f"hour {h} is in two bands")
                    band_of_hour[h] = band
        if sorted(band_of_hour) != list(range(24)):
            raise ValueError("bands must cover all 24 hours exactly once")

        # month -> price per hour, every month must belong to exactly one season
        table = np.full((13, 24), np.nan)
        for season in cfg["seasons"].values():
            for m in season["months"]:
                if not np.isnan(table[m, 0]):
                    raise ValueError(f"month {m} is in two seasons")
                table[m, :] = [season["prices"][band_of_hour[h]] for h in range(24)]
        if np.isnan(table[1:]).any():
            raise ValueError("seasons must cover all 12 months exactly once")
        return cls(name=cfg["name"], source=cfg["source"], table=table)

    def price_array(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Price for each hourly timestamp (vectorised table lookup)."""
        return self.table[index.month.to_numpy(), index.hour.to_numpy()]

    @property
    def max_price(self) -> float:
        return float(self.table[1:].max())

    @property
    def mean_price(self) -> float:
        """Average price over a typical year (months weighted by their number of days)."""
        per_month = self.table[1:].mean(axis=1)
        return float((per_month * DAYS_IN_MONTH).sum() / DAYS_IN_MONTH.sum())
