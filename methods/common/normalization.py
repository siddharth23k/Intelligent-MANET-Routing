"""Min max normalisation statistics fitted on training runs only.

The earlier pipeline normalised with statistics computed over the whole
dataframe, which meant the minimum and maximum of the held out runs leaked into
the scaling applied to the training rows. It is a small leak, two order
statistics per column, but it is still a leak and it is trivial to remove: fit
on the training runs, persist the numbers, apply them everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd


class MinMaxStats:
    """Per column (min, max) fitted on a chosen subset of rows."""

    def __init__(self, stats: Dict[str, Dict[str, float]] | None = None):
        self.stats: Dict[str, Dict[str, float]] = stats or {}

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Sequence[str]) -> "MinMaxStats":
        stats: Dict[str, Dict[str, float]] = {}
        for c in columns:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            lo = float(np.nanmin(s.to_numpy())) if len(s) else 0.0
            hi = float(np.nanmax(s.to_numpy())) if len(s) else 0.0
            stats[c] = {"min": lo, "max": hi}
        return cls(stats)

    def transform_series(self, s: pd.Series, column: str) -> pd.Series:
        if column not in self.stats:
            raise KeyError(
                f"no normalisation statistics for '{column}'. "
                f"Known columns: {sorted(self.stats)}"
            )
        lo = self.stats[column]["min"]
        hi = self.stats[column]["max"]
        s = pd.to_numeric(s, errors="coerce").astype(float)
        if np.isclose(hi - lo, 0.0):
            return pd.Series(np.zeros(len(s)), index=s.index)
        # Clip so held out rows outside the training range cannot leave [0, 1].
        return ((s - lo) / (hi - lo)).clip(0.0, 1.0)

    def transform(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        out = df.copy()
        for c in (columns if columns is not None else self.stats.keys()):
            out[c] = self.transform_series(out[c], c)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxStats":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def columns(self) -> list:
        return sorted(self.stats)
