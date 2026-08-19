"""Min max statistics fitted on training rows only.

Computing min and max over the whole frame lets the held out runs influence the
scaling applied to training rows. Small leak, trivial to remove.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


class MinMaxStats:
    def __init__(self, stats: Dict[str, Dict[str, float]] | None = None):
        self.stats: Dict[str, Dict[str, float]] = stats or {}

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: Sequence[str]) -> "MinMaxStats":
        stats: Dict[str, Dict[str, float]] = {}
        for column in columns:
            values = pd.to_numeric(df[column], errors="coerce").astype(float).to_numpy()
            lo = float(np.nanmin(values)) if values.size else 0.0
            hi = float(np.nanmax(values)) if values.size else 0.0
            stats[column] = {"min": lo, "max": hi}
        return cls(stats)

    def transform_series(self, series: pd.Series, column: str) -> pd.Series:
        if column not in self.stats:
            raise KeyError(f"no statistics for '{column}'. Known: {sorted(self.stats)}")
        lo, hi = self.stats[column]["min"], self.stats[column]["max"]
        series = pd.to_numeric(series, errors="coerce").astype(float)
        if np.isclose(hi - lo, 0.0):
            return pd.Series(np.zeros(len(series)), index=series.index)
        # Clip so held out rows outside the training range stay inside [0, 1].
        return ((series - lo) / (hi - lo)).clip(0.0, 1.0)

    def transform(self, df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
        out = df.copy()
        for column in columns if columns is not None else list(self.stats):
            out[column] = self.transform_series(out[column], column)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.stats, handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxStats":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def columns(self) -> List[str]:
        return sorted(self.stats)
