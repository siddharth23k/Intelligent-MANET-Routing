"""The paper's adaptive threshold heuristic.

Supplies the regression target for the SFRNNR's threshold head: a logistic
combination of six normalised link factors.

The lookups used to be "<name>_norm", which no pipeline stage ever produced, so
`dict.get(key, 0.5)` silently defaulted five of six inputs and the "adaptive"
threshold varied with one variable. Keys are now the real column names and
missing keys raise.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from schema import THRESHOLD_INPUTS  # noqa: E402

# Link quality, signal strength and stability raise the threshold (tolerate more
# predicted risk); latency and a sparse neighbourhood lower it.
WEIGHTS = {"LQ_mean": 1.8, "RSSI": 1.2, "LS": 1.0, "LET": 0.8, "LL_d": -1.0}
SPARSITY_WEIGHT = -0.8          # applied to (1 - ND)
THRESHOLD_BASE = 0.35
THRESHOLD_SPAN = 0.30
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80


class MissingThresholdInput(KeyError):
    """A required normalised factor was not supplied."""


class AdaptiveThresholdModel:
    """All inputs must already be min max normalised into [0, 1]."""

    REQUIRED_INPUTS = tuple(THRESHOLD_INPUTS)

    @staticmethod
    def _logit(values: Mapping[str, float]):
        z = sum(weight * values[key] for key, weight in WEIGHTS.items())
        return z + SPARSITY_WEIGHT * (1.0 - values["ND"])

    @classmethod
    def _squash(cls, z):
        return np.clip(
            THRESHOLD_BASE + THRESHOLD_SPAN / (1.0 + np.exp(-z)), THRESHOLD_MIN, THRESHOLD_MAX
        )

    @classmethod
    def predict_threshold(cls, feature_row: Mapping[str, float], strict: bool = True) -> float:
        missing = [k for k in cls.REQUIRED_INPUTS if k not in feature_row]
        if missing and strict:
            raise MissingThresholdInput(
                f"needs {list(cls.REQUIRED_INPUTS)}, missing {missing}. "
                "Run pipeline/engineer_features.py so the normalised factors exist."
            )
        values = {k: float(feature_row.get(k, 0.5)) for k in cls.REQUIRED_INPUTS}
        return float(cls._squash(cls._logit(values)))

    @classmethod
    def predict_threshold_frame(cls, frame) -> np.ndarray:
        """Vectorised path over a dataframe exposing every required column."""
        missing = [k for k in cls.REQUIRED_INPUTS if k not in frame.columns]
        if missing:
            raise MissingThresholdInput(f"needs {list(cls.REQUIRED_INPUTS)}, missing {missing}")
        values = {k: frame[k].to_numpy(dtype=float) for k in cls.REQUIRED_INPUTS}
        return cls._squash(cls._logit(values)).astype(np.float32)

    @classmethod
    def predict_threshold_batch(cls, lq, rssi, ls, let, ll_d, nd) -> np.ndarray:
        values = {
            "LQ_mean": np.asarray(lq, dtype=float),
            "RSSI": np.asarray(rssi, dtype=float),
            "LS": np.asarray(ls, dtype=float),
            "LET": np.asarray(let, dtype=float),
            "LL_d": np.asarray(ll_d, dtype=float),
            "ND": np.asarray(nd, dtype=float),
        }
        return cls._squash(cls._logit(values)).astype(np.float32)
