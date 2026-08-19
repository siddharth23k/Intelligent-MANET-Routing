"""The paper's adaptive threshold heuristic.

This supplies the regression target for the SFRNNR's threshold head. It is a
hand written logistic combination of six normalised link factors.

Bug this file used to contain, kept in the docstring because it is worth
remembering: the lookups were written as "RSSI_norm", "LS_norm", "LET_norm",
"LL_d_norm" and "ND_norm". No stage of the pipeline ever produced a column with
a `_norm` suffix, so `dict.get(key, 0.5)` silently returned the default for five
of the six inputs and the "adaptive" threshold was effectively a function of one
variable. A defensive `.get` with a default turned a schema mismatch into a
silent wrong answer. The keys are now the real column names, and missing keys
raise instead of defaulting.
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

# Coefficients as described in the paper: link quality, signal strength and link
# stability push the threshold up (tolerate more predicted risk), while latency
# and a sparse neighbourhood push it down.
WEIGHTS = {
    "LQ_mean": 1.8,
    "RSSI": 1.2,
    "LS": 1.0,
    "LET": 0.8,
    "LL_d": -1.0,
}
SPARSITY_WEIGHT = -0.8          # applied to (1 - ND)
THRESHOLD_BASE = 0.35
THRESHOLD_SPAN = 0.30
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80


class MissingThresholdInput(KeyError):
    pass


class AdaptiveThresholdModel:
    """All inputs must already be min max normalised into [0, 1]."""

    REQUIRED_INPUTS = tuple(THRESHOLD_INPUTS)

    @staticmethod
    def _z(values: Mapping[str, float]):
        z = 0.0
        for key, w in WEIGHTS.items():
            z = z + w * values[key]
        return z + SPARSITY_WEIGHT * (1.0 - values["ND"])

    @classmethod
    def predict_threshold(cls, feature_row: Mapping[str, float], strict: bool = True) -> float:
        missing = [k for k in cls.REQUIRED_INPUTS if k not in feature_row]
        if missing and strict:
            raise MissingThresholdInput(
                f"adaptive threshold needs {list(cls.REQUIRED_INPUTS)}; missing {missing}. "
                "Run pipeline/engineer_features.py so the normalised factors exist."
            )
        values = {k: float(feature_row.get(k, 0.5)) for k in cls.REQUIRED_INPUTS}
        z = cls._z(values)
        s = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(THRESHOLD_BASE + THRESHOLD_SPAN * s, THRESHOLD_MIN, THRESHOLD_MAX))

    @classmethod
    def predict_threshold_frame(cls, frame) -> np.ndarray:
        """Vectorised path. `frame` must expose every required column."""
        missing = [k for k in cls.REQUIRED_INPUTS if k not in frame.columns]
        if missing:
            raise MissingThresholdInput(
                f"adaptive threshold needs {list(cls.REQUIRED_INPUTS)}; missing {missing}"
            )
        values = {k: frame[k].to_numpy(dtype=float) for k in cls.REQUIRED_INPUTS}
        z = cls._z(values)
        s = 1.0 / (1.0 + np.exp(-z))
        return np.clip(THRESHOLD_BASE + THRESHOLD_SPAN * s, THRESHOLD_MIN, THRESHOLD_MAX).astype(np.float32)

    @staticmethod
    def predict_threshold_batch(lq, rssi, ls, let, ll_d, nd) -> np.ndarray:
        z = (
            WEIGHTS["LQ_mean"] * np.asarray(lq, dtype=float)
            + WEIGHTS["RSSI"] * np.asarray(rssi, dtype=float)
            + WEIGHTS["LS"] * np.asarray(ls, dtype=float)
            + WEIGHTS["LET"] * np.asarray(let, dtype=float)
            + WEIGHTS["LL_d"] * np.asarray(ll_d, dtype=float)
            + SPARSITY_WEIGHT * (1.0 - np.asarray(nd, dtype=float))
        )
        s = 1.0 / (1.0 + np.exp(-z))
        return np.clip(THRESHOLD_BASE + THRESHOLD_SPAN * s, THRESHOLD_MIN, THRESHOLD_MAX).astype(np.float32)
