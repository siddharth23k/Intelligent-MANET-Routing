"""The single source of truth for feature names and column contracts.

Every place that builds a feature matrix imports from here. That is what stops
the training code and the routing code from disagreeing about column order,
which is the failure mode that produced a scaler and a model that did not
match each other.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

# --- our predictor ---------------------------------------------------------
# Order matters. The scaler and both models are fitted in exactly this order and
# assert_feature_frame below is the only sanctioned way to build the matrix.
FEATURES: List[str] = [
    "neighbor_count",
    "x",
    "y",
    "time",
    "avg_rssi",
    "dist_to_center",
    "rssi_velocity",
    "neighbor_velocity",
    "pdr",
    "log_delay",
    "rssi_trend_3",
    "neighbor_trend_3",
    "rssi_std_5",
    "neighbor_std_5",
]

TARGET = "link_failure"

# Traffic derived features. They come from NS-3 FlowMonitor. When the raw data
# only carries end of run aggregates these are not causal at time t, so the
# dataset validator refuses to pass unless the caller explicitly opts in.
TRAFFIC_FEATURES: List[str] = ["pdr", "log_delay"]

# --- the paper baseline ----------------------------------------------------
# The nine link factors described in the base paper, in the order the SFRNNR
# consumes them.
FACTOR_COLS: List[str] = [
    "d_res",
    "LET",
    "ND",
    "RSSI",
    "LS",
    "LA",
    "LQ_mean",
    "LL_d",
    "T_hello",
]

# Columns the paper baseline's adaptive threshold heuristic reads. Named
# explicitly because an earlier version looked up "<name>_norm" keys that never
# existed and silently fell back to a default for five of six inputs.
THRESHOLD_INPUTS: List[str] = ["LQ_mean", "RSSI", "LS", "LET", "LL_d", "ND"]

IDENTIFIER_COLS: List[str] = ["run_id", "time", "node_id"]


class SchemaError(ValueError):
    """Raised when a frame or matrix does not match the declared contract."""


def assert_columns(frame, required: Sequence[str], where: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"{where}: missing required columns {missing}. "
            f"Present columns: {sorted(frame.columns)[:40]}"
        )


def feature_matrix(frame, features: Sequence[str] = None) -> np.ndarray:
    """Build the model input matrix in the canonical column order."""
    features = list(features or FEATURES)
    assert_columns(frame, features, "feature_matrix")
    return frame[features].to_numpy(dtype=float)


def assert_matrix_shape(X: np.ndarray, expected: int, where: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != expected:
        raise SchemaError(
            f"{where}: expected {expected} columns, got {X.shape[1]}. "
            f"Canonical order is {FEATURES}"
        )
    return X


def constant_columns(frame, columns: Iterable[str]) -> List[str]:
    """Return the subset of `columns` that carry no information at all."""
    dead = []
    for c in columns:
        if c not in frame.columns:
            continue
        s = frame[c]
        if s.isna().all() or s.nunique(dropna=True) <= 1:
            dead.append(c)
    return dead
