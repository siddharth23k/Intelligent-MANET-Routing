"""Feature names and column contracts, in one place.

Training, inference and routing all import from here. Two modules disagreeing
about column order is what let a scaler and a model be fitted on different data
without anything raising.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

# Order is part of the contract: the scaler and both models are fitted in
# exactly this order.
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

# Derived from NS-3 FlowMonitor. Not causal at time t when the raw data only
# carries end of run aggregates, so validate_dataset gates on them.
TRAFFIC_FEATURES: List[str] = ["pdr", "log_delay"]

# The paper's nine link factors, in the order the SFRNNR consumes them.
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

# Inputs to the paper's adaptive threshold. Named explicitly because an earlier
# version looked up "<name>_norm" keys that never existed.
THRESHOLD_INPUTS: List[str] = ["LQ_mean", "RSSI", "LS", "LET", "LL_d", "ND"]

IDENTIFIER_COLS: List[str] = ["run_id", "time", "node_id"]


class SchemaError(ValueError):
    """A frame or matrix does not match the declared contract."""


def assert_columns(frame, required: Sequence[str], where: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"{where}: missing columns {missing}. Present: {sorted(frame.columns)[:40]}"
        )


def feature_matrix(frame, features: Sequence[str] | None = None) -> np.ndarray:
    """Model input matrix in canonical column order."""
    features = list(features or FEATURES)
    assert_columns(frame, features, "feature_matrix")
    return frame[features].to_numpy(dtype=float)


def assert_matrix_shape(X: np.ndarray, expected: int, where: str) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != expected:
        raise SchemaError(
            f"{where}: expected {expected} columns, got {X.shape[1]}. Order: {FEATURES}"
        )
    return X


def constant_columns(frame, columns: Iterable[str]) -> List[str]:
    """Columns that are all null or carry a single value."""
    dead = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        if series.isna().all() or series.nunique(dropna=True) <= 1:
            dead.append(column)
    return dead
