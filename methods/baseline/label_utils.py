"""The shared link failure label.

Both methods train on this exact definition, which is the basis of the "same
data, same label, same split" comparison. Thresholds live in the config so
neither method can drift.

Known coupling: the label is a forward difference of `neighbor_count`, which the
model also observes at time t. A dense node is mechanically more likely to be
labelled a failure and an isolated node can never satisfy that condition.
`label_diagnostics` quantifies it so it appears in reports instead of hiding
inside a feature importance number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from config_loader import get_config  # noqa: E402

CFG = get_config()
RSSI_SENTINEL = CFG.rssi_sentinel
RSSI_FLOOR = CFG.rssi_floor
DEFAULT_HORIZON = CFG.label_horizon


def add_link_failure_labels(df: pd.DataFrame, horizon: int | None = None) -> pd.DataFrame:
    horizon = DEFAULT_HORIZON if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError(f"label horizon must be >= 1, got {horizon}")

    neighbour_drop = float(CFG.labels["neighbour_drop"])
    rssi_drop = float(CFG.labels["rssi_drop_db"])

    df = df.copy()
    grouped = df.groupby(["run_id", "node_id"], sort=False)
    df["f_neighbors"] = grouped["neighbor_count"].shift(-horizon)
    df["f_rssi"] = grouped["avg_rssi"].shift(-horizon)

    lost_neighbours = (df["neighbor_count"] - df["f_neighbors"]) >= neighbour_drop
    rssi_collapse = (df["avg_rssi"] - df["f_rssi"]) >= rssi_drop

    # Prefer the explicit isolation flag when engineer_features produced it,
    # because the RSSI sentinel is replaced with a physical floor by then.
    if "is_isolated" in df.columns:
        goes_isolated = grouped["is_isolated"].shift(-horizon).fillna(0).astype(bool)
    else:
        goes_isolated = df["f_rssi"] <= RSSI_SENTINEL

    df["link_failure"] = (
        lost_neighbours.fillna(False) | rssi_collapse.fillna(False) | goes_isolated
    ).astype(int)
    return df


def drop_label_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["f_neighbors", "f_rssi"], errors="ignore")


def label_diagnostics(df: pd.DataFrame, horizon: int | None = None) -> dict:
    """Measure how much of the label is implied by features visible at time t."""
    labelled = add_link_failure_labels(df, horizon=horizon)
    y = labelled["link_failure"].to_numpy(dtype=float)
    out = {
        "label_rate": float(y.mean()),
        "horizon": DEFAULT_HORIZON if horizon is None else int(horizon),
    }

    for column in ("neighbor_count", "avg_rssi"):
        if column in labelled.columns:
            x = labelled[column].to_numpy(dtype=float)
            if np.nanstd(x) > 0 and y.std() > 0:
                out[f"corr_{column}_vs_label"] = float(np.corrcoef(x, y)[0, 1])

    if "neighbor_count" in labelled.columns:
        isolated = labelled[labelled["neighbor_count"] == 0]
        if len(isolated):
            out["label_rate_when_isolated"] = float(isolated["link_failure"].mean())
        dense = labelled[labelled["neighbor_count"] >= labelled["neighbor_count"].median()]
        if len(dense):
            out["label_rate_when_dense"] = float(dense["link_failure"].mean())
    return out
