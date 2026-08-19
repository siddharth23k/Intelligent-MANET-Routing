"""The shared link failure label.

Both the paper baseline and our predictor train on this exact definition, which
is the whole basis of the "same data, same label, same split" comparison claim.
The thresholds live in config/paper_scenarios.yaml so neither method can drift.

Known limitation, stated here because it is the sharpest criticism of the whole
project: the label is a forward difference of quantities the model also observes
at time t. In particular `neighbor_count(t) - neighbor_count(t+H) >= drop` means
a node with a high degree is mechanically more likely to be labelled a failure,
and a node with zero neighbours can never satisfy it. `label_diagnostics` below
quantifies that coupling so it is visible in the reports rather than hidden.
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


def add_link_failure_labels(df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
    horizon = DEFAULT_HORIZON if horizon is None else int(horizon)
    if horizon < 1:
        raise ValueError(f"label horizon must be >= 1, got {horizon}")

    lab = CFG.labels
    neighbour_drop = float(lab["neighbour_drop"])
    rssi_drop = float(lab["rssi_drop_db"])

    df = df.copy()
    g = df.groupby(["run_id", "node_id"], sort=False)
    df["f_neighbors"] = g["neighbor_count"].shift(-horizon)
    df["f_rssi"] = g["avg_rssi"].shift(-horizon)

    isolated_now = df.get("is_isolated")
    future_isolated = (
        g["is_isolated"].shift(-horizon) if isolated_now is not None else None
    )

    lost_neighbours = (df["neighbor_count"] - df["f_neighbors"]) >= neighbour_drop
    rssi_collapse = (df["avg_rssi"] - df["f_rssi"]) >= rssi_drop
    if future_isolated is not None:
        goes_isolated = future_isolated.fillna(0).astype(bool)
    else:
        goes_isolated = df["f_rssi"] <= RSSI_SENTINEL

    df["link_failure"] = (
        lost_neighbours.fillna(False)
        | rssi_collapse.fillna(False)
        | goes_isolated
    ).astype(int)
    return df


def drop_label_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["f_neighbors", "f_rssi"], errors="ignore")


def label_diagnostics(df: pd.DataFrame, horizon: int = None) -> dict:
    """Quantify how much of the label is implied by features visible at time t."""
    labelled = add_link_failure_labels(df, horizon=horizon)
    y = labelled["link_failure"].to_numpy(dtype=float)
    out = {"label_rate": float(y.mean()), "horizon": DEFAULT_HORIZON if horizon is None else horizon}
    for col in ("neighbor_count", "avg_rssi"):
        if col in labelled.columns:
            x = labelled[col].to_numpy(dtype=float)
            if np.nanstd(x) > 0 and y.std() > 0:
                out[f"corr_{col}_vs_label"] = float(np.corrcoef(x, y)[0, 1])
    # A node with zero neighbours cannot lose `neighbour_drop` neighbours.
    if "neighbor_count" in labelled.columns:
        isolated = labelled[labelled["neighbor_count"] == 0]
        if len(isolated):
            out["label_rate_when_isolated"] = float(isolated["link_failure"].mean())
        dense = labelled[labelled["neighbor_count"] >= labelled["neighbor_count"].median()]
        if len(dense):
            out["label_rate_when_dense"] = float(dense["link_failure"].mean())
    return out
