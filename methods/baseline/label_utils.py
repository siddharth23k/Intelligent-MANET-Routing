"""Shared link-failure label logic (paper pipeline)."""

import numpy as np
import pandas as pd

RSSI_SENTINEL = -1000.0
DEFAULT_HORIZON = 5


def add_link_failure_labels(df: pd.DataFrame, horizon: int = DEFAULT_HORIZON) -> pd.DataFrame:
    df = df.copy()
    df["f_neighbors"] = df.groupby(["run_id", "node_id"])["neighbor_count"].shift(-horizon)
    df["f_rssi"] = df.groupby(["run_id", "node_id"])["avg_rssi"].shift(-horizon)
    cond1 = df["neighbor_count"] - df["f_neighbors"] >= 2
    cond2 = (df["avg_rssi"] - df["f_rssi"] >= 15.0) & (df["f_rssi"] > RSSI_SENTINEL)
    cond3 = df["f_rssi"] == RSSI_SENTINEL
    df["link_failure"] = (cond1 | cond2 | cond3).astype(int)
    return df


def drop_label_aux_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["f_neighbors", "f_rssi"], errors="ignore")
