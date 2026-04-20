import os

import numpy as np
import pandas as pd

INPUT_FILE = "data/processed/paper_raw_dataset.csv"
OUTPUT_FILE = "data/processed/paper_featured_dataset.csv"

RSSI_SENTINEL = -1000.0
COMM_RADIUS = 150.0
HELLO_INTERVAL = 1.0
ALPHA = 10.0
LAMBDA_EPOCH = 0.05


def _safe_norm(s):
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if np.isclose(hi - lo, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run generate_data.py first.")

    df = pd.read_csv(INPUT_FILE).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    df["distance_from_center"] = np.sqrt((df["x"] - 250.0) ** 2 + (df["y"] - 250.0) ** 2)
    df["dist_to_center"] = df["distance_from_center"]
    df["d_res"] = np.clip(COMM_RADIUS - df["distance_from_center"], 0.0, COMM_RADIUS)

    df["ND"] = df["neighbor_count"].astype(float)

    g = df.groupby(["run_id", "node_id"])
    df["neighbor_delta"] = g["neighbor_count"].transform(lambda s: s.diff().fillna(0))
    df["rssi_velocity"] = g["avg_rssi"].transform(lambda s: s.shift(1).diff()).fillna(0)
    df["neighbor_velocity"] = g["neighbor_count"].transform(lambda s: s.shift(1).diff()).fillna(0)
    df["pdr"] = np.where(df["tx_packets"] > 0, df["rx_packets"] / df["tx_packets"], 1.0)
    df["log_delay"] = np.log1p(df["delay_sum"])
    df["rssi_trend_3"] = g["rssi_velocity"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df["neighbor_trend_3"] = g["neighbor_velocity"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df["rssi_std_5"] = g["avg_rssi"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
    df["neighbor_std_5"] = g["neighbor_count"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)

    df["LET"] = g["d_res"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(method="bfill"))
    df["LS"] = g["avg_rssi"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(method="bfill"))
    df["RSSI"] = _safe_norm(df["avg_rssi"])
    df["LA"] = g["avg_rssi"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std().fillna(method="bfill"))
    df["LQ_mean"] = g["pdr"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(method="bfill"))
    df["LL_d"] = g["log_delay"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean().fillna(method="bfill"))
    df["T_hello"] = HELLO_INTERVAL

    out_cols = [
        "run_id",
        "time",
        "node_id",
        "x",
        "y",
        "neighbor_count",
        "avg_rssi",
        "tx_packets",
        "rx_packets",
        "lost_packets",
        "delay_sum",
        "dist_to_center",
        "rssi_velocity",
        "neighbor_velocity",
        "pdr",
        "log_delay",
        "rssi_trend_3",
        "neighbor_trend_3",
        "rssi_std_5",
        "neighbor_std_5",
        "d_res",
        "T_hello",
        "ND",
        "LET",
        "LS",
        "RSSI",
        "LA",
        "LQ_mean",
        "LL_d",
    ]
    df[out_cols].to_csv(OUTPUT_FILE, index=False)
    

if __name__ == "__main__":
    main()
