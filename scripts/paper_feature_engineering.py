import os

import numpy as np
import pandas as pd

INPUT_FILE = "dataset/paper/processed/paper_raw_dataset.csv"
OUTPUT_FILE = "dataset/paper/processed/paper_featured_dataset.csv"

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
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run paper_build_dataset.py first.")

    df = pd.read_csv(INPUT_FILE).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    # Paper-inspired feature proxies from available columns
    df["distance_from_center"] = np.sqrt((df["x"] - 250.0) ** 2 + (df["y"] - 250.0) ** 2)
    # Keep alias for compatibility with our existing model feature names.
    df["dist_to_center"] = df["distance_from_center"]
    df["d_res"] = np.clip(COMM_RADIUS - df["distance_from_center"], 0.0, COMM_RADIUS)

    # Node density proxy from neighborhood count
    df["ND"] = df["neighbor_count"].astype(float)

    # LET proxy: expected time before link edge (larger residual distance and slower change => larger LET)
    g = df.groupby(["run_id", "node_id"])
    df["neighbor_delta"] = g["neighbor_count"].transform(lambda s: s.diff().fillna(0))
    # Our model features (same schema as experiments/training.py).
    df["rssi_velocity"] = g["avg_rssi"].transform(lambda s: s.shift(1).diff()).fillna(0)
    df["neighbor_velocity"] = g["neighbor_count"].transform(lambda s: s.shift(1).diff()).fillna(0)
    df["pdr"] = np.where(df["tx_packets"] > 0, df["rx_packets"] / df["tx_packets"], 1.0)
    df["log_delay"] = np.log1p(df["delay_sum"])
    df["rssi_trend_3"] = g["rssi_velocity"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df["neighbor_trend_3"] = g["neighbor_velocity"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df["rssi_std_5"] = g["avg_rssi"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
    df["neighbor_std_5"] = g["neighbor_count"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)

    speed_proxy = np.abs(df["neighbor_delta"]) + 1.0
    df["LET"] = df["d_res"] / speed_proxy
    df["LS"] = 1.0 - np.exp(-df["LET"] / ALPHA)

    # RSSI sanitization
    df["RSSI"] = df["avg_rssi"].replace(RSSI_SENTINEL, -95.0)

    # Link availability / quality proxies
    df["LA"] = np.exp(-2.0 * LAMBDA_EPOCH * HELLO_INTERVAL)
    df["LQ_mean"] = _safe_norm(df["RSSI"] + 100.0)

    # Link load proxy from delay and traffic
    denom = np.maximum(df["tx_packets"].astype(float), 1.0)
    df["LL_d"] = np.log1p(df["delay_sum"].astype(float)) / denom
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
    print(f"Saved {OUTPUT_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
