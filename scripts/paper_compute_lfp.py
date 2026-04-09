import os

import numpy as np
import pandas as pd

INPUT_FILE = "dataset/paper/processed/paper_featured_dataset.csv"
OUTPUT_FILE = "dataset/paper/processed/paper_lfp_dataset.csv"
RSSI_SENTINEL = -1000.0
HORIZON = 5

# Paper-inspired weighted LFP factors (normalized weighted sum).
WEIGHTS = {
    "d_res": 0.12,
    "LET": 0.12,
    "ND": 0.10,
    "RSSI": 0.14,
    "LS": 0.10,
    "LA": 0.10,
    "LQ_mean": 0.16,
    "LL_d": 0.12,
    "T_hello": 0.04,
}


def _minmax(s):
    lo, hi = s.min(), s.max()
    if np.isclose(hi - lo, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run paper_feature_engineering.py first.")

    df = pd.read_csv(INPUT_FILE).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    # Supervised label identical to main project for fair comparisons.
    df["f_neighbors"] = df.groupby(["run_id", "node_id"])["neighbor_count"].shift(-HORIZON)
    df["f_rssi"] = df.groupby(["run_id", "node_id"])["avg_rssi"].shift(-HORIZON)
    cond1 = (df["neighbor_count"] - df["f_neighbors"] >= 2)
    cond2 = (df["avg_rssi"] - df["f_rssi"] >= 15.0) & (df["f_rssi"] > RSSI_SENTINEL)
    cond3 = (df["f_rssi"] == RSSI_SENTINEL)
    df["link_failure"] = (cond1 | cond2 | cond3).astype(int)

    # Normalize factors to [0,1]
    n = {}
    n["d_res"] = _minmax(df["d_res"])
    n["LET"] = _minmax(df["LET"])
    n["ND"] = _minmax(df["ND"])
    n["RSSI"] = _minmax(df["RSSI"])
    n["LS"] = _minmax(df["LS"])
    n["LA"] = _minmax(df["LA"])
    n["LQ_mean"] = _minmax(df["LQ_mean"])
    n["LL_d"] = _minmax(df["LL_d"])
    n["T_hello"] = _minmax(df["T_hello"])

    # Convert stability factors into failure tendency where needed.
    fail_score = (
        WEIGHTS["d_res"] * (1.0 - n["d_res"])
        + WEIGHTS["LET"] * (1.0 - n["LET"])
        + WEIGHTS["ND"] * (1.0 - n["ND"])
        + WEIGHTS["RSSI"] * (1.0 - n["RSSI"])
        + WEIGHTS["LS"] * (1.0 - n["LS"])
        + WEIGHTS["LA"] * (1.0 - n["LA"])
        + WEIGHTS["LQ_mean"] * (1.0 - n["LQ_mean"])
        + WEIGHTS["LL_d"] * n["LL_d"]
        + WEIGHTS["T_hello"] * n["T_hello"]
    )
    df["lfp"] = np.clip(fail_score, 0.0, 1.0)

    # Adaptive threshold proxy (SFRNNR-inspired adaptive boundary).
    z = (
        1.8 * n["LQ_mean"]
        + 1.2 * n["RSSI"]
        + 1.0 * n["LS"]
        + 0.8 * n["LET"]
        - 1.0 * n["LL_d"]
        - 0.8 * (1.0 - n["ND"])
    )
    df["lfp_threshold"] = np.clip(0.35 + 0.3 * _sigmoid(z - z.mean()), 0.2, 0.8)
    df["paper_predicted_failure"] = (df["lfp"] > df["lfp_threshold"]).astype(int)

    df = df.drop(columns=["f_neighbors", "f_rssi"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
