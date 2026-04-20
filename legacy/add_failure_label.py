import pandas as pd
import numpy as np

INPUT_FILE  = "dataset/manet_raw_dataset.csv"
OUTPUT_FILE = "dataset/manet_dataset.csv"

RSSI_SENTINEL = -1000.0
PREDICT_HORIZON = 5

def generate_labels():
    df = pd.read_csv(INPUT_FILE)
    df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    df["f_neighbors"] = df.groupby(["run_id", "node_id"])["neighbor_count"].shift(-PREDICT_HORIZON)
    df["f_rssi"] = df.groupby(["run_id", "node_id"])["avg_rssi"].shift(-PREDICT_HORIZON)

    RSSI_DROP = 15.0
    NB_DROP = 2

    cond1 = (df["neighbor_count"] - df["f_neighbors"] >= NB_DROP)
    cond2 = (df["avg_rssi"] - df["f_rssi"] >= RSSI_DROP) & (df["f_rssi"] > RSSI_SENTINEL)
    cond3 = (df["f_rssi"] == RSSI_SENTINEL)
    df["link_failure"] = (cond1 | cond2 | cond3).astype(int)
    
    df = df.drop(columns=["f_neighbors", "f_rssi"])
    df.to_csv(OUTPUT_FILE, index=False)
    
    rate = df["link_failure"].mean()
    per_run = df.groupby("run_id")["link_failure"].mean()

if __name__ == "__main__":
    generate_labels()