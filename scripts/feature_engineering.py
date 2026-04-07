import pandas as pd
import numpy as np

INPUT_FILE = "dataset/manet_dataset.csv"
OUTPUT_FILE = "dataset/manet_featured_dataset.csv"

def engineer():
    df = pd.read_csv("dataset/manet_dataset.csv") # Use the output from build_dataset
    df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

    # Core Features
    df["dist_to_center"] = np.sqrt((df["x"]-250)**2 + (df["y"]-250)**2)
    df["rssi_velocity"] = df.groupby(["run_id", "node_id"])["avg_rssi"].diff().fillna(0)
    df["neighbor_velocity"] = df.groupby(["run_id", "node_id"])["neighbor_count"].diff().fillna(0)

    # Network Features (The missing ones!)
    df["pdr"] = np.where(df["tx_packets"] > 0, df["rx_packets"] / df["tx_packets"], 1.0)
    df["log_delay"] = np.log1p(df["delay_sum"])

    # Rolling Historical Features
    group = df.groupby(["run_id", "node_id"])
    df["rssi_trend_3"] = group["rssi_velocity"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df["neighbor_trend_3"] = group["neighbor_velocity"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    df["rssi_std_5"] = group["avg_rssi"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)
    df["neighbor_std_5"] = group["neighbor_count"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std()).fillna(0)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Featured dataset saved with {len(df.columns)} columns.")

if __name__ == "__main__":
    engineer()