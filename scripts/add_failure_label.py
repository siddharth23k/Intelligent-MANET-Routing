"""
add_failure_label.py
--------------------
Generates ground-truth link_failure labels from temporal simulation data.

Label definition (physics-based, not heuristic):
  A node is marked link_failure = 1 at timestep T if ANY of the following
  hold between timestep T and T+1:
    1. neighbor_count drops by >= 1  (a neighbour actually left range)
    2. avg_rssi drops by >= 10 dBm   (signal rapidly deteriorating)
    3. avg_rssi is the sentinel -1000 (node is completely isolated)

This is grounded in real network behaviour:
  - A drop in neighbor_count is a direct observation of link loss
  - A rapid RSSI drop is a leading indicator of imminent link loss
  - Sentinel -1000 means NS-3 reported no neighbours at all

The 10% XOR noise from the old version is removed — it was corrupting labels.
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "dataset/manet_raw_dataset.csv"
OUTPUT_FILE = "dataset/manet_dataset.csv"

RSSI_SENTINEL       = -1000.0   # NS-3 value when node has no neighbours
RSSI_DROP_THRESHOLD = 10.0      # dBm drop between timesteps = deteriorating link
NEIGHBOR_DROP       = 1         # minimum drop in neighbor_count to flag failure

print("Loading raw dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Rows loaded: {len(df)}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Run IDs    : {sorted(df['run_id'].unique())}")

# ── Sort so temporal shifts work correctly ───────────────────────────────────
df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

# ── Compute next-timestep values (within same run + node) ────────────────────
# shift(-1) gives the value at T+1; we only use it when run_id and node_id
# match — mismatches get NaN which we fill with 0 (no change at last timestep)

df["next_neighbor_count"] = (
    df.groupby(["run_id", "node_id"])["neighbor_count"]
    .shift(-1)
)
df["next_avg_rssi"] = (
    df.groupby(["run_id", "node_id"])["avg_rssi"]
    .shift(-1)
)

# ── Three failure conditions ──────────────────────────────────────────────────

# Condition 1: neighbor count drops at next step
cond_neighbor_drop = (
    df["next_neighbor_count"].notna() &
    (df["neighbor_count"] - df["next_neighbor_count"] >= NEIGHBOR_DROP)
)

# Condition 2: RSSI drops sharply at next step
#   (only when both current and next are real values, not sentinel)
valid_rssi_now  = df["avg_rssi"] > RSSI_SENTINEL
valid_rssi_next = df["next_avg_rssi"].notna() & (df["next_avg_rssi"] > RSSI_SENTINEL)
cond_rssi_drop = (
    valid_rssi_now & valid_rssi_next &
    (df["avg_rssi"] - df["next_avg_rssi"] >= RSSI_DROP_THRESHOLD)
)

# Condition 3: node is currently isolated (sentinel value)
cond_isolated = (df["avg_rssi"] == RSSI_SENTINEL)

# ── Combine ───────────────────────────────────────────────────────────────────
df["link_failure"] = (
    cond_neighbor_drop | cond_rssi_drop | cond_isolated
).astype(int)

# ── Drop helper columns ───────────────────────────────────────────────────────
df = df.drop(columns=["next_neighbor_count", "next_avg_rssi"])

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)

# ── Report ────────────────────────────────────────────────────────────────────
counts = df["link_failure"].value_counts()
total  = len(df)
print(f"\nLabel distribution:")
print(f"  Stable (0) : {counts.get(0, 0):>6}  ({100*counts.get(0,0)/total:.1f}%)")
print(f"  Failure (1): {counts.get(1, 0):>6}  ({100*counts.get(1,0)/total:.1f}%)")
print(f"\nDataset saved to: {OUTPUT_FILE}")
print(f"Total rows       : {total}")

# Sanity check — make sure we have failures in every run
per_run = df.groupby("run_id")["link_failure"].mean()
print(f"\nFailure rate per run (min / mean / max):")
print(f"  {per_run.min():.3f} / {per_run.mean():.3f} / {per_run.max():.3f}")