"""
feature_engineering.py
-----------------------
Derives richer features from manet_dataset.csv (which already has link_failure
labels from add_failure_label.py).

Original features (kept):
  neighbor_count, x, y, time, avg_rssi

New features derived:
  1. dist_to_center     -- Euclidean distance from node to simulation area center
                           (1000x1000 grid → center = 500,500)
                           Nodes at edges have fewer stable links.

  2. rssi_velocity      -- Change in avg_rssi from previous timestep
                           (within same run + node)
                           A fast-dropping RSSI predicts imminent link loss.
                           First timestep gets 0 (no previous to compare).

  3. neighbor_velocity  -- Change in neighbor_count from previous timestep
                           Captures whether the node is currently gaining or
                           losing neighbours (momentum signal).
                           First timestep gets 0.

  4. pdr               -- Packet Delivery Ratio = rx_packets / tx_packets
                           Measures how many sent packets actually arrived.
                           0 when tx_packets == 0 (early in simulation).

  5. log_delay         -- log1p(delay_sum) — log-transformed total delay.
                           delay_sum is heavily right-skewed; log compression
                           makes it usable as a linear feature.

Output: dataset/manet_featured_dataset.csv
  All original columns are preserved + 5 new columns added.
  The training notebook will load this file instead of manet_dataset.csv.

Run order:
  1. scripts/add_failure_label.py   → dataset/manet_dataset.csv
  2. scripts/feature_engineering.py → dataset/manet_featured_dataset.csv
  3. notebooks/training.ipynb       (loads manet_featured_dataset.csv)
"""

import pandas as pd
import numpy as np

INPUT_FILE  = "dataset/manet_dataset.csv"
OUTPUT_FILE = "dataset/manet_featured_dataset.csv"

# Simulation area is 1000m x 1000m (standard NS-3 MANET config)
AREA_CENTER_X = 500.0
AREA_CENTER_Y = 500.0

print("Loading labelled dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Rows  : {len(df)}")
print(f"  Cols  : {list(df.columns)}")

# ── Sort for temporal features ────────────────────────────────────────────────
df = df.sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: dist_to_center
# Nodes near the boundary have fewer potential neighbours on all sides.
# This captures spatial isolation that neighbor_count alone misses
# (a node could have 3 neighbours but all on one side, about to walk off-edge).
# ─────────────────────────────────────────────────────────────────────────────
df["dist_to_center"] = np.sqrt(
    (df["x"] - AREA_CENTER_X) ** 2 +
    (df["y"] - AREA_CENTER_Y) ** 2
)
print("  [✓] dist_to_center computed")

# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: rssi_velocity
# Rate of change of RSSI between consecutive timesteps.
# A node whose RSSI is dropping at -5 dBm/step is much more at risk than one
# whose RSSI is stable at -60 dBm.
# We set sentinel (-1000) rows to NaN before differencing so they don't
# create enormous fake velocity values.
# ─────────────────────────────────────────────────────────────────────────────
RSSI_SENTINEL = -1000.0

# Replace sentinel with NaN temporarily for clean differencing
rssi_clean = df["avg_rssi"].replace(RSSI_SENTINEL, np.nan)
df["rssi_velocity"] = (
    rssi_clean
    .groupby([df["run_id"], df["node_id"]])
    .diff()           # current - previous (positive = improving, negative = dropping)
    .fillna(0.0)      # first timestep per node gets 0
)
print("  [✓] rssi_velocity computed")

# ─────────────────────────────────────────────────────────────────────────────
# Feature 3: neighbor_velocity
# Rate of change of neighbor_count between consecutive timesteps.
# -2 means the node just lost 2 neighbours — strong failure predictor.
# +1 means a new neighbour just came into range — stability signal.
# ─────────────────────────────────────────────────────────────────────────────
df["neighbor_velocity"] = (
    df.groupby(["run_id", "node_id"])["neighbor_count"]
    .diff()
    .fillna(0.0)
)
print("  [✓] neighbor_velocity computed")

# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: pdr (Packet Delivery Ratio)
# rx_packets / tx_packets — the fraction of sent packets that were received.
# PDR < 1.0 means some packets are already being lost, which correlates with
# an unstable or congested link.
# When tx_packets == 0 (early in simulation), PDR is set to 1.0 (optimistic
# default — no evidence of loss yet).
# ─────────────────────────────────────────────────────────────────────────────
df["pdr"] = np.where(
    df["tx_packets"] > 0,
    df["rx_packets"] / df["tx_packets"],
    1.0   # no traffic yet → assume perfect delivery
)
# Clip to [0, 1] in case of any NS-3 reporting quirks
df["pdr"] = df["pdr"].clip(0.0, 1.0)
print("  [✓] pdr computed")

# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: log_delay
# delay_sum is the total end-to-end packet delay accumulated in the run.
# It is heavily right-skewed (most nodes have 0, a few have very large values).
# log1p(x) = log(1 + x) compresses the scale and handles 0s cleanly.
# High delay correlates with congestion and queuing — both link stress signals.
# ─────────────────────────────────────────────────────────────────────────────
df["log_delay"] = np.log1p(df["delay_sum"])
print("  [✓] log_delay computed")

# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks
# ─────────────────────────────────────────────────────────────────────────────
print("\nFeature summary:")
new_features = ["dist_to_center", "rssi_velocity", "neighbor_velocity", "pdr", "log_delay"]
for feat in new_features:
    col = df[feat]
    print(f"  {feat:22s}  min={col.min():8.3f}  mean={col.mean():8.3f}  max={col.max():8.3f}  nulls={col.isna().sum()}")

# Verify no NaNs in any feature column
all_features = ["neighbor_count", "x", "y", "time", "avg_rssi"] + new_features
null_counts = df[all_features].isna().sum()
if null_counts.sum() == 0:
    print("\n  [✓] No null values in any feature column")
else:
    print("\n  [!] Warning — nulls found:")
    print(null_counts[null_counts > 0])

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nFeatured dataset saved to : {OUTPUT_FILE}")
print(f"Total rows                 : {len(df)}")
print(f"Total columns              : {len(df.columns)}")
print(f"Final columns              : {list(df.columns)}")

# Label distribution check
counts = df["link_failure"].value_counts()
total  = len(df)
print(f"\nLabel distribution (preserved from input):")
print(f"  Stable (0) : {counts.get(0,0):>6}  ({100*counts.get(0,0)/total:.1f}%)")
print(f"  Failure (1): {counts.get(1,0):>6}  ({100*counts.get(1,0)/total:.1f}%)")
