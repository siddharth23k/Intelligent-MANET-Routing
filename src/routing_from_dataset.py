"""
routing_from_dataset.py
-----------------------
Builds a MANET graph from a dataset snapshot at a given timestep,
uses the ensemble ML model to assign reliability-based edge weights,
and exposes routing + visualisation methods.

Changes from v1:
  - Feature set expanded from 4 to 10 (matches training.py exactly)
  - Edge features now computed per-pair (pairwise distance, avg RSSI, etc.)
  - Both -log(R) and 1/R weighting available (default: -log(R))
  - Baseline hop-count graph also built for fair comparison
  - visualize_graph() shows both ML route and baseline route side by side
  - Color-coded edges by reliability (green/orange/red)
  - compute_route_metrics() returns structured dict for evaluate_routing.py

Feature construction per edge (u, v):
  We compute features from node u's perspective (the source node of the edge).
  For a proper edge-level model you would average u and v features — but since
  our model was trained on node-level rows, using u's features is consistent
  with the training data structure.

  The 5 engineered features (dist_to_center, rssi_velocity, neighbor_velocity,
  pdr, log_delay) are computed live from the snapshot dataframe using the same
  formulas as feature_engineering.py.
"""

import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from predict import LinkFailurePredictor

# Simulation area center (1000x1000m grid)
AREA_CENTER_X = 500.0
AREA_CENTER_Y = 500.0
RSSI_SENTINEL = -1000.0

# Communication radius used to define edges (must match NS-3 config)
DEFAULT_RADIUS = 250.0

# Reliability thresholds for edge colouring
COLOR_GREEN  = 0.70   # reliability >= 0.70 → stable
COLOR_ORANGE = 0.45   # reliability >= 0.45 → medium
                      # reliability <  0.45 → unstable (red)


class DatasetRouter:

    def __init__(self):
        print("Initializing DatasetRouter...")
        self.predictor = LinkFailurePredictor()

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_snapshot(self, csv_path, time_step):
        """Load all rows for a specific timestep from the featured dataset."""
        df = pd.read_csv(csv_path)
        snapshot = df[df["time"] == time_step].copy().reset_index(drop=True)
        if snapshot.empty:
            raise ValueError(f"No data found for time_step={time_step}")
        return snapshot

    # ── Feature engineering (live, from snapshot) ─────────────────────────────

    def _compute_node_features(self, snapshot):
        """
        Computes the 5 engineered features for each node in the snapshot.
        For rssi_velocity and neighbor_velocity, we only have the current
        timestep — so we set them to 0 (neutral, no temporal context).
        In evaluate_routing.py we pass two consecutive snapshots so these
        can be computed properly.

        Returns a dict: node_id → feature_vector (length 10)
        """
        node_features = {}
        for _, row in snapshot.iterrows():
            nid = int(row["node_id"])

            # Original features
            neighbor_count = row["neighbor_count"]
            x, y           = row["x"], row["y"]
            time           = row["time"]
            avg_rssi       = row["avg_rssi"]

            # Engineered features
            dist_to_center    = np.sqrt((x - AREA_CENTER_X)**2 + (y - AREA_CENTER_Y)**2)
            rssi_velocity     = row.get("rssi_velocity", 0.0)
            neighbor_velocity = row.get("neighbor_velocity", 0.0)

            tx = row.get("tx_packets", 0)
            rx = row.get("rx_packets", 0)
            pdr = (rx / tx) if tx > 0 else 1.0
            pdr = np.clip(pdr, 0.0, 1.0)

            log_delay = np.log1p(row.get("delay_sum", 0.0))

            node_features[nid] = np.array([
                neighbor_count,
                x,
                y,
                time,
                avg_rssi,
                dist_to_center,
                rssi_velocity,
                neighbor_velocity,
                pdr,
                log_delay,
            ])
        return node_features

    # ── Graph building ────────────────────────────────────────────────────────

    def build_graph(self, snapshot, radius=DEFAULT_RADIUS):
        """
        Builds two graphs from the snapshot:
          G_ml       — edges weighted by ML failure probability (-log reliability)
          G_baseline — edges weighted by hop count (all weights = 1)

        Returns (G_ml, G_baseline, pos_dict)
        """
        node_features = self._compute_node_features(snapshot)
        rows          = snapshot.to_dict("records")

        G_ml       = nx.Graph()
        G_baseline = nx.Graph()

        for nid in node_features:
            G_ml.add_node(nid)
            G_baseline.add_node(nid)

        # ── Identify edges within communication radius ─────────────────────
        edge_pairs    = []
        edge_features = []

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                n1 = int(rows[i]["node_id"])
                n2 = int(rows[j]["node_id"])

                if n1 == n2:
                    continue

                x1, y1 = rows[i]["x"], rows[i]["y"]
                x2, y2 = rows[j]["x"], rows[j]["y"]
                dist   = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                if dist <= radius:
                    # Use average of both nodes' features for edge representation
                    feat_avg = (node_features[n1] + node_features[n2]) / 2.0
                    edge_pairs.append((n1, n2))
                    edge_features.append(feat_avg)

        if len(edge_features) == 0:
            print("  Warning: no edges found within radius. Check radius value.")
            return G_ml, G_baseline, {}

        # ── Batch predict all edges at once ───────────────────────────────
        X = np.array(edge_features)
        reliabilities, _ = self.predictor.predict(X)

        # ── Add edges to both graphs ───────────────────────────────────────
        for (u, v), r in zip(edge_pairs, reliabilities):
            r = float(np.clip(r, 1e-6, 1.0 - 1e-6))

            # ML graph: -log(R) weight — Dijkstra minimises this,
            # which is equivalent to maximising the product of reliabilities
            ml_weight = -np.log(r)

            G_ml.add_edge(u, v, weight=ml_weight, reliability=r)
            G_baseline.add_edge(u, v, weight=1)   # hop count

        # ── Position dict for visualisation ───────────────────────────────
        pos = {}
        for _, row in snapshot.iterrows():
            pos[int(row["node_id"])] = (row["x"], row["y"])

        return G_ml, G_baseline, pos

    # ── Routing ───────────────────────────────────────────────────────────────

    def find_ml_path(self, G_ml, source, target):
        """Returns ML-weighted shortest path, or None if unreachable."""
        try:
            return nx.shortest_path(G_ml, source, target, weight="weight")
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def find_baseline_path(self, G_baseline, source, target):
        """Returns hop-count shortest path, or None if unreachable."""
        try:
            return nx.shortest_path(G_baseline, source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    # ── Metrics ───────────────────────────────────────────────────────────────

    def compute_route_metrics(self, G_ml, path):
        """
        Given a path (list of node IDs) on G_ml, computes:
          - hop_count       : number of hops (len(path) - 1)
          - avg_reliability : mean reliability of all edges in path
          - min_reliability : bottleneck link reliability (weakest link)
          - path_exists     : True/False
        """
        if path is None or len(path) < 2:
            return {
                "path_exists"     : False,
                "hop_count"       : None,
                "avg_reliability" : None,
                "min_reliability" : None,
            }

        edge_reliabilities = []
        for u, v in zip(path[:-1], path[1:]):
            r = G_ml[u][v].get("reliability", 0.0)
            edge_reliabilities.append(r)

        return {
            "path_exists"     : True,
            "hop_count"       : len(path) - 1,
            "avg_reliability" : float(np.mean(edge_reliabilities)),
            "min_reliability" : float(np.min(edge_reliabilities)),
        }

    # ── Visualisation ─────────────────────────────────────────────────────────

    def visualize_graph(self, G_ml, pos, ml_path=None, baseline_path=None,
                        title="MANET Topology", save_path=None):
        """
        Draws the MANET topology with:
          - Edges colour-coded by reliability (green/orange/red)
          - Blue path  = ML-selected route
          - Purple path = baseline hop-count route
        """
        fig, ax = plt.subplots(figsize=(11, 8))

        # ── Draw all edges coloured by reliability ─────────────────────────
        for u, v, data in G_ml.edges(data=True):
            r = data.get("reliability", 0.5)
            if r >= COLOR_GREEN:
                color = "#4CAF50"    # green
            elif r >= COLOR_ORANGE:
                color = "#FF9800"    # orange
            else:
                color = "#F44336"    # red
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            ax.plot(x_vals, y_vals, color=color, alpha=0.6, linewidth=1.2)

        # ── Draw baseline route (purple, behind ML route) ──────────────────
        if baseline_path and len(baseline_path) >= 2:
            for u, v in zip(baseline_path[:-1], baseline_path[1:]):
                x_vals = [pos[u][0], pos[v][0]]
                y_vals = [pos[u][1], pos[v][1]]
                ax.plot(x_vals, y_vals, color="#9C27B0",
                        linewidth=4, alpha=0.7, zorder=3,
                        linestyle="--")

        # ── Draw ML route (blue, on top) ───────────────────────────────────
        if ml_path and len(ml_path) >= 2:
            for u, v in zip(ml_path[:-1], ml_path[1:]):
                x_vals = [pos[u][0], pos[v][0]]
                y_vals = [pos[u][1], pos[v][1]]
                ax.plot(x_vals, y_vals, color="#2196F3",
                        linewidth=4, alpha=0.9, zorder=4)

        # ── Draw nodes ─────────────────────────────────────────────────────
        for nid, (x, y) in pos.items():
            is_endpoint = (ml_path and (nid == ml_path[0] or nid == ml_path[-1]))
            color  = "#FF5722" if is_endpoint else "#37474F"
            size   = 120 if is_endpoint else 60
            ax.scatter(x, y, c=color, s=size, zorder=5)
            ax.annotate(str(nid), (x, y),
                        textcoords="offset points", xytext=(4, 4),
                        fontsize=7, color="white" if not is_endpoint else "#FF5722")

        # ── Legend ─────────────────────────────────────────────────────────
        legend_elements = [
            mpatches.Patch(color="#4CAF50", label=f"Reliable (R ≥ {COLOR_GREEN})"),
            mpatches.Patch(color="#FF9800", label=f"Medium ({COLOR_ORANGE} ≤ R < {COLOR_GREEN})"),
            mpatches.Patch(color="#F44336", label=f"Unstable (R < {COLOR_ORANGE})"),
            plt.Line2D([0],[0], color="#2196F3", lw=3, label="ML Route"),
            plt.Line2D([0],[0], color="#9C27B0", lw=3, linestyle="--",
                       label="Baseline Route"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Y position (m)")
        ax.set_xlim(-30, 1030)
        ax.set_ylim(-30, 1030)
        ax.grid(alpha=0.15)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Plot saved: {save_path}")
        plt.show()


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    DATASET = "dataset/manet_featured_dataset.csv"
    TIME    = 30.0
    SOURCE  = 0
    TARGET  = 15

    router = DatasetRouter()

    print(f"\nLoading snapshot at t={TIME}...")
    snapshot = router.load_snapshot(DATASET, TIME)
    print(f"  Nodes in snapshot: {len(snapshot)}")

    print("\nBuilding graphs...")
    G_ml, G_base, pos = router.build_graph(snapshot)
    print(f"  Edges in ML graph      : {G_ml.number_of_edges()}")
    print(f"  Edges in baseline graph: {G_base.number_of_edges()}")

    ml_path   = router.find_ml_path(G_ml, SOURCE, TARGET)
    base_path = router.find_baseline_path(G_base, SOURCE, TARGET)

    ml_metrics   = router.compute_route_metrics(G_ml, ml_path)
    base_metrics = router.compute_route_metrics(G_ml, base_path)

    print(f"\nRouting {SOURCE} → {TARGET}")
    print(f"  ML path       : {ml_path}")
    print(f"  Baseline path : {base_path}")
    print(f"\n  ML       — hops: {ml_metrics['hop_count']}, "
          f"avg_rel: {ml_metrics['avg_reliability']:.3f}, "
          f"min_rel: {ml_metrics['min_reliability']:.3f}")
    print(f"  Baseline — hops: {base_metrics['hop_count']}, "
          f"avg_rel: {base_metrics['avg_reliability']:.3f}, "
          f"min_rel: {base_metrics['min_reliability']:.3f}")

    os.makedirs("assets", exist_ok=True)
    router.visualize_graph(
        G_ml, pos,
        ml_path=ml_path,
        baseline_path=base_path,
        title=f"MANET Topology at t={TIME} | Source={SOURCE} Target={TARGET}",
        save_path="assets/topology_snapshot.jpg"
    )