import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from predict import LinkFailurePredictor

AREA_CENTER_X = 250.0  
AREA_CENTER_Y = 250.0
DEFAULT_RADIUS = 150.0

class DatasetRouter:
    def __init__(self):
                self.predictor = LinkFailurePredictor()

    def _compute_node_features(self, snapshot):
        node_features = {}
        for _, row in snapshot.iterrows():
            nid = int(row["node_id"])
            
            feat_vector = np.array([
                row["neighbor_count"],
                row["x"],
                row["y"],
                row["time"],
                row["avg_rssi"],
                row.get("dist_to_center", 0.0),
                row.get("rssi_velocity", 0.0),
                row.get("neighbor_velocity", 0.0),
                row.get("pdr", 1.0),
                row.get("log_delay", 0.0),
                row.get("rssi_trend_3", 0.0),
                row.get("neighbor_trend_3", 0.0),
                row.get("rssi_std_5", 0.0),
                row.get("neighbor_std_5", 0.0)
            ])
            node_features[nid] = feat_vector
        return node_features

    def build_graph(self, snapshot, radius=DEFAULT_RADIUS):
        """
        Build connectivity graphs for a single time snapshot.

        IMPORTANT LIMITATION / DESIGN CHOICE:
        - The ML model is trained on *node-level* features and predicts node instability.
        - Routing requires *edge-level* reliabilities. Without per-link measurements, we approximate
          edge reliability from node reliabilities (conservative: weakest endpoint dominates).
        """
        node_features = self._compute_node_features(snapshot)
        rows = snapshot.to_dict("records")

        G_ml = nx.Graph()
        G_base = nx.Graph()

        for nid in node_features:
            G_ml.add_node(nid)
            G_base.add_node(nid)

        edge_pairs = []
        node_ids = sorted(node_features.keys())
        X_nodes = np.vstack([node_features[nid] for nid in node_ids]) if node_ids else np.zeros((0, 0))
        node_reliability = {}
        if len(node_ids) > 0:
            reliabilities, _ = self.predictor.predict(X_nodes)
            node_reliability = {nid: float(r) for nid, r in zip(node_ids, reliabilities)}

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                n1, n2 = int(rows[i]["node_id"]), int(rows[j]["node_id"])
                if n1 == n2:
                    # Can happen if snapshot contains duplicate node_ids (e.g., mixed runs).
                    continue
                dist = np.sqrt((rows[i]["x"] - rows[j]["x"])**2 + (rows[i]["y"] - rows[j]["y"])**2)

                if dist <= radius:
                    edge_pairs.append((n1, n2))

        for (u, v) in edge_pairs:
            ru = float(node_reliability.get(u, 0.5))
            rv = float(node_reliability.get(v, 0.5))
            # Conservative edge reliability: weakest endpoint dominates.
            r = float(np.clip(min(ru, rv), 0.001, 0.999))
            G_ml.add_edge(u, v, weight=-np.log(r), reliability=r)
            G_base.add_edge(u, v, weight=1, reliability=r)

        return G_ml, G_base, {int(r["node_id"]): (r["x"], r["y"]) for r in rows}

    def find_ml_path(self, G, src, dst):
        try: return nx.shortest_path(G, src, dst, weight="weight")
        except: return None

    def find_baseline_path(self, G, src, dst):
        try: return nx.shortest_path(G, src, dst)
        except: return None

    def compute_route_metrics(self, G, path):
        if not path or len(path) < 2: return {"avg_reliability": 0, "min_reliability": 0, "hop_count": 0}
        rels = [G[u][v]["reliability"] for u, v in zip(path[:-1], path[1:])]
        return {"avg_reliability": np.mean(rels), "min_reliability": np.min(rels), "hop_count": len(path)-1}