"""Standalone Dijkstra routing with ML weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.append(str(Path(__file__).resolve().parent))

_project_root = Path(__file__).resolve().parent.parent
_cache_root = _project_root / ".cache"
try:
    _cache_root.mkdir(parents=True, exist_ok=True)
except Exception:
    _cache_root = None

if _cache_root is not None:
    os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import networkx as nx
import numpy as np

from predict import LinkFailurePredictor


NodeId = int
EdgeWithFeatures = Tuple[NodeId, NodeId, Sequence[float]]

FEATURES: List[str] = [
    "neighbor_count", "x", "y", "time", "avg_rssi",
    "dist_to_center", "rssi_velocity", "neighbor_velocity",
    "pdr", "log_delay",
    "rssi_trend_3", "neighbor_trend_3", "rssi_std_5", "neighbor_std_5",
]

class MLWeightedDijkstra:
    """
    Two supported build modes:
    - build_graph(nodes, edges): explicit edges with feature vectors
    - build_graph_from_snapshot(snapshot, radius=...): build edges from a dataset snapshot
    """

    def __init__(self):
        self.predictor = LinkFailurePredictor()
        self._n_features = int(getattr(self.predictor.scaler, "n_features_in_", 0) or 0)

        if self._n_features <= 0:
            raise RuntimeError(
                "Could not infer model feature dimension from scaler. "
                "Check that `models/scaler.pkl` is compatible with sklearn."
            )

        if self._n_features != len(FEATURES):
            raise RuntimeError(
                "Feature schema mismatch: model expects "
                f"{self._n_features} features but this module defines {len(FEATURES)}. "
                "Update `FEATURES` here (and keep it identical to training)."
            )

    def _assert_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features:
            raise ValueError(
                f"Invalid feature matrix shape {X.shape}; expected (*, {self._n_features}). "
                f"Expected feature order: {FEATURES}"
            )
        return X

    def build_graph(self, nodes: Iterable[NodeId], edges: Iterable[EdgeWithFeatures]) -> nx.Graph:
        """
        - **nodes**: iterable of node ids
        - **edges**: iterable of tuples (u, v, features)

        `features` must match the trained schema exactly; mismatches raise an error.
        """
        G = nx.Graph()
        for n in nodes:
            G.add_node(int(n))

        edge_pairs: List[Tuple[NodeId, NodeId]] = []
        edge_features: List[np.ndarray] = []

        for (u, v, features) in edges:
            u_i, v_i = int(u), int(v)
            edge_pairs.append((u_i, v_i))
            edge_features.append(np.asarray(features, dtype=float))

        if not edge_pairs:
            return G

        X = self._assert_feature_matrix(np.vstack([f.reshape(1, -1) for f in edge_features]))
        reliabilities, _ = self.predictor.predict(X)

        for (u, v), r in zip(edge_pairs, reliabilities):
            r = float(np.clip(r, 0.001, 0.999))
            weight = float(-np.log(r))
            G.add_edge(u, v, weight=weight, reliability=r)

        return G

    def build_graph_from_snapshot(self, snapshot, radius: float = 150.0):
        """Builds a graph from a dataset snapshot."""
        import pandas as pd  # type: ignore

        if not isinstance(snapshot, pd.DataFrame):
            raise TypeError("snapshot must be a pandas DataFrame.")
        if snapshot.empty:
            return nx.Graph(), {}

        def node_feat(row) -> np.ndarray:
            return np.array(
                [
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
                    row.get("neighbor_std_5", 0.0),
                ],
                dtype=float,
            )

        rows = snapshot.to_dict("records")
        node_features: Dict[NodeId, np.ndarray] = {int(r["node_id"]): node_feat(r) for r in rows}

        G = nx.Graph()
        for nid in node_features:
            G.add_node(nid)

        edge_pairs: List[Tuple[NodeId, NodeId]] = []
        edge_features: List[np.ndarray] = []

        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                n1, n2 = int(rows[i]["node_id"]), int(rows[j]["node_id"])
                dx = float(rows[i]["x"] - rows[j]["x"])
                dy = float(rows[i]["y"] - rows[j]["y"])
                dist = float(np.sqrt(dx * dx + dy * dy))
                if dist <= radius:
                    feat_avg = (node_features[n1] + node_features[n2]) / 2.0
                    edge_pairs.append((n1, n2))
                    edge_features.append(feat_avg)

        if edge_pairs:
            X = self._assert_feature_matrix(np.array(edge_features, dtype=float))
            reliabilities, _ = self.predictor.predict(X)
            for (u, v), r in zip(edge_pairs, reliabilities):
                r = float(np.clip(r, 0.001, 0.999))
                G.add_edge(u, v, weight=float(-np.log(r)), reliability=r)

        pos = {int(r["node_id"]): (float(r["x"]), float(r["y"])) for r in rows}
        return G, pos

    def find_path(self, G: nx.Graph, source: NodeId, target: NodeId) -> Optional[List[NodeId]]:
        try:
            return nx.shortest_path(G, int(source), int(target), weight="weight")
        except Exception:
            return None


def _demo_from_dataset():
    import pandas as pd  # type: ignore

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "dataset" / "manet_featured_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at `{dataset_path}`.")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise RuntimeError("Dataset CSV is empty.")

    snapshot = None
    chosen_t = None
    for t in sorted(df["time"].unique()):
        cand = df[df["time"] == t]
        cand = cand[(cand["x"] != 0) | (cand["y"] != 0)]
        if len(cand) >= 3:
            snapshot = cand
            chosen_t = float(t)
            break

    if snapshot is None:
        raise RuntimeError("No usable snapshot found (all nodes appear at (0,0) or too few nodes).")

    router = MLWeightedDijkstra()
    G, _pos = router.build_graph_from_snapshot(snapshot, radius=150.0)

    src, dst = 0, 5
    path = router.find_path(G, src, dst)
    

def _demo_toy_graph():
    router = MLWeightedDijkstra()

    nodes = [0, 1, 2, 3, 4]
    edges = [
        (0, 1, [3, 120, 200, 10, -70, 50, -0.2, -1, 0.9, 1.0, -0.1, -0.3, 2.0, 1.0]),
        (1, 2, [2, 200, 250, 10, -75, 60, -0.1, 0, 0.8, 1.2, -0.05, -0.1, 1.5, 1.2]),
        (0, 3, [1, 300, 100, 10, -85, 80, 0.0, -2, 0.7, 1.4, 0.02, 0.2, 3.0, 2.5]),
        (3, 4, [4, 250, 150, 10, -65, 40, -0.3, 1, 0.95, 0.9, -0.2, -0.4, 1.0, 0.8]),
        (4, 2, [2, 260, 200, 10, -72, 55, -0.15, 0, 0.85, 1.1, -0.08, -0.2, 1.8, 1.1]),
    ]

    G = router.build_graph(nodes, edges)
    path = router.find_path(G, 0, 2)

    print(f"Best path based on ML reliability: {path}")

    print("Edge reliabilities:")
    for u, v, data in G.edges(data=True):
        print(f"{u}-{v} reliability: {round(float(data.get('reliability', 0.0)), 3)}")


if __name__ == "__main__":
    try:
        _demo_from_dataset()
    except Exception as e:
        print(f"Dataset demo unavailable ({e}). Falling back to toy demo.")
        _demo_toy_graph()