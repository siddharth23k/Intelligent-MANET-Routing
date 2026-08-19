"""Reliability weighted Dijkstra with explicit per edge feature vectors.

Use this when per link features are genuinely available, for example after the
simulation is re instrumented to log neighbour lists rather than counts. It
shares graph construction and the weighting rule with routing_from_dataset, so
the two cannot disagree about what an edge weight means.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from config_loader import get_config  # noqa: E402
from graph_build import build_snapshot_graphs, clip_reliability, reliability_to_weight  # noqa: E402
from predict import LinkFailurePredictor  # noqa: E402
from schema import FEATURES, assert_matrix_shape  # noqa: E402

CFG = get_config()

NodeId = int
EdgeWithFeatures = Tuple[NodeId, NodeId, Sequence[float]]


class MLWeightedDijkstra:
    """Two build modes: explicit edge features, or a dataset snapshot."""

    def __init__(self, predictor: Optional[LinkFailurePredictor] = None):
        self.predictor = predictor or LinkFailurePredictor()
        self.n_features = self.predictor.n_features
        if self.n_features != len(FEATURES):
            raise RuntimeError(
                f"model expects {self.n_features} features, schema declares {len(FEATURES)}. "
                "Retrain with pipeline/train_predictor.py."
            )

    def build_graph(self, nodes: Iterable[NodeId], edges: Iterable[EdgeWithFeatures]) -> nx.Graph:
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(int(node))

        pairs: List[Tuple[NodeId, NodeId]] = []
        features: List[np.ndarray] = []
        for u, v, feature_vector in edges:
            pairs.append((int(u), int(v)))
            features.append(np.asarray(feature_vector, dtype=float).reshape(1, -1))

        if not pairs:
            return graph

        X = assert_matrix_shape(np.vstack(features), self.n_features, "build_graph")
        reliabilities, _ = self.predictor.predict(X)
        for (u, v), reliability in zip(pairs, reliabilities):
            reliability = float(clip_reliability(reliability))
            graph.add_edge(
                u, v, weight=float(reliability_to_weight(reliability)), reliability=reliability
            )
        return graph

    def build_graph_from_snapshot(self, snapshot, radius: float | None = None):
        radius = CFG.communication_radius_default if radius is None else radius
        from routing_from_dataset import DatasetRouter  # local import avoids a cycle

        router = DatasetRouter(predictor=self.predictor)
        graphs = build_snapshot_graphs(snapshot, router.node_reliabilities(snapshot), radius)
        return graphs.ml, graphs.positions

    def find_path(self, graph: nx.Graph, source: NodeId, target: NodeId) -> Optional[List[NodeId]]:
        try:
            return nx.shortest_path(graph, int(source), int(target), weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


def _demo() -> None:
    """Runnable example, so this module is not dead code with a stale path."""
    router = MLWeightedDijkstra()
    rng = np.random.default_rng(0)
    edges = [
        (0, 1, rng.normal(size=len(FEATURES))),
        (1, 2, rng.normal(size=len(FEATURES))),
        (0, 3, rng.normal(size=len(FEATURES))),
        (3, 4, rng.normal(size=len(FEATURES))),
        (4, 2, rng.normal(size=len(FEATURES))),
    ]
    graph = router.build_graph([0, 1, 2, 3, 4], edges)
    print("path 0 -> 2:", router.find_path(graph, 0, 2))
    for u, v, data in graph.edges(data=True):
        print(f"  {u}-{v} reliability {data['reliability']:.3f} weight {data['weight']:.3f}")


if __name__ == "__main__":
    _demo()
