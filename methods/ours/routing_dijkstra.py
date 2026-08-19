"""Reliability weighted Dijkstra with explicit per edge feature vectors.

This is the API to use when per link features are actually available, for
example if the simulation is re instrumented to log neighbour lists rather than
neighbour counts. It shares graph construction and the weighting rule with
routing_from_dataset via graph_build, so the two cannot disagree about what an
edge weight means. Previously this module averaged the two endpoints' feature
vectors while the module on the evaluation path took the weaker endpoint, and
nothing recorded which one produced the published numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from config_loader import get_config  # noqa: E402
from graph_build import (  # noqa: E402
    build_snapshot_graphs,
    clip_reliability,
    reliability_to_weight,
)
from predict import LinkFailurePredictor  # noqa: E402
from schema import FEATURES, assert_matrix_shape  # noqa: E402

CFG = get_config()

NodeId = int
EdgeWithFeatures = Tuple[NodeId, NodeId, Sequence[float]]


class MLWeightedDijkstra:
    """Two build modes:

    build_graph(nodes, edges)            explicit per edge feature vectors
    build_graph_from_snapshot(snapshot)  node level features, weaker endpoint wins
    """

    def __init__(self, predictor: Optional[LinkFailurePredictor] = None):
        self.predictor = predictor or LinkFailurePredictor()
        self.n_features = self.predictor.n_features
        if self.n_features != len(FEATURES):
            raise RuntimeError(
                f"model expects {self.n_features} features but schema declares "
                f"{len(FEATURES)}. Retrain with pipeline/train_predictor.py."
            )

    def build_graph(self, nodes: Iterable[NodeId], edges: Iterable[EdgeWithFeatures]) -> nx.Graph:
        g = nx.Graph()
        for n in nodes:
            g.add_node(int(n))

        pairs: List[Tuple[NodeId, NodeId]] = []
        feats: List[np.ndarray] = []
        for u, v, f in edges:
            pairs.append((int(u), int(v)))
            feats.append(np.asarray(f, dtype=float).reshape(1, -1))

        if not pairs:
            return g

        X = assert_matrix_shape(np.vstack(feats), self.n_features, "MLWeightedDijkstra.build_graph")
        reliability, _ = self.predictor.predict(X)
        for (u, v), r in zip(pairs, reliability):
            r = float(clip_reliability(r))
            g.add_edge(u, v, weight=float(reliability_to_weight(r)), reliability=r)
        return g

    def build_graph_from_snapshot(self, snapshot, radius: float = None):
        radius = CFG.communication_radius_default if radius is None else radius
        from routing_from_dataset import DatasetRouter  # local import avoids a cycle

        router = DatasetRouter(predictor=self.predictor)
        graphs = build_snapshot_graphs(snapshot, router.node_reliabilities(snapshot), radius)
        return graphs.ml, graphs.positions

    def find_path(self, g: nx.Graph, source: NodeId, target: NodeId) -> Optional[List[NodeId]]:
        try:
            return nx.shortest_path(g, int(source), int(target), weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


def _demo() -> None:
    """Small runnable example, so the module is not dead code with a stale path."""
    router = MLWeightedDijkstra()
    nodes = [0, 1, 2, 3, 4]
    rng = np.random.default_rng(0)
    edges = [
        (0, 1, rng.normal(size=len(FEATURES))),
        (1, 2, rng.normal(size=len(FEATURES))),
        (0, 3, rng.normal(size=len(FEATURES))),
        (3, 4, rng.normal(size=len(FEATURES))),
        (4, 2, rng.normal(size=len(FEATURES))),
    ]
    g = router.build_graph(nodes, edges)
    print("path 0 -> 2:", router.find_path(g, 0, 2))
    for u, v, d in g.edges(data=True):
        print(f"  {u}-{v} reliability {d['reliability']:.3f} weight {d['weight']:.3f}")


if __name__ == "__main__":
    _demo()
