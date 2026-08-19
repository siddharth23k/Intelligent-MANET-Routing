"""Our router: reliability weighted Dijkstra over a dataset snapshot.

Graph construction lives in graph_build so this and routing_dijkstra cannot
drift apart. Routing failures are counted rather than swallowed, because a bare
except is how a fully degenerate baseline went unnoticed in the comparison.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from config_loader import get_config  # noqa: E402
from graph_build import SnapshotGraphs, build_snapshot_graphs, node_feature_matrix  # noqa: E402
from metrics import route_metrics  # noqa: E402
from predict import LinkFailurePredictor  # noqa: E402

CFG = get_config()
DEFAULT_RADIUS = CFG.communication_radius_default

NodeId = int


class DatasetRouter:
    """Predicts node reliability, then routes on -log(reliability) weights."""

    def __init__(self, predictor: Optional[LinkFailurePredictor] = None):
        self.predictor = predictor or LinkFailurePredictor()
        self.failures: Counter = Counter()

    def node_reliabilities(self, snapshot) -> Dict[NodeId, float]:
        node_ids, X = node_feature_matrix(snapshot, self.predictor.features)
        if not node_ids:
            return {}
        reliability, _ = self.predictor.predict(X)
        return {int(n): float(r) for n, r in zip(node_ids, reliability)}

    def build_graphs(self, snapshot, radius: float = DEFAULT_RADIUS) -> SnapshotGraphs:
        return build_snapshot_graphs(snapshot, self.node_reliabilities(snapshot), radius)

    def build_graph(self, snapshot, radius: float = DEFAULT_RADIUS):
        """Tuple form kept for the animation and older callers."""
        graphs = self.build_graphs(snapshot, radius=radius)
        return graphs.ml, graphs.hop, graphs.positions

    def find_ml_path(self, graph: nx.Graph, src: NodeId, dst: NodeId) -> Optional[List[NodeId]]:
        return self._shortest_path(graph, src, dst, weight="weight", tag="ml")

    def find_baseline_path(self, graph: nx.Graph, src: NodeId, dst: NodeId) -> Optional[List[NodeId]]:
        return self._shortest_path(graph, src, dst, weight=None, tag="hop")

    def _shortest_path(self, graph, src, dst, weight, tag) -> Optional[List[NodeId]]:
        try:
            return nx.shortest_path(graph, int(src), int(dst), weight=weight)
        except nx.NodeNotFound:
            self.failures[f"{tag}:node_not_found"] += 1
        except nx.NetworkXNoPath:
            self.failures[f"{tag}:no_path"] += 1
        return None

    @staticmethod
    def compute_route_metrics(graph: nx.Graph, path) -> Dict[str, float]:
        return route_metrics(graph, path)

    def failure_report(self) -> Dict[str, int]:
        return dict(self.failures)
