"""Snapshot graph construction, shared by every router.

Both routers call in here so they cannot disagree about what an edge weight
means. The dataset has one row per node per second, not one per link, so edge
reliability is approximated from node reliabilities: a link is only as good as
its weaker end.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from schema import FEATURES, assert_columns  # noqa: E402

NodeId = int

RELIABILITY_FLOOR = 1e-3
RELIABILITY_CEIL = 1.0 - 1e-3


def clip_reliability(reliability) -> np.ndarray:
    """Keep reliability strictly inside (0, 1).

    The ceiling is load bearing: Dijkstra needs non negative weights and
    -log(r) goes negative as soon as r exceeds 1.
    """
    return np.clip(np.asarray(reliability, dtype=float), RELIABILITY_FLOOR, RELIABILITY_CEIL)


def reliability_to_weight(reliability) -> np.ndarray:
    """w = -log(r), so summing weights along a path multiplies reliabilities."""
    return -np.log(clip_reliability(reliability))


def node_feature_matrix(
    snapshot, features: Sequence[str] | None = None
) -> Tuple[List[NodeId], np.ndarray]:
    """(node_ids, X) for one snapshot, in canonical column order."""
    features = list(features or FEATURES)
    assert_columns(snapshot, ["node_id"] + features, "node_feature_matrix")
    frame = snapshot.sort_values("node_id")
    node_ids = [int(n) for n in frame["node_id"].to_numpy()]
    return node_ids, frame[features].to_numpy(dtype=float)


def geometric_edges(snapshot, radius: float) -> List[Tuple[NodeId, NodeId]]:
    """Unordered node pairs within `radius`.

    Bucketed on a uniform grid of cell size `radius`, so each node only compares
    against its own cell and the eight neighbours. Expected O(n) rather than the
    O(n^2) double loop this replaced.
    """
    frame = snapshot.sort_values("node_id")
    ids = frame["node_id"].to_numpy().astype(int)
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)

    if radius <= 0 or len(ids) < 2:
        return []

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(ids)):
        buckets.setdefault((int(xs[i] // radius), int(ys[i] // radius)), []).append(i)

    radius_sq = radius * radius
    edges: List[Tuple[NodeId, NodeId]] = []
    seen = set()
    for (cx, cy), members in buckets.items():
        candidates: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(buckets.get((cx + dx, cy + dy), ()))
        for i in members:
            for j in candidates:
                if j <= i:
                    continue
                if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 <= radius_sq:
                    u, v = int(ids[i]), int(ids[j])
                    if u == v:
                        continue
                    key = (u, v) if u < v else (v, u)
                    if key not in seen:
                        seen.add(key)
                        edges.append(key)
    return edges


@dataclass
class SnapshotGraphs:
    ml: nx.Graph                    # weighted by -log(reliability)
    hop: nx.Graph                   # unweighted, the shortest path baseline
    positions: Dict[NodeId, Tuple[float, float]] = field(default_factory=dict)
    node_reliability: Dict[NodeId, float] = field(default_factory=dict)


def build_snapshot_graphs(
    snapshot, node_reliability: Dict[NodeId, float], radius: float
) -> SnapshotGraphs:
    """Build the weighted and hop count graphs over an identical edge set.

    Sharing the edge set is what makes the paired comparison valid: routes can
    differ because of the weighting, never because of the topology.
    """
    frame = snapshot.sort_values("node_id")
    positions = {
        int(row.node_id): (float(row.x), float(row.y)) for row in frame.itertuples(index=False)
    }

    g_ml, g_hop = nx.Graph(), nx.Graph()
    for node_id in positions:
        g_ml.add_node(node_id)
        g_hop.add_node(node_id)

    for u, v in geometric_edges(frame, radius):
        weaker = min(float(node_reliability.get(u, 0.5)), float(node_reliability.get(v, 0.5)))
        reliability = float(clip_reliability(weaker))
        g_ml.add_edge(u, v, weight=float(reliability_to_weight(reliability)), reliability=reliability)
        g_hop.add_edge(u, v, weight=1.0, reliability=reliability)

    return SnapshotGraphs(
        ml=g_ml, hop=g_hop, positions=positions, node_reliability=dict(node_reliability)
    )
