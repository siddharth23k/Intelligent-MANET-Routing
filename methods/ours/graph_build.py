"""The one place a routing graph is built from a dataset snapshot.

There used to be two implementations with different semantics. One averaged the
two endpoints' feature vectors and predicted once per edge, the other predicted
per node and took the weaker endpoint. Only one was on the evaluation path, and
nothing said which. Both routers now call into this module, so they cannot
disagree.

Design note on the node to edge step. The dataset has one row per node per
second, not one row per link, so there is no per link label to train on. Edge
reliability is therefore approximated from node reliabilities, conservatively:
a link is only as good as its weaker end. That approximation is stated in the
report output so it is visible rather than buried.
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


def clip_reliability(r) -> np.ndarray:
    """Keep reliabilities strictly inside (0, 1).

    The ceiling matters for correctness, not just numerics: Dijkstra requires
    non negative edge weights, and w = -log(r) is negative for any r > 1.
    """
    return np.clip(np.asarray(r, dtype=float), RELIABILITY_FLOOR, RELIABILITY_CEIL)


def reliability_to_weight(r) -> np.ndarray:
    """w = -log(r). Summing w along a path equals -log of the product of r,
    so Dijkstra's minimum weight path is exactly the maximum reliability path."""
    return -np.log(clip_reliability(r))


def node_feature_matrix(snapshot, features: Sequence[str] = None) -> Tuple[List[NodeId], np.ndarray]:
    """Canonical (node_ids, X) for one time snapshot, in schema column order."""
    features = list(features or FEATURES)
    assert_columns(snapshot, ["node_id"] + features, "node_feature_matrix")
    frame = snapshot.sort_values("node_id")
    node_ids = [int(n) for n in frame["node_id"].to_numpy()]
    X = frame[features].to_numpy(dtype=float)
    return node_ids, X


def geometric_edges(snapshot, radius: float) -> List[Tuple[NodeId, NodeId]]:
    """Every unordered pair within `radius`.

    Uses a uniform grid bucketed at the radius, so each node only compares
    against its own cell and the eight neighbouring cells. That is O(n) expected
    instead of the O(n^2) double loop this replaced, which mattered as soon as
    node counts went past a few hundred.
    """
    frame = snapshot.sort_values("node_id")
    ids = frame["node_id"].to_numpy().astype(int)
    xs = frame["x"].to_numpy(dtype=float)
    ys = frame["y"].to_numpy(dtype=float)

    if radius <= 0 or len(ids) < 2:
        return []

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(ids)):
        key = (int(xs[i] // radius), int(ys[i] // radius))
        buckets.setdefault(key, []).append(i)

    r2 = radius * radius
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
                if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 <= r2:
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
    ml: nx.Graph                       # weighted by -log(reliability)
    hop: nx.Graph                      # unweighted, the shortest path baseline
    positions: Dict[NodeId, Tuple[float, float]] = field(default_factory=dict)
    node_reliability: Dict[NodeId, float] = field(default_factory=dict)


def build_snapshot_graphs(
    snapshot,
    node_reliability: Dict[NodeId, float],
    radius: float,
) -> SnapshotGraphs:
    """Build the ML weighted graph and the hop count graph over the same edges.

    Both graphs always carry the identical edge set, which is what makes the
    paired comparison in compare_methods valid: any difference in the routes
    comes from the weighting, never from the topology.
    """
    edges = geometric_edges(snapshot, radius)
    frame = snapshot.sort_values("node_id")
    positions = {
        int(r.node_id): (float(r.x), float(r.y)) for r in frame.itertuples(index=False)
    }

    g_ml = nx.Graph()
    g_hop = nx.Graph()
    for nid in positions:
        g_ml.add_node(nid)
        g_hop.add_node(nid)

    for u, v in edges:
        ru = float(node_reliability.get(u, 0.5))
        rv = float(node_reliability.get(v, 0.5))
        r = float(clip_reliability(min(ru, rv)))
        g_ml.add_edge(u, v, weight=float(reliability_to_weight(r)), reliability=r)
        g_hop.add_edge(u, v, weight=1.0, reliability=r)

    return SnapshotGraphs(
        ml=g_ml, hop=g_hop, positions=positions, node_reliability=dict(node_reliability)
    )
