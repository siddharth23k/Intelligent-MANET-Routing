"""The paper's FRLFP router: exclude predicted-risky nodes, then route by hops.

Behaviour worth understanding before reading the numbers this produces. When the
SFRNNR flags most of the network as risky, the filtered graph disconnects and
this router falls back to the unfiltered graph, which makes it identical to plain
shortest path. The original implementation did that silently inside a bare
`except`, so a run in which the baseline degenerated on every single decision
looked exactly like a run in which it worked. Fallbacks are now counted and the
rate is reported by compare_methods.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

from config_loader import get_config  # noqa: E402
from graph_build import clip_reliability, geometric_edges  # noqa: E402
from metrics import route_metrics  # noqa: E402

CFG = get_config()
NodeId = int


class FRLFPRouter:
    def __init__(self):
        self.stats: Counter = Counter()

    def build_graphs(self, snapshot, radius: float = None):
        """Return (full graph, risk filtered graph, positions, risky node set).

        Edge reliability follows the paper: one minus the worse of the two
        endpoints' predicted link failure probabilities. Both graphs are built
        over the same geometric edge set as our router, so any difference in
        routes comes from the method, never from the topology.
        """
        radius = CFG.communication_radius_default if radius is None else radius
        frame = snapshot.sort_values("node_id")
        for col in ("lfp", "lfp_threshold"):
            if col not in frame.columns:
                raise KeyError(
                    f"FRLFPRouter needs column '{col}'. Run pipeline/train_models.py "
                    "to attach SFRNNR outputs to the dataset."
                )

        lfp = {int(r.node_id): float(r.lfp) for r in frame.itertuples(index=False)}
        thr = {int(r.node_id): float(r.lfp_threshold) for r in frame.itertuples(index=False)}
        risky: Set[NodeId] = {n for n in lfp if lfp[n] > thr[n]}

        positions = {
            int(r.node_id): (float(r.x), float(r.y)) for r in frame.itertuples(index=False)
        }

        g_full = nx.Graph()
        g_filtered = nx.Graph()
        for nid in positions:
            g_full.add_node(nid)
            if nid not in risky:
                g_filtered.add_node(nid)

        for u, v in geometric_edges(frame, radius):
            rel = float(clip_reliability(1.0 - max(lfp[u], lfp[v])))
            g_full.add_edge(u, v, weight=1.0, reliability=rel)
            if u not in risky and v not in risky:
                g_filtered.add_edge(u, v, weight=1.0, reliability=rel)

        self.stats["snapshots"] += 1
        self.stats["risky_nodes"] += len(risky)
        self.stats["nodes"] += len(positions)
        return g_full, g_filtered, positions, risky

    def route(self, g_full: nx.Graph, g_filtered: nx.Graph, src: NodeId, dst: NodeId) -> Optional[List[NodeId]]:
        """Route on the filtered graph, falling back to the full graph.

        The fallback counter only counts cases where the filter is what broke
        routing, that is where the filtered graph had no path but the full graph
        did. A pair that is disconnected in both graphs is counted separately as
        `no_path`, otherwise the fallback rate would be inflated by topology that
        has nothing to do with the baseline's predictions.
        """
        src, dst = int(src), int(dst)
        try:
            path = nx.shortest_path(g_filtered, src, dst)
            self.stats["routed_on_filtered"] += 1
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        try:
            path = nx.shortest_path(g_full, src, dst)
            self.stats["fell_back_to_full"] += 1
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self.stats["no_path"] += 1
            return None

    @staticmethod
    def route_metrics(graph: nx.Graph, path) -> Dict[str, float]:
        return route_metrics(graph, path)

    def report(self) -> Dict[str, float]:
        s = self.stats
        routed = s["routed_on_filtered"] + s["fell_back_to_full"]
        return {
            "snapshots": int(s["snapshots"]),
            "routed_on_filtered": int(s["routed_on_filtered"]),
            "fell_back_to_full_graph": int(s["fell_back_to_full"]),
            "no_path": int(s["no_path"]),
            "fallback_rate": float(s["fell_back_to_full"] / routed) if routed else float("nan"),
            "mean_risky_node_fraction": (
                float(s["risky_nodes"] / s["nodes"]) if s["nodes"] else float("nan")
            ),
        }
