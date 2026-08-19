"""FRLFP router: exclude predicted risky nodes, then route by hop count.

When the SFRNNR flags most of the network the filtered graph disconnects and
this falls back to the full graph, which makes it identical to plain shortest
path. That used to happen silently inside a bare except, so a run where the
baseline degenerated on every decision looked like a run where it worked.
Fallbacks are counted and the rate is reported.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import networkx as nx

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

    def build_graphs(self, snapshot, radius: float | None = None):
        """Return (full graph, risk filtered graph, positions, risky nodes).

        Edge reliability follows the paper: one minus the worse of the two
        endpoints' predicted failure probabilities. The edge set matches our
        router's, so route differences come from the method, not the topology.
        """
        radius = CFG.communication_radius_default if radius is None else radius
        frame = snapshot.sort_values("node_id")
        for column in ("lfp", "lfp_threshold"):
            if column not in frame.columns:
                raise KeyError(
                    f"FRLFPRouter needs column '{column}'. Run pipeline/train_models.py "
                    "to attach SFRNNR outputs to the dataset."
                )

        lfp = {int(r.node_id): float(r.lfp) for r in frame.itertuples(index=False)}
        threshold = {int(r.node_id): float(r.lfp_threshold) for r in frame.itertuples(index=False)}
        risky: Set[NodeId] = {n for n in lfp if lfp[n] > threshold[n]}
        positions = {
            int(r.node_id): (float(r.x), float(r.y)) for r in frame.itertuples(index=False)
        }

        g_full, g_filtered = nx.Graph(), nx.Graph()
        for node_id in positions:
            g_full.add_node(node_id)
            if node_id not in risky:
                g_filtered.add_node(node_id)

        for u, v in geometric_edges(frame, radius):
            reliability = float(clip_reliability(1.0 - max(lfp[u], lfp[v])))
            g_full.add_edge(u, v, weight=1.0, reliability=reliability)
            if u not in risky and v not in risky:
                g_filtered.add_edge(u, v, weight=1.0, reliability=reliability)

        self.stats["snapshots"] += 1
        self.stats["risky_nodes"] += len(risky)
        self.stats["nodes"] += len(positions)
        return g_full, g_filtered, positions, risky

    def route(
        self, g_full: nx.Graph, g_filtered: nx.Graph, src: NodeId, dst: NodeId
    ) -> Optional[List[NodeId]]:
        """Route on the filtered graph, falling back to the full graph.

        Only counts a fallback when the filter is what broke routing. A pair
        disconnected in both graphs is counted as no_path, otherwise the
        fallback rate would be inflated by topology.
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
        stats = self.stats
        routed = stats["routed_on_filtered"] + stats["fell_back_to_full"]
        return {
            "snapshots": int(stats["snapshots"]),
            "routed_on_filtered": int(stats["routed_on_filtered"]),
            "fell_back_to_full_graph": int(stats["fell_back_to_full"]),
            "no_path": int(stats["no_path"]),
            "fallback_rate": float(stats["fell_back_to_full"] / routed) if routed else float("nan"),
            "mean_risky_node_fraction": (
                float(stats["risky_nodes"] / stats["nodes"]) if stats["nodes"] else float("nan")
            ),
        }
