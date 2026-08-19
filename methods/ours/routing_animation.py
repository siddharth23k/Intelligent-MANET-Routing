"""Animated view of reliability weighted routing against hop count routing.

Visualisation only. Nothing here feeds the reported numbers; those come from
pipeline/compare_methods.py.

    python methods/ours/routing_animation.py --run-id 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from config_loader import get_config  # noqa: E402
from routing_from_dataset import DatasetRouter  # noqa: E402

CFG = get_config()
DEFAULT_RADIUS = CFG.communication_radius_default
ASSETS_DIR = ROOT / "assets"

ML_COLOUR = "#1f77b4"
BASE_COLOUR = "#ff33aa"
PAIR_SEARCH_ATTEMPTS = 200


class MANETAnimation:
    def __init__(
        self,
        dataset_path: str | Path,
        start_time: float = 10.0,
        run_id: Optional[int] = None,
        pair_change_every: int = 10,
        min_hops: int = 3,
        radius: float = DEFAULT_RADIUS,
        random_seed: int = 42,
    ):
        self.df = pd.read_csv(dataset_path)
        self.router = DatasetRouter()
        self.random = random.Random(int(random_seed))
        self.pair_change_every = max(1, int(pair_change_every))
        self.min_hops = max(1, int(min_hops))
        self.radius = float(radius)

        if "run_id" in self.df.columns:
            available = sorted(self.df["run_id"].dropna().unique().astype(int).tolist())
            self.run_id = int(run_id) if run_id is not None else (available[0] if available else None)
            if self.run_id is not None:
                self.df = self.df[self.df["run_id"] == self.run_id].copy()
        else:
            self.run_id = None

        all_times = sorted(self.df["time"].unique())
        self.times = [t for t in all_times if t >= float(start_time)] or all_times
        self._drop_clustered_snapshots()

        self.source: Optional[int] = None
        self.target: Optional[int] = None
        self.ml_scores: List[float] = []
        self.baseline_scores: List[float] = []
        self.time_points: List[float] = []

    def _drop_clustered_snapshots(self, std_threshold: float = 50.0) -> None:
        """Skip snapshots where every node sits in one cluster; nothing to see."""
        kept = [
            t for t in self.times
            if not self.df[self.df["time"] == t].empty
            and float(self.df[self.df["time"] == t]["x"].std(ddof=0)) > std_threshold
        ]
        if kept:
            self.times = kept

    def _pick_pair(self, graph: nx.Graph) -> Tuple[Optional[int], Optional[int]]:
        """Random (src, dst) in the largest component with at least min_hops hops."""
        if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
            return None, None
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        if not components or len(components[0]) < 2:
            return None, None

        nodes = list(components[0])
        component = graph.subgraph(nodes)
        for _ in range(PAIR_SEARCH_ATTEMPTS):
            src, dst = self.random.sample(nodes, 2)
            try:
                hops = nx.shortest_path_length(component, src, dst)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if hops >= self.min_hops:
                return int(src), int(dst)
        return None, None

    @staticmethod
    def path_reliability(graph: nx.Graph, path) -> Optional[float]:
        if not path or len(path) < 2:
            return None
        return float(np.mean([graph[u][v]["reliability"] for u, v in zip(path[:-1], path[1:])]))

    @staticmethod
    def _draw_edges(ax, graph: nx.Graph, positions, min_reliability: float = 0.3, top_k: int = 4) -> None:
        """Draw a readable subset of edges: above a threshold, top k per node."""
        edges = [
            (u, v, data)
            for u, v, data in graph.edges(data=True)
            if u != v and float(data.get("reliability", 0.0)) >= min_reliability
        ]
        if not edges:
            return

        by_node: dict = {}
        for u, v, data in edges:
            reliability = float(data.get("reliability", 0.0))
            by_node.setdefault(u, []).append((reliability, u, v))
            by_node.setdefault(v, []).append((reliability, u, v))

        keep = set()
        for incident in by_node.values():
            for _, u, v in sorted(incident, key=lambda item: item[0], reverse=True)[:top_k]:
                keep.add((min(u, v), max(u, v)))

        drawn, colours = [], []
        for u, v, data in edges:
            if (min(u, v), max(u, v)) not in keep:
                continue
            reliability = float(data.get("reliability", 0.0))
            drawn.append((u, v))
            colours.append("green" if reliability > 0.7 else "orange" if reliability > 0.4 else "red")

        if drawn:
            nx.draw_networkx_edges(
                graph, positions, edgelist=drawn, edge_color=colours, width=1.4, alpha=0.35, ax=ax
            )

    @staticmethod
    def _draw_path(ax, positions, path, colour: str, label: str) -> None:
        if not path or len(path) < 2:
            return
        xs = [positions[n][0] for n in path if n in positions]
        ys = [positions[n][1] for n in path if n in positions]
        if len(xs) < 2:
            return
        ax.plot(xs, ys, color=colour, linewidth=3.2, alpha=0.95, zorder=5)
        for x0, y0, x1, y1 in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=colour, lw=2.2, alpha=0.9), zorder=6,
            )
        ax.text(
            xs[0], ys[0], label, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85), zorder=7,
        )

    def _update(self, ax, frame: int, limits) -> None:
        (x_min, x_max), (y_min, y_max) = limits
        ax.clear()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        t = self.times[frame]
        snapshot = self.df[self.df["time"] == t]
        snapshot = snapshot[(snapshot["x"] != 0) | (snapshot["y"] != 0)]
        if snapshot.empty:
            return

        g_ml, g_hop, positions = self.router.build_graph(snapshot, radius=self.radius)

        if frame % self.pair_change_every == 0 or self.source is None or self.target is None:
            src, dst = self._pick_pair(g_hop)
            if src is not None:
                self.source, self.target = src, dst

        ml_path = self.router.find_ml_path(g_ml, self.source, self.target) if self.source is not None else None
        base_path = self.router.find_baseline_path(g_hop, self.source, self.target) if self.source is not None else None

        ml_reliability = self.path_reliability(g_ml, ml_path)
        base_reliability = self.path_reliability(g_ml, base_path)
        if ml_reliability is not None and base_reliability is not None:
            self.ml_scores.append(ml_reliability)
            self.baseline_scores.append(base_reliability)
            self.time_points.append(float(t))

        self._draw_edges(ax, g_ml, positions)
        if base_path:
            self._draw_path(ax, positions, base_path, BASE_COLOUR,
                            f"BASE {self.source}->{self.target}, {len(base_path) - 1} hops")
        if ml_path:
            self._draw_path(ax, positions, ml_path, ML_COLOUR,
                            f"ML {self.source}->{self.target}, {len(ml_path) - 1} hops")

        nx.draw_networkx_nodes(
            g_ml, positions, node_size=300, node_color="#9ecae1",
            edgecolors="black", linewidths=0.8, ax=ax,
        )
        nx.draw_networkx_labels(g_ml, positions, font_size=9, ax=ax)

        for node, colour in ((self.source, "green"), (self.target, "red")):
            if node is not None and node in positions:
                ax.scatter([positions[node][0]], [positions[node][1]], s=380,
                           marker="*", c=colour, edgecolors="black", zorder=8)

        if not ml_path or not base_path:
            missing = " and ".join(n for n, p in (("ML", ml_path), ("baseline", base_path)) if not p)
            ax.text(0.5, 0.05, f"No path found for {missing}", transform=ax.transAxes,
                    horizontalalignment="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.95))

        ax.text(
            0.02, 0.98,
            f"time {t:.1f}s\n"
            f"src -> dst: {self.source} -> {self.target}\n"
            "blue: reliability weighted path\n"
            "pink: hop count path\n"
            "edge colour: green >0.7, orange >0.4, red below",
            transform=ax.transAxes, verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )
        ax.set_title("Intelligent MANET Routing", fontsize=14)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.grid(True, linestyle="--", alpha=0.3)

    def animate(self, save_summary: bool = True) -> None:
        fig, ax = plt.subplots(figsize=(11, 8))
        pad = 30
        limits = (
            (self.df["x"].min() - pad, self.df["x"].max() + pad),
            (self.df["y"].min() - pad, self.df["y"].max() + pad),
        )

        # Keep a reference so the animation is not garbage collected before show.
        self._anim = FuncAnimation(
            fig, lambda frame: self._update(ax, frame, limits),
            frames=len(self.times), interval=300, repeat=False,
        )
        plt.show()

        if save_summary and self.time_points:
            self._plot_reliability_over_time()

    def _plot_reliability_over_time(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        times = np.array(self.time_points)
        ml = np.array(self.ml_scores)
        base = np.array(self.baseline_scores)

        ax.plot(times, ml, label="reliability weighted", color=ML_COLOUR, linewidth=2)
        ax.plot(times, base, label="hop count", color=BASE_COLOUR, linewidth=2)
        ax.fill_between(times, base, ml, where=(ml >= base), alpha=0.15, color=ML_COLOUR)
        ax.axhline(float(np.mean(ml)), color=ML_COLOUR, linestyle="--", alpha=0.4,
                   label=f"mean {np.mean(ml):.2f}")
        ax.axhline(float(np.mean(base)), color=BASE_COLOUR, linestyle="--", alpha=0.4,
                   label=f"mean {np.mean(base):.2f}")

        ax.set_xlabel("time (s)")
        ax.set_ylabel("average route reliability")
        ax.set_title("Route reliability over time (both scored on the same graph)")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        output = ASSETS_DIR / "reliability_vs_time.jpg"
        plt.tight_layout()
        plt.savefig(output, dpi=150)
        print(f"[routing_animation] wrote {output}")
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate MANET routing over a featured dataset.")
    parser.add_argument("--dataset", default=str(ROOT / "data" / "processed" / "paper_lfp_dataset.csv"))
    parser.add_argument("--start-time", type=float, default=10.0)
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--pair-change-every", type=int, default=10)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    args = parser.parse_args()

    MANETAnimation(
        args.dataset,
        start_time=args.start_time,
        run_id=args.run_id,
        pair_change_every=args.pair_change_every,
        min_hops=args.min_hops,
        radius=args.radius,
        random_seed=args.seed,
    ).animate()


if __name__ == "__main__":
    main()
