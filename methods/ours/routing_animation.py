import sys
import random
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from routing_from_dataset import DatasetRouter

class MANETAnimation:
    def __init__(self, dataset_path, start_time=10.0, run_id=None, pair_change_every=10, min_hops=3, random_seed=42):
        self.df = pd.read_csv(dataset_path)

        self.router = DatasetRouter()

        self.random = random.Random(int(random_seed))
        self.pair_change_every = int(pair_change_every)
        self.min_hops = int(min_hops)

        all_runs = sorted(self.df["run_id"].dropna().unique().astype(int).tolist()) if "run_id" in self.df.columns else []
        if run_id is None:
            self.run_id = int(all_runs[0]) if all_runs else None
        else:
            self.run_id = int(run_id)

        if self.run_id is not None and "run_id" in self.df.columns:
            self.df = self.df[self.df["run_id"] == self.run_id].copy()
            
        all_times = sorted(self.df["time"].unique())
        self.times = [t for t in all_times if t >= float(start_time)]

        if not self.times:
                        self.times = all_times

        self.ml_scores = []
        self.baseline_scores = []
        self.time_points = []

        self.source = None
        self.target = None

        self._apply_cluster_skip(std_threshold=50.0)

    def _choose_source_target(self):
        raise NotImplementedError("Use `_pick_pair_for_snapshot` (dynamic per timestep).")

    def _apply_cluster_skip(self, std_threshold: float = 50.0):
        kept = []
        for t in self.times:
            snap = self.df[self.df["time"] == t]
            if snap.empty:
                continue
            if float(snap["x"].std(ddof=0)) > float(std_threshold):
                kept.append(t)
        if kept and len(kept) != len(self.times):
            first = float(kept[0])
            self.times = kept

    def _pick_pair_for_snapshot(self, G: nx.Graph):
        """
        Pick a random (src,dst) in the current graph with a shortest-path length >= min_hops.
        Returns (src, dst, hop_len) or (None, None, None) if not found.
        """
        if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
            return None, None, None

        # Work within the largest connected component.
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        if not components or len(components[0]) < 2:
            return None, None, None

        comp_nodes = list(components[0])
        H = G.subgraph(comp_nodes)

        # Try a bounded number of random attempts to find a multi-hop pair.
        for _ in range(200):
            src, dst = self.random.sample(comp_nodes, 2)
            try:
                hop_len = nx.shortest_path_length(H, src, dst)
            except Exception:
                continue
            if hop_len >= self.min_hops:
                return int(src), int(dst), int(hop_len)

        return None, None, None

    def path_reliability(self, G, path):
        if not path or len(path) < 2:
            return None
        rels = [G[u][v]["reliability"] for u, v in zip(path[:-1], path[1:])]
        return np.mean(rels)

    def _draw_filtered_edges(self, ax, G: nx.Graph, pos, min_rel: float = 0.3, top_k: int = 4):
        """
        Visual-only edge filtering:
        - keep only edges with reliability >= min_rel
        - for each node, keep only its top_k most reliable incident edges
        """
        # Remove self-loops defensively
        edges = [(u, v, d) for u, v, d in G.edges(data=True) if u != v]
        if not edges:
            return

        # Keep edges above threshold
        edges = [(u, v, d) for u, v, d in edges if float(d.get("reliability", 0.0)) >= float(min_rel)]
        if not edges:
            return

        # Per-node top-k
        by_node = {}
        for u, v, d in edges:
            r = float(d.get("reliability", 0.0))
            by_node.setdefault(u, []).append((r, u, v))
            by_node.setdefault(v, []).append((r, u, v))

        keep = set()
        for n, lst in by_node.items():
            for r, u, v in sorted(lst, key=lambda x: x[0], reverse=True)[: int(top_k)]:
                keep.add((min(u, v), max(u, v)))

        filtered = []
        colors = []
        for u, v, d in edges:
            key = (min(u, v), max(u, v))
            if key not in keep:
                continue
            r = float(d.get("reliability", 0.0))
            color = "green" if r > 0.7 else "orange" if r > 0.4 else "red"
            filtered.append((u, v))
            colors.append(color)

        if not filtered:
            return

        nx.draw_networkx_edges(G, pos, edgelist=filtered, edge_color=colors, width=1.4, alpha=0.35, ax=ax)

    def _draw_path_polyline(self, ax, pos, path, color: str, label: str):
        if not path or len(path) < 2:
            return
        xs = [pos[n][0] for n in path if n in pos]
        ys = [pos[n][1] for n in path if n in pos]
        if len(xs) < 2:
            return
        ax.plot(xs, ys, color=color, linewidth=3.2, alpha=0.95, zorder=5)
        # Direction arrows for each segment
        for (x0, y0, x1, y1) in zip(xs[:-1], ys[:-1], xs[1:], ys[1:]):
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2, alpha=0.9),
                zorder=6,
            )
        ax.text(xs[0], ys[0], label, fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85),
                zorder=7)

    def animate(self):
        fig, ax = plt.subplots(figsize=(11, 8))

        pad = 30
        x_min, x_max = self.df["x"].min() - pad, self.df["x"].max() + pad
        y_min, y_max = self.df["y"].min() - pad, self.df["y"].max() + pad

        def update(frame):
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            t = self.times[frame]
            snapshot = self.df[self.df["time"] == t]

            # Filter nodes stuck at (0,0) (should largely be solved by the sim fix, but keep as guard)
            snapshot = snapshot[(snapshot["x"] != 0) | (snapshot["y"] != 0)]
            if snapshot.empty:
                return

            # Use the same connectivity radius as evaluation (avoid mismatched graph density).
            G_ml, G_base, pos = self.router.build_graph(snapshot, radius=150.0)

            # Change (src,dst) every K timesteps to showcase diverse scenarios
            if (frame % self.pair_change_every == 0) or (self.source is None) or (self.target is None):
                src, dst, hop_len = self._pick_pair_for_snapshot(G_base)
                if src is not None and dst is not None:
                    self.source, self.target = src, dst

            source, target = self.source, self.target
            ml_path = self.router.find_ml_path(G_ml, source, target) if source is not None else None
            base_path = self.router.find_baseline_path(G_base, source, target) if source is not None else None

            # Reliability tracking
            ml_rel = self.path_reliability(G_ml, ml_path)
            baseline_rel = self.path_reliability(G_base, base_path)

            if ml_rel is not None and baseline_rel is not None:
                self.ml_scores.append(ml_rel)
                self.baseline_scores.append(baseline_rel)
                self.time_points.append(t)

            # Draw filtered edges (visual-only) to avoid hairball.
            self._draw_filtered_edges(ax, G_ml, pos, min_rel=0.3, top_k=4)

            # Draw paths as full polylines with direction arrows (multi-hop visible).
            if base_path:
                self._draw_path_polyline(
                    ax, pos, base_path, color="#ff33aa",
                    label=f"BASE {source}→{target} via {len(base_path)-1} hops"
                )
            if ml_path:
                self._draw_path_polyline(
                    ax, pos, ml_path, color="blue",
                    label=f"ML {source}→{target} via {len(ml_path)-1} hops"
                )

            nx.draw_networkx_nodes(
                G_ml, pos,
                node_size=300, node_color="#9ecae1",
                edgecolors="black", linewidths=0.8, ax=ax
            )
            nx.draw_networkx_labels(G_ml, pos, font_size=9, ax=ax)

            # Mark source/destination
            if source is not None and source in pos:
                ax.scatter([pos[source][0]], [pos[source][1]], s=380, marker="*", c="green", edgecolors="black", zorder=8)
            if target is not None and target in pos:
                ax.scatter([pos[target][0]], [pos[target][1]], s=380, marker="*", c="red", edgecolors="black", zorder=8)

            if not ml_path or not base_path:
                ax.text(
                    0.5, 0.05,
                    f"No path found for {'ML' if not ml_path else ''}{' and ' if (not ml_path and not base_path) else ''}{'Baseline' if not base_path else ''}",
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95)
                )

            legend_text = (
                f"TIME: {t:.1f}s\n\n"
                f"SRC→DST: {source}→{target}\n"
                "BLUE : ML Path (Reliable)\n"
                "PINK : Baseline (Shortest)\n"
                "GREEN: High Reliability (>0.7)\n"
                "ORANGE: Medium Reliability\n"
                "RED  : Link Failure Risk"
            )
            ax.text(
                0.02, 0.98, legend_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9)
            )

            ax.set_title("Intelligent MANET Routing Animation", fontsize=14)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.grid(True, linestyle="--", alpha=0.3)

        anim = FuncAnimation(fig, update, frames=len(self.times), interval=300, repeat=False)
        plt.show()

        # Reliability comparison plot
        if self.time_points:
            fig, ax = plt.subplots(figsize=(10, 5))

            ml = np.array(self.ml_scores)
            base = np.array(self.baseline_scores)
            times = np.array(self.time_points)

            ax.plot(times, ml, label="ML Routing", color="blue", linewidth=2)
            ax.plot(times, base, label="Baseline Routing", color="#ff66cc", linewidth=2)

            ax.fill_between(times, base, ml, where=(ml >= base),
                            alpha=0.15, color="blue", label="ML Advantage")

            worst_idx = np.argmin(base)
            ax.annotate(
                f"Baseline drops to {base[worst_idx]:.2f}",
                xy=(times[worst_idx], base[worst_idx]),
                xytext=(times[worst_idx] + 2, base[worst_idx] + 0.1),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=9
            )

            ax.axhline(np.mean(ml), color="blue", linestyle="--", alpha=0.4,
                       label=f"ML Mean: {np.mean(ml):.2f}")
            ax.axhline(np.mean(base), color="#ff66cc", linestyle="--", alpha=0.4,
                       label=f"Baseline Mean: {np.mean(base):.2f}")

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Average Route Reliability")
            ax.set_title("ML Routing vs Baseline Routing Reliability")
            ax.legend(loc="lower right")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("assets/reliability_vs_time.jpg", dpi=150)
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate MANET routing over a featured dataset.")
    parser.add_argument(
        "--dataset",
        default="dataset/paper/processed/paper_lfp_dataset.csv",
        help="Path to featured dataset CSV (default: shared paper dataset).",
    )
    parser.add_argument("--start-time", type=float, default=10.0, help="Starting time for animation.")
    parser.add_argument("--run-id", type=int, default=None, help="Specific run_id to animate.")
    parser.add_argument("--pair-change-every", type=int, default=10, help="Change src/dst pair every N frames.")
    parser.add_argument("--min-hops", type=int, default=3, help="Minimum hop count when sampling src/dst.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for pair sampling.")
    args = parser.parse_args()

    animation = MANETAnimation(
        args.dataset,
        start_time=args.start_time,
        run_id=args.run_id,
        pair_change_every=args.pair_change_every,
        min_hops=args.min_hops,
        random_seed=args.seed,
    )
    animation.animate()