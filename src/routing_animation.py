import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from routing_from_dataset import DatasetRouter

class MANETAnimation:
    def __init__(self, dataset_path, start_time=10.0):
        print("Loading dataset...")
        self.df = pd.read_csv(dataset_path)

        print("Loading ML predictor...")
        self.router = DatasetRouter()  # use DatasetRouter instead of raw predictor

        all_times = sorted(self.df["time"].unique())
        self.times = [t for t in all_times if t >= start_time]

        if not self.times:
            print("Warning: No data found after start_time. Using all data.")
            self.times = all_times

        self.ml_scores = []
        self.baseline_scores = []
        self.time_points = []

    def path_reliability(self, G, path):
        if not path or len(path) < 2:
            return None
        rels = [G[u][v]["reliability"] for u, v in zip(path[:-1], path[1:])]
        return np.mean(rels)

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

            # Filter nodes stuck at (0,0)
            snapshot = snapshot[(snapshot["x"] != 0) | (snapshot["y"] != 0)]
            if snapshot.empty:
                return

            # Use DatasetRouter to build graph (better averaged features + radius=65)
            G_ml, G_base, pos = self.router.build_graph(snapshot, radius=65)

            source, target = 0, 5
            ml_path = self.router.find_ml_path(G_ml, source, target)
            base_path = self.router.find_baseline_path(G_base, source, target)

            # Reliability tracking
            ml_rel = self.path_reliability(G_ml, ml_path)
            baseline_rel = self.path_reliability(G_base, base_path)

            if ml_rel is not None and baseline_rel is not None:
                self.ml_scores.append(ml_rel)
                self.baseline_scores.append(baseline_rel)
                self.time_points.append(t)

            # Draw edges colored by reliability (use G_ml for display)
            edge_colors = []
            for u, v, data in G_ml.edges(data=True):
                r = data["reliability"]
                color = "green" if r > 0.75 else "orange" if r > 0.4 else "red"
                edge_colors.append(color)

            nx.draw_networkx_edges(
                G_ml, pos,
                edge_color=edge_colors,
                width=1.2, alpha=0.4, ax=ax
            )

            # Baseline path
            if base_path:
                nx.draw_networkx_edges(
                    G_ml, pos,
                    edgelist=list(zip(base_path[:-1], base_path[1:])),
                    width=3.5, edge_color="#ff66cc", alpha=0.9, ax=ax
                )

            # ML path
            if ml_path:
                nx.draw_networkx_edges(
                    G_ml, pos,
                    edgelist=list(zip(ml_path[:-1], ml_path[1:])),
                    width=4.0, edge_color="blue", ax=ax
                )

            nx.draw_networkx_nodes(
                G_ml, pos,
                node_size=300, node_color="#9ecae1",
                edgecolors="black", linewidths=0.8, ax=ax
            )
            nx.draw_networkx_labels(G_ml, pos, font_size=9, ax=ax)

            legend_text = (
                f"TIME: {t:.1f}s\n\n"
                "BLUE : ML Path (Reliable)\n"
                "PINK : Baseline (Shortest)\n"
                "GREEN: High Reliability (>0.75)\n"
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
    animation = MANETAnimation("dataset/manet_featured_dataset.csv", start_time=10.0)
    animation.animate()