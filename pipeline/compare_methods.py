import os
import random
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent / "methods/ours"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "methods/baseline"))
from routing_from_dataset import DatasetRouter
from frlfp_router import FRLFPRouter


def _run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _summarize_relative_to_baseline(df, method_name):
    d = df[df["method"] == method_name].copy()
    if d.empty:
        return None
    return {
        "method": method_name,
        "n_samples": len(d),
        "avg_reliability": float(d["ml_avg_reliability"].mean()),
        "min_reliability": float(d["ml_min_reliability"].mean()),
        "avg_hops": float(d["ml_hops"].mean()),
        "baseline_avg_reliability": float(d["base_avg_reliability"].mean()),
        "baseline_min_reliability": float(d["base_min_reliability"].mean()),
        "baseline_avg_hops": float(d["base_hops"].mean()),
        "delta_avg_rel_pct": float(
            (d["ml_avg_reliability"].mean() - d["base_avg_reliability"].mean())
            / max(d["base_avg_reliability"].mean(), 1e-12)
            * 100.0
        ),
        "delta_min_rel_pct": float(
            (d["ml_min_reliability"].mean() - d["base_min_reliability"].mean())
            / max(d["base_min_reliability"].mean(), 1e-12)
            * 100.0
        ),
        "delta_hops": float(d["ml_hops"].mean() - d["base_hops"].mean()),
    }


def _paired_test(df, method_name):
    d = df[df["method"] == method_name].copy()
    if d.empty:
        return None
    by_run = d.groupby("run_id")[["ml_avg_reliability", "base_avg_reliability"]].mean()
    if len(by_run) < 2:
        return {"method": method_name, "p_value": np.nan, "n_runs": int(len(by_run))}
    _, p = stats.ttest_rel(by_run["ml_avg_reliability"], by_run["base_avg_reliability"])
    return {"method": method_name, "p_value": float(p), "n_runs": int(len(by_run))}


def _df_to_simple_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_\n"
    cols = [str(c) for c in df.columns]
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def _build_geometry_graph(snapshot, radius):
    rows = snapshot.to_dict("records")
    g = nx.Graph()
    for r in rows:
        g.add_node(int(r["node_id"]))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            u, v = int(rows[i]["node_id"]), int(rows[j]["node_id"])
            d = float(np.sqrt((rows[i]["x"] - rows[j]["x"]) ** 2 + (rows[i]["y"] - rows[j]["y"]) ** 2))
            if d <= radius:
                g.add_edge(u, v, weight=1.0)
    return g


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)

    # Build paper-style dataset artifacts (single shared dataset for ALL methods)
    # _run("python pipeline/generate_data.py")
    # _run("python pipeline/engineer_features.py")
    # _run("python pipeline/train_models.py")

    # Train our model on the same shared paper dataset.
    # _run("python pipeline/train_models.py --dataset data/processed/paper_lfp_dataset.csv")

    # Unified evaluation loop on identical sampled decisions.
    df = pd.read_csv("data/processed/paper_lfp_dataset.csv")
    run_ids = sorted(df["run_id"].dropna().unique().astype(int).tolist())
    seed = 42
    radius = 150.0
    pairs_per_step = 5
    test_runs = sorted(random.Random(seed).sample(run_ids, k=min(6, len(run_ids))))
    test = df[df["run_id"].isin(test_runs)].copy()

    ours_router = DatasetRouter()
    paper_router = FRLFPRouter()

    ours_rows = []
    paper_rows = []
    classic_rows = []

    random.seed(seed)
    np.random.seed(seed)

    for run_id in sorted(test["run_id"].unique()):
        run_df = test[test["run_id"] == run_id]
        for t in sorted(run_df["time"].unique()):
            snap = run_df[run_df["time"] == t].reset_index(drop=True)

            g_geom = _build_geometry_graph(snap, radius)
            if g_geom.number_of_edges() == 0 or g_geom.number_of_nodes() < 2:
                continue

            g_ours, _, _ = ours_router.build_graph(snap, radius=radius)
            g_paper_full, g_paper_filtered, _, _ = paper_router.build_graphs(snap, radius=radius)
            nodes = list(g_geom.nodes())

            tried = 0
            done = 0
            while done < pairs_per_step and tried < 100:
                tried += 1
                src, dst = random.sample(nodes, 2)

                try:
                    base_path = nx.shortest_path(g_geom, src, dst)
                except Exception:
                    continue

                ours_path = ours_router.find_ml_path(g_ours, src, dst)
                paper_path = paper_router.route(g_paper_full, g_paper_filtered, src, dst)
                if ours_path is None or paper_path is None:
                    continue

                ours_m = ours_router.compute_route_metrics(g_ours, ours_path)
                ours_b = ours_router.compute_route_metrics(g_ours, base_path)
                paper_m = paper_router.route_metrics(g_paper_full, paper_path)
                paper_b = paper_router.route_metrics(g_paper_full, base_path)

                ours_rows.append(
                    {
                        "method": "ours",
                        "run_id": run_id,
                        "time": t,
                        "source": src,
                        "target": dst,
                        "ml_avg_reliability": ours_m["avg_reliability"],
                        "ml_min_reliability": ours_m["min_reliability"],
                        "ml_hops": ours_m["hop_count"],
                        "base_avg_reliability": ours_b["avg_reliability"],
                        "base_min_reliability": ours_b["min_reliability"],
                        "base_hops": ours_b["hop_count"],
                    }
                )
                paper_rows.append(
                    {
                        "method": "paper_baseline",
                        "run_id": run_id,
                        "time": t,
                        "source": src,
                        "target": dst,
                        "ml_avg_reliability": paper_m["avg_reliability"],
                        "ml_min_reliability": paper_m["min_reliability"],
                        "ml_hops": paper_m["hop_count"],
                        "base_avg_reliability": paper_b["avg_reliability"],
                        "base_min_reliability": paper_b["min_reliability"],
                        "base_hops": paper_b["hop_count"],
                    }
                )
                classic_rows.append(
                    {"method": "classic_baseline", "run_id": run_id, "time": t, "source": src, "target": dst, "hop_count": len(base_path) - 1}
                )
                done += 1

    ours = pd.DataFrame(ours_rows)
    paper = pd.DataFrame(paper_rows)
    classic = pd.DataFrame(classic_rows)
    ours.to_csv("results/ours_results.csv", index=False)
    paper.to_csv("results/paper_baseline_results.csv", index=False)
    classic.to_csv("results/classic_baseline_results.csv", index=False)

    all_rel = pd.concat([ours, paper], ignore_index=True)

    summaries = []
    for m in ["ours", "paper_baseline"]:
        s = _summarize_relative_to_baseline(all_rel, m)
        if s is not None:
            summaries.append(s)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("results/comparison_metrics.csv", index=False)

    tests = []
    for m in ["ours", "paper_baseline"]:
        t = _paired_test(all_rel, m)
        if t is not None:
            tests.append(t)
    pd.DataFrame(tests).to_csv("results/stat_tests.csv", index=False)

    with open("results/comparison_summary.md", "w", encoding="utf-8") as f:
        f.write("# Comparison Summary\n\n")
        if not summary_df.empty:
            f.write(_df_to_simple_markdown(summary_df))
            f.write("\n\n")
        else:
            f.write("No summary rows generated.\n\n")
        f.write("## Notes\n")
        f.write("- `paper_baseline` is a paper-inspired FRLFP re-implementation.\n")
        f.write("- `ours` is the current ensemble + reliability-weighted routing method.\n")
        f.write("- `classic_baseline` output is available in `results/classic_baseline_results.csv`.\n")

    print("Saved:")
    print(" - results/comparison_metrics.csv")
    print(" - results/stat_tests.csv")
    print(" - results/comparison_summary.md")


if __name__ == "__main__":
    main()
