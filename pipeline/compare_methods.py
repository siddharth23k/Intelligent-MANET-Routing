"""Stage 6: the three way comparison.

Protocol, now enforced rather than assumed:
  - one shared dataset, one shared label, one shared run level split
  - every method routes over an identical geometric edge set per snapshot
  - every method gets the same sampled source destination pairs, and a decision
    is recorded only when all three produce a path

What is reported has changed. Scoring a route by the same reliabilities Dijkstra
minimised over makes our method incapable of losing, so alongside that metric we
report path survival (replayed from the mobility trace, invisible to any model),
win / tie / loss rates, the FRLFP fallback rate, and Wilcoxon next to the paired
t test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from config_loader import get_config  # noqa: E402
from frlfp_router import FRLFPRouter  # noqa: E402
from graph_build import geometric_edges  # noqa: E402
from metrics import paired_run_test, proportion_test, win_loss_tie  # noqa: E402
from path_survival import PositionLookup, evaluate_path_survival, summarise_survival  # noqa: E402
from routing_from_dataset import DatasetRouter  # noqa: E402
from splits import make_run_split  # noqa: E402

CFG = get_config()
DATASET = ROOT / "data" / "processed" / "paper_lfp_dataset.csv"
RESULTS = ROOT / "results"

METHODS = ("ours", "paper_baseline", "classic_baseline")
MAX_PAIR_ATTEMPTS = 100


def build_geometry_graph(snapshot, radius: float) -> nx.Graph:
    graph = nx.Graph()
    for node_id in snapshot["node_id"].astype(int):
        graph.add_node(int(node_id))
    for u, v in geometric_edges(snapshot, radius):
        graph.add_edge(u, v, weight=1.0)
    return graph


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_\n"
    columns = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    smoke_cfg = CFG.smoke
    parser = argparse.ArgumentParser(description="Compare ours, the paper baseline and shortest path.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    parser.add_argument("--snapshots-per-run", type=int, default=None, help="0 uses every snapshot")
    parser.add_argument("--pairs-per-step", type=int, default=None)
    args = parser.parse_args()

    smoke = args.smoke

    def pick(explicit, smoke_key, default):
        if explicit is not None:
            return int(explicit)
        return int(smoke_cfg[smoke_key]) if smoke else int(default)

    test_run_count = pick(args.test_run_count, "test_run_count", CFG.test_run_count)
    val_run_count = pick(args.val_run_count, "val_run_count", CFG.val_run_count)
    snapshots_per_run = pick(args.snapshots_per_run, "snapshots_per_run", 0)
    pairs_per_step = pick(args.pairs_per_step, "pairs_per_step", CFG.pairs_per_step)

    RESULTS.mkdir(parents=True, exist_ok=True)
    radius = CFG.communication_radius_default
    survival_horizon = CFG.survival_horizon

    df = pd.read_csv(args.dataset)
    run_ids = sorted(df["run_id"].dropna().unique().astype(int).tolist())
    split = make_run_split(
        run_ids, seed=args.seed, test_run_count=test_run_count, val_run_count=val_run_count
    )
    test = df[df["run_id"].isin(split.test_runs)].copy()

    # Survival is replayed against the full trace, including snapshots after the
    # end of the evaluated window.
    lookup = PositionLookup(df[["run_id", "time", "node_id", "x", "y"]])

    ours_router = DatasetRouter()
    paper_router = FRLFPRouter()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    rows: list[dict] = []
    skipped = {"no_edges": 0, "no_common_path": 0}

    for run_id in sorted(test["run_id"].unique()):
        run_df = test[test["run_id"] == run_id]
        times = sorted(run_df["time"].unique())
        if snapshots_per_run > 0:
            step = max(1, len(times) // snapshots_per_run)
            times = times[::step][:snapshots_per_run]

        for t in times:
            snapshot = run_df[run_df["time"] == t].reset_index(drop=True)
            geometry = build_geometry_graph(snapshot, radius)
            if geometry.number_of_edges() == 0 or geometry.number_of_nodes() < 2:
                skipped["no_edges"] += 1
                continue

            graphs = ours_router.build_graphs(snapshot, radius=radius)
            paper_full, paper_filtered, _positions, risky = paper_router.build_graphs(
                snapshot, radius=radius
            )
            nodes = list(geometry.nodes())

            recorded, attempts = 0, 0
            while recorded < pairs_per_step and attempts < MAX_PAIR_ATTEMPTS:
                attempts += 1
                src, dst = rng.sample(nodes, 2)

                classic_path = ours_router.find_baseline_path(graphs.hop, src, dst)
                ours_path = ours_router.find_ml_path(graphs.ml, src, dst)
                paper_path = paper_router.route(paper_full, paper_filtered, src, dst)
                if classic_path is None or ours_path is None or paper_path is None:
                    skipped["no_common_path"] += 1
                    continue

                row = {
                    "run_id": int(run_id),
                    "time": float(t),
                    "source": int(src),
                    "target": int(dst),
                    "risky_fraction": float(len(risky) / max(1, geometry.number_of_nodes())),
                }
                for name, path, graph in (
                    ("ours", ours_path, graphs.ml),
                    ("paper_baseline", paper_path, paper_full),
                    ("classic_baseline", classic_path, graphs.ml),
                ):
                    quality = ours_router.compute_route_metrics(graph, path)
                    row[f"{name}_avg_reliability"] = quality["avg_reliability"]
                    row[f"{name}_min_reliability"] = quality["min_reliability"]
                    row[f"{name}_hops"] = quality["hop_count"]

                    survival = evaluate_path_survival(
                        path, lookup, run_id, t, radius, survival_horizon
                    )
                    row[f"{name}_survived"] = survival["survived"]
                    row[f"{name}_evaluable"] = survival["evaluable"]
                    row[f"{name}_surviving_fraction"] = survival["surviving_fraction"]
                    row[f"{name}_broken_hops"] = survival["broken_hops"]
                    row[f"{name}_path_len"] = len(path)

                rows.append(row)
                recorded += 1

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        raise RuntimeError(
            f"no routing decisions recorded (skipped: {skipped}). "
            "Check the radius, the test runs and the dataset."
        )
    decisions.to_csv(RESULTS / "routing_decisions.csv", index=False)

    # --- model derived route quality, versus hop count on the same graph ----
    baseline = "classic_baseline"
    summaries = []
    for name in ("ours", "paper_baseline"):
        summary = {
            "method": name,
            "n_decisions": int(len(decisions)),
            "avg_reliability": float(decisions[f"{name}_avg_reliability"].mean()),
            "min_reliability": float(decisions[f"{name}_min_reliability"].mean()),
            "avg_hops": float(decisions[f"{name}_hops"].mean()),
            "baseline_avg_reliability": float(decisions[f"{baseline}_avg_reliability"].mean()),
            "baseline_min_reliability": float(decisions[f"{baseline}_min_reliability"].mean()),
            "baseline_avg_hops": float(decisions[f"{baseline}_hops"].mean()),
        }
        summary["delta_avg_rel_pct"] = (
            (summary["avg_reliability"] - summary["baseline_avg_reliability"])
            / max(abs(summary["baseline_avg_reliability"]), 1e-12) * 100.0
        )
        summary["delta_min_rel_pct"] = (
            (summary["min_reliability"] - summary["baseline_min_reliability"])
            / max(abs(summary["baseline_min_reliability"]), 1e-12) * 100.0
        )
        summary["delta_hops"] = summary["avg_hops"] - summary["baseline_avg_hops"]
        summary.update({
            f"wlt_{k}": v for k, v in win_loss_tie(
                decisions, f"{name}_avg_reliability", f"{baseline}_avg_reliability"
            ).items()
        })
        summary.update({
            f"stat_{k}": v for k, v in paired_run_test(
                decisions, f"{name}_avg_reliability", f"{baseline}_avg_reliability"
            ).items()
        })
        summaries.append(summary)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "comparison_metrics.csv", index=False)

    # --- ground truth: does the chosen path still exist H steps later? ------
    survival_rows = []
    for name in METHODS:
        records = decisions[[
            f"{name}_survived", f"{name}_evaluable",
            f"{name}_surviving_fraction", f"{name}_broken_hops",
        ]].rename(columns=lambda c: c.replace(f"{name}_", "")).to_dict("records")
        survival_rows.append({"method": name, **summarise_survival(records)})
    survival_df = pd.DataFrame(survival_rows)[
        ["method", "n_evaluable", "survival_rate", "mean_surviving_fraction", "mean_broken_hops"]
    ]
    survival_df.to_csv(RESULTS / "path_survival.csv", index=False)

    evaluable = decisions[decisions[f"{baseline}_evaluable"] == 1]
    survival_tests = []
    for name in ("ours", "paper_baseline"):
        subset = evaluable[evaluable[f"{name}_evaluable"] == 1]
        proportion = proportion_test(
            int(subset[f"{name}_survived"].sum()), int(len(subset)),
            int(subset[f"{baseline}_survived"].sum()), int(len(subset)),
        )
        by_run = paired_run_test(subset, f"{name}_survived", f"{baseline}_survived")
        survival_tests.append(
            {"method": name, **proportion, **{f"run_{k}": v for k, v in by_run.items()}}
        )
    pd.DataFrame(survival_tests).to_csv(RESULTS / "path_survival_tests.csv", index=False)

    baseline_report = paper_router.report()
    diagnostics = {
        "smoke": bool(smoke),
        "seed": int(args.seed),
        "split": split.as_dict(),
        "radius_m": radius,
        "survival_horizon_steps": survival_horizon,
        "snapshots_per_run": snapshots_per_run or "all",
        "pairs_per_step": pairs_per_step,
        "decisions": int(len(decisions)),
        "skipped": skipped,
        "frlfp": baseline_report,
        "ours_routing_failures": ours_router.failure_report(),
    }
    with open(RESULTS / "comparison_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    summary_lines = [
        "# Comparison Summary",
        "",
        f"Held out runs: {split.test_runs}. Decisions: {len(decisions)}. "
        f"Radius: {radius} m. Survival horizon: {survival_horizon} steps.",
        "",
        "## 1. Ground truth: path survival",
        "",
        "Fraction of chosen routes whose every hop is still within radius "
        "`survival_horizon` steps later, replayed from the mobility trace. No model "
        "output enters this number.",
        "",
        df_to_markdown(survival_df),
        "",
        "## 2. Model derived route quality (report with care)",
        "",
        "This is the metric the base paper reports. Our router minimises the sum of "
        "-log of the same reliabilities averaged here, so the optimiser and the scorer "
        "are the same function. A win rate near 1.0 with a loss rate of 0.0 is a "
        "property of that algebra, not evidence of predictive skill. Read section 1 first.",
        "",
        df_to_markdown(summary_df),
        "",
        "## 3. Baseline health",
        "",
        f"- FRLFP flagged on average {baseline_report['mean_risky_node_fraction']:.1%} of nodes as risky.",
        f"- FRLFP fell back to the unfiltered graph on {baseline_report['fallback_rate']:.1%} of "
        "routing attempts. At a high fallback rate it is indistinguishable from hop count "
        "shortest path and any comparison against it is meaningless.",
        f"- Our router failed to find a path: {diagnostics['ours_routing_failures'] or 'none'}.",
        f"- Skipped snapshots or pairs: {skipped}.",
        "",
        "## 4. Notes",
        "",
        "- All three methods route over an identical edge set and identical sampled pairs.",
        "- Significance is tested at the run level: decisions inside one run share almost "
        "all of their topology.",
        "- Per decision records are in `results/routing_decisions.csv`.",
    ]
    with open(RESULTS / "comparison_summary.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print(f"[compare_methods] {len(decisions)} decisions over runs {split.test_runs}")
    print("[compare_methods] path survival (ground truth):")
    for _, row in survival_df.iterrows():
        print(f"    {row['method']:<18s} survival={row['survival_rate']:.4f} "
              f"hops_intact={row['mean_surviving_fraction']:.4f} n={int(row['n_evaluable'])}")
    print("[compare_methods] model derived route quality vs shortest path:")
    for _, row in summary_df.iterrows():
        print(f"    {row['method']:<18s} avg_rel={row['avg_reliability']:.4f} "
              f"({row['delta_avg_rel_pct']:+.2f}%) hops={row['avg_hops']:.2f} "
              f"win={row['wlt_win_rate']:.2f} tie={row['wlt_tie_rate']:.2f} "
              f"loss={row['wlt_loss_rate']:.2f}")
    print(f"[compare_methods] FRLFP fallback rate: {baseline_report['fallback_rate']:.1%}, "
          f"risky node fraction: {baseline_report['mean_risky_node_fraction']:.1%}")
    print(f"[compare_methods] wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
