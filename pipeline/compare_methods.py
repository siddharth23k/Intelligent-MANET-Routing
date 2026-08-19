"""Stage 6: the three way comparison.

Protocol, unchanged in intent from the original design and now enforced rather
than assumed:

  * one shared dataset, one shared label definition, one shared run level split
  * for every held out run and every snapshot, the same geometric edge set is
    used by all three methods, so routes differ only because of the method
  * the same sampled source destination pairs are given to all three methods,
    and a decision is only recorded when all three produce a path

What is new is what gets measured. The original scored routes by the mean and
minimum of the same reliabilities Dijkstra had just minimised over, which made
our method mathematically incapable of losing. That metric is still reported,
because it is what the base paper reports, but it is now reported next to:

  * path survival, replayed from the mobility trace, which no model can see
  * win / tie / loss rates, so a suspiciously perfect win rate is visible
  * the FRLFP fallback rate, so a baseline that has silently degenerated into
    shortest path cannot be presented as a fuzzy neural competitor
  * Wilcoxon alongside the paired t test, since the number of independent units
    is the number of runs, not the number of decisions
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


def build_geometry_graph(snapshot, radius: float) -> nx.Graph:
    g = nx.Graph()
    for nid in snapshot["node_id"].astype(int):
        g.add_node(int(nid))
    for u, v in geometric_edges(snapshot, radius):
        g.add_edge(u, v, weight=1.0)
    return g


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_\n"
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    sm = CFG.smoke
    parser = argparse.ArgumentParser(description="Compare ours, the paper baseline and shortest path.")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    parser.add_argument("--snapshots-per-run", type=int, default=None,
                        help="0 means every snapshot in the run")
    parser.add_argument("--pairs-per-step", type=int, default=None)
    args = parser.parse_args()

    smoke = args.smoke
    test_run_count = args.test_run_count if args.test_run_count is not None else int(
        sm["test_run_count"] if smoke else CFG.test_run_count
    )
    val_run_count = args.val_run_count if args.val_run_count is not None else int(
        sm["val_run_count"] if smoke else CFG.val_run_count
    )
    snapshots_per_run = args.snapshots_per_run if args.snapshots_per_run is not None else int(
        sm["snapshots_per_run"] if smoke else 0
    )
    pairs_per_step = args.pairs_per_step if args.pairs_per_step is not None else int(
        sm["pairs_per_step"] if smoke else CFG.pairs_per_step
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    radius = CFG.communication_radius_default
    survival_horizon = CFG.survival_horizon

    df = pd.read_csv(args.dataset)
    run_ids = sorted(df["run_id"].dropna().unique().astype(int).tolist())
    split = make_run_split(run_ids, seed=args.seed, test_run_count=test_run_count, val_run_count=val_run_count)
    test = df[df["run_id"].isin(split.test_runs)].copy()

    # The survival metric is replayed against the full trace, including snapshots
    # after the end of the evaluated window.
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
            snap = run_df[run_df["time"] == t].reset_index(drop=True)
            g_geom = build_geometry_graph(snap, radius)
            if g_geom.number_of_edges() == 0 or g_geom.number_of_nodes() < 2:
                skipped["no_edges"] += 1
                continue

            graphs = ours_router.build_graphs(snap, radius=radius)
            g_paper_full, g_paper_filtered, _pos, risky = paper_router.build_graphs(snap, radius=radius)
            nodes = list(g_geom.nodes())

            done, tried = 0, 0
            while done < pairs_per_step and tried < 100:
                tried += 1
                src, dst = rng.sample(nodes, 2)

                classic_path = ours_router.find_baseline_path(graphs.hop, src, dst)
                ours_path = ours_router.find_ml_path(graphs.ml, src, dst)
                paper_path = paper_router.route(g_paper_full, g_paper_filtered, src, dst)
                if classic_path is None or ours_path is None or paper_path is None:
                    skipped["no_common_path"] += 1
                    continue

                row = {
                    "run_id": int(run_id),
                    "time": float(t),
                    "source": int(src),
                    "target": int(dst),
                    "risky_fraction": float(len(risky) / max(1, g_geom.number_of_nodes())),
                }
                for name, path, graph in (
                    ("ours", ours_path, graphs.ml),
                    ("paper_baseline", paper_path, g_paper_full),
                    ("classic_baseline", classic_path, graphs.ml),
                ):
                    m = ours_router.compute_route_metrics(graph, path)
                    row[f"{name}_avg_reliability"] = m["avg_reliability"]
                    row[f"{name}_min_reliability"] = m["min_reliability"]
                    row[f"{name}_hops"] = m["hop_count"]
                    s = evaluate_path_survival(path, lookup, run_id, t, radius, survival_horizon)
                    row[f"{name}_survived"] = s["survived"]
                    row[f"{name}_evaluable"] = s["evaluable"]
                    row[f"{name}_surviving_fraction"] = s["surviving_fraction"]
                    row[f"{name}_broken_hops"] = s["broken_hops"]
                    row[f"{name}_path_len"] = len(path)

                rows.append(row)
                done += 1

    decisions = pd.DataFrame(rows)
    if decisions.empty:
        raise RuntimeError(
            f"no routing decisions recorded (skipped: {skipped}). "
            "Check the radius, the test runs and the dataset."
        )
    decisions.to_csv(RESULTS / "routing_decisions.csv", index=False)

    # ---- model derived route quality, versus hop count on the same graph ----
    summaries = []
    for name in ("ours", "paper_baseline"):
        base = "classic_baseline"
        d = decisions
        s = {
            "method": name,
            "n_decisions": int(len(d)),
            "avg_reliability": float(d[f"{name}_avg_reliability"].mean()),
            "min_reliability": float(d[f"{name}_min_reliability"].mean()),
            "avg_hops": float(d[f"{name}_hops"].mean()),
            "baseline_avg_reliability": float(d[f"{base}_avg_reliability"].mean()),
            "baseline_min_reliability": float(d[f"{base}_min_reliability"].mean()),
            "baseline_avg_hops": float(d[f"{base}_hops"].mean()),
        }
        denom = max(abs(s["baseline_avg_reliability"]), 1e-12)
        s["delta_avg_rel_pct"] = (s["avg_reliability"] - s["baseline_avg_reliability"]) / denom * 100.0
        denom_min = max(abs(s["baseline_min_reliability"]), 1e-12)
        s["delta_min_rel_pct"] = (s["min_reliability"] - s["baseline_min_reliability"]) / denom_min * 100.0
        s["delta_hops"] = s["avg_hops"] - s["baseline_avg_hops"]
        s.update({f"wlt_{k}": v for k, v in win_loss_tie(
            d, f"{name}_avg_reliability", f"{base}_avg_reliability"
        ).items()})
        s.update({f"stat_{k}": v for k, v in paired_run_test(
            d, f"{name}_avg_reliability", f"{base}_avg_reliability"
        ).items()})
        summaries.append(s)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(RESULTS / "comparison_metrics.csv", index=False)

    # ---- ground truth: does the chosen path still exist H steps later? -----
    survival_rows = []
    for name in METHODS:
        recs = decisions[[f"{name}_survived", f"{name}_evaluable",
                          f"{name}_surviving_fraction", f"{name}_broken_hops"]].rename(
            columns={
                f"{name}_survived": "survived",
                f"{name}_evaluable": "evaluable",
                f"{name}_surviving_fraction": "surviving_fraction",
                f"{name}_broken_hops": "broken_hops",
            }
        ).to_dict("records")
        s = summarise_survival(recs)
        s["method"] = name
        survival_rows.append(s)
    survival_df = pd.DataFrame(survival_rows)[
        ["method", "n_evaluable", "survival_rate", "mean_surviving_fraction", "mean_broken_hops"]
    ]
    survival_df.to_csv(RESULTS / "path_survival.csv", index=False)

    ev = decisions[decisions["classic_baseline_evaluable"] == 1]
    survival_tests = []
    for name in ("ours", "paper_baseline"):
        sub = ev[ev[f"{name}_evaluable"] == 1]
        test_res = proportion_test(
            int(sub[f"{name}_survived"].sum()), int(len(sub)),
            int(sub["classic_baseline_survived"].sum()), int(len(sub)),
        )
        run_res = paired_run_test(sub, f"{name}_survived", "classic_baseline_survived")
        survival_tests.append({"method": name, **test_res, **{f"run_{k}": v for k, v in run_res.items()}})
    pd.DataFrame(survival_tests).to_csv(RESULTS / "path_survival_tests.csv", index=False)

    baseline_report = paper_router.report()
    ours_failures = ours_router.failure_report()
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
        "ours_routing_failures": ours_failures,
    }
    with open(RESULTS / "comparison_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # ---- human readable summary -------------------------------------------
    lines = ["# Comparison Summary", ""]
    lines.append(f"Held out runs: {split.test_runs}. Decisions: {len(decisions)}. "
                 f"Radius: {radius} m. Survival horizon: {survival_horizon} steps.")
    lines.append("")
    lines.append("## 1. Ground truth: path survival")
    lines.append("")
    lines.append("Fraction of chosen routes whose every hop is still within radius "
                 "`survival_horizon` steps later, replayed from the mobility trace. "
                 "No model output enters this number.")
    lines.append("")
    lines.append(df_to_markdown(survival_df))
    lines.append("")
    lines.append("## 2. Model derived route quality (report with care)")
    lines.append("")
    lines.append("This is the metric the base paper reports. Note that our router "
                 "minimises the sum of -log of the same reliabilities that are averaged "
                 "here, so the optimiser and the scorer are the same function. A win rate "
                 "near 1.0 with a loss rate of 0.0 is a property of that algebra, not "
                 "evidence of predictive skill. Read section 1 first.")
    lines.append("")
    lines.append(df_to_markdown(summary_df))
    lines.append("")
    lines.append("## 3. Baseline health")
    lines.append("")
    lines.append(f"- FRLFP flagged on average {baseline_report['mean_risky_node_fraction']:.1%} "
                 f"of nodes as risky.")
    lines.append(f"- FRLFP fell back to the unfiltered graph on "
                 f"{baseline_report['fallback_rate']:.1%} of routing attempts. "
                 "At a high fallback rate the paper baseline is indistinguishable from "
                 "hop count shortest path and any comparison against it is meaningless.")
    lines.append(f"- Our router failed to find a path: {ours_failures or 'none'}.")
    lines.append(f"- Skipped snapshots or pairs: {skipped}.")
    lines.append("")
    lines.append("## 4. Notes")
    lines.append("")
    lines.append("- All three methods route over an identical geometric edge set and an "
                 "identical set of sampled source destination pairs.")
    lines.append("- Significance is tested at the run level, not the decision level: "
                 "decisions inside one run share almost all of their topology.")
    lines.append("- Per decision records are in `results/routing_decisions.csv`.")
    with open(RESULTS / "comparison_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[compare_methods] {len(decisions)} decisions over runs {split.test_runs}")
    print("[compare_methods] path survival (ground truth):")
    for _, r in survival_df.iterrows():
        print(f"    {r['method']:<18s} survival={r['survival_rate']:.4f} "
              f"hops_intact={r['mean_surviving_fraction']:.4f} n={int(r['n_evaluable'])}")
    print("[compare_methods] model derived route quality vs shortest path:")
    for _, r in summary_df.iterrows():
        print(f"    {r['method']:<18s} avg_rel={r['avg_reliability']:.4f} "
              f"({r['delta_avg_rel_pct']:+.2f}%) hops={r['avg_hops']:.2f} "
              f"win={r['wlt_win_rate']:.2f} tie={r['wlt_tie_rate']:.2f} loss={r['wlt_loss_rate']:.2f}")
    print(f"[compare_methods] FRLFP fallback rate: {baseline_report['fallback_rate']:.1%}, "
          f"risky node fraction: {baseline_report['mean_risky_node_fraction']:.1%}")
    print(f"[compare_methods] wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
