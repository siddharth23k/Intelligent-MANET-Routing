# ═══════════════════════════════════════════════════════════════════════════════
# evaluate_routing.py
# -------------------
# Systematic evaluation of ML-based routing vs hop-count baseline routing
# across all test simulation runs and all timesteps.
#
# What this script does:
#   1. Loads the featured dataset and restricts to TEST runs (same as training.py)
#   2. For every timestep in every test run, picks 5 random source-destination pairs
#   3. Runs both ML routing and baseline routing on each pair
#   4. Records: avg_reliability, min_reliability, hop_count for each
#   5. Generates 6 plots:
#        - Reliability over time (ML vs baseline)
#        - Hop count over time
#        - Min reliability (bottleneck link) over time
#        - Distribution of avg_reliability (histogram)
#        - Distribution of hop count (histogram)
#        - Per-run reliability improvement bar chart
#   6. Prints a clean results table
#
# Run from project root:
#   python notebooks/evaluate_routing.py
# ═══════════════════════════════════════════════════════════════════════════════

import sys
import os
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats   # for paired t-test

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from routing_from_dataset import DatasetRouter

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "dataset/manet_featured_dataset.csv"
TEST_RUNS      = [25, 26, 27, 28, 29, 30]   # must match training.py
PAIRS_PER_STEP = 5                           # source-dest pairs per timestep
RADIUS         = 250.0                       # communication radius (metres)
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs("assets", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Initialise
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("MANET Routing Evaluation — ML vs Hop-Count Baseline")
print("=" * 65)

router = DatasetRouter()

print(f"\nLoading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

test_df    = df[df["run_id"].isin(TEST_RUNS)].copy()
all_times  = sorted(test_df["time"].unique())
all_runs   = sorted(test_df["run_id"].unique())

print(f"  Test runs      : {all_runs}")
print(f"  Timesteps      : {len(all_times)} ({all_times[0]} → {all_times[-1]})")
print(f"  Total test rows: {len(test_df)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Running routing evaluation...")
print("=" * 65)

records = []   # one record per (run_id, time, pair_index)

for run_id in all_runs:
    run_df = test_df[test_df["run_id"] == run_id]
    times  = sorted(run_df["time"].unique())
    print(f"\n  Run {run_id} — {len(times)} timesteps")

    for t in times:
        snapshot = run_df[run_df["time"] == t].copy().reset_index(drop=True)

        try:
            G_ml, G_base, pos = router.build_graph(snapshot, radius=RADIUS)
        except Exception as e:
            print(f"    t={t}: graph build failed ({e}), skipping.")
            continue

        if G_ml.number_of_edges() == 0:
            continue

        nodes = list(G_ml.nodes())
        if len(nodes) < 2:
            continue

        # Generate PAIRS_PER_STEP random source-destination pairs
        pairs_tried = 0
        pairs_done  = 0

        while pairs_done < PAIRS_PER_STEP and pairs_tried < 50:
            pairs_tried += 1
            src, dst = random.sample(nodes, 2)

            ml_path   = router.find_ml_path(G_ml, src, dst)
            base_path = router.find_baseline_path(G_base, src, dst)

            # Only record if BOTH paths exist (fair comparison)
            if ml_path is None or base_path is None:
                continue

            ml_m   = router.compute_route_metrics(G_ml, ml_path)
            base_m = router.compute_route_metrics(G_ml, base_path)
            # Note: base_path metrics are also evaluated on G_ml
            # (same reliability values) — this is the correct comparison.
            # The baseline path was *selected* by hop count but *evaluated*
            # by reliability, same as the ML path.

            records.append({
                "run_id"             : run_id,
                "time"               : t,
                "source"             : src,
                "target"             : dst,
                "ml_avg_reliability" : ml_m["avg_reliability"],
                "ml_min_reliability" : ml_m["min_reliability"],
                "ml_hops"            : ml_m["hop_count"],
                "base_avg_reliability": base_m["avg_reliability"],
                "base_min_reliability": base_m["min_reliability"],
                "base_hops"          : base_m["hop_count"],
            })
            pairs_done += 1

        print(f"    t={t:.1f}: {pairs_done} pairs evaluated, "
              f"edges={G_ml.number_of_edges()}", end="\r")

print(f"\n\nTotal routing decisions recorded: {len(records)}")

if len(records) == 0:
    print("ERROR: No records collected. Check dataset path, run IDs, and radius.")
    sys.exit(1)

results = pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Aggregate metrics
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Aggregate Results")
print("=" * 65)

ml_avg_rel   = results["ml_avg_reliability"].mean()
base_avg_rel = results["base_avg_reliability"].mean()
ml_min_rel   = results["ml_min_reliability"].mean()
base_min_rel = results["base_min_reliability"].mean()
ml_hops      = results["ml_hops"].mean()
base_hops    = results["base_hops"].mean()

rel_improvement = (ml_avg_rel - base_avg_rel) / base_avg_rel * 100
min_improvement = (ml_min_rel - base_min_rel) / base_min_rel * 100

print(f"\n  {'Metric':<30} {'Baseline':>10} {'ML Routing':>12} {'Δ':>8}")
print(f"  {'─'*30} {'─'*10} {'─'*12} {'─'*8}")
print(f"  {'Avg Route Reliability':<30} {base_avg_rel:>10.4f} {ml_avg_rel:>12.4f} "
      f"{rel_improvement:>+7.1f}%")
print(f"  {'Min Link Reliability':<30} {base_min_rel:>10.4f} {ml_min_rel:>12.4f} "
      f"{min_improvement:>+7.1f}%")
print(f"  {'Avg Hop Count':<30} {base_hops:>10.4f} {ml_hops:>12.4f} "
      f"{ml_hops - base_hops:>+8.2f}")

# Statistical significance — paired t-test on per-decision reliability
t_stat, p_value = stats.ttest_rel(
    results["ml_avg_reliability"],
    results["base_avg_reliability"]
)
print(f"\n  Paired t-test (avg reliability):")
print(f"    t-statistic : {t_stat:.4f}")
print(f"    p-value     : {p_value:.6f}")
if p_value < 0.05:
    print(f"    Result      : SIGNIFICANT (p < 0.05) ✓")
else:
    print(f"    Result      : not significant (p ≥ 0.05)")

# Per-run summary
print(f"\n  Per-run average reliability:")
per_run = results.groupby("run_id")[
    ["ml_avg_reliability", "base_avg_reliability"]
].mean()
per_run["improvement_%"] = (
    (per_run["ml_avg_reliability"] - per_run["base_avg_reliability"])
    / per_run["base_avg_reliability"] * 100
)
print(per_run.round(4).to_string())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Time-aggregated series (for time-series plots)
# ═══════════════════════════════════════════════════════════════════════════════
time_series = results.groupby("time").agg(
    ml_avg_rel   = ("ml_avg_reliability",  "mean"),
    base_avg_rel = ("base_avg_reliability","mean"),
    ml_min_rel   = ("ml_min_reliability",  "mean"),
    base_min_rel = ("base_min_reliability","mean"),
    ml_hops      = ("ml_hops",             "mean"),
    base_hops    = ("base_hops",           "mean"),
).reset_index()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Plots
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Generating evaluation plots...")
print("=" * 65)

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

ML_COLOR   = "#2196F3"
BASE_COLOR = "#9C27B0"

# ── Plot 1: Avg Reliability over time ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(time_series["time"], time_series["ml_avg_rel"],
         color=ML_COLOR,   lw=2.5, label=f"ML Routing (mean={ml_avg_rel:.3f})")
ax1.plot(time_series["time"], time_series["base_avg_rel"],
         color=BASE_COLOR, lw=2.5, linestyle="--",
         label=f"Baseline (mean={base_avg_rel:.3f})")
ax1.fill_between(time_series["time"],
                 time_series["ml_avg_rel"],
                 time_series["base_avg_rel"],
                 where=time_series["ml_avg_rel"] >= time_series["base_avg_rel"],
                 alpha=0.15, color=ML_COLOR, label="ML advantage region")
ax1.set_xlabel("Simulation Timestep")
ax1.set_ylabel("Average Route Reliability")
ax1.set_title("Average Route Reliability over Time", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 1.05)

# ── Plot 2: Per-run improvement bar ───────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
runs       = per_run.index.tolist()
bar_colors = [ML_COLOR if v > 0 else "#F44336"
              for v in per_run["improvement_%"]]
bars = ax2.bar([str(r) for r in runs],
               per_run["improvement_%"],
               color=bar_colors, edgecolor="white")
ax2.axhline(0, color="gray", lw=1)
ax2.set_xlabel("Run ID")
ax2.set_ylabel("Reliability Improvement (%)")
ax2.set_title("Per-Run Reliability Improvement\n(ML vs Baseline)",
              fontsize=13, fontweight="bold")
for bar, val in zip(bars, per_run["improvement_%"]):
    ypos = bar.get_height() + 0.1 if val >= 0 else bar.get_height() - 0.5
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f"{val:+.1f}%", ha="center", va="bottom", fontsize=9)
ax2.grid(axis="y", alpha=0.3)

# ── Plot 3: Min reliability over time ─────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_series["time"], time_series["ml_min_rel"],
         color=ML_COLOR,   lw=2.5, label=f"ML (mean={ml_min_rel:.3f})")
ax3.plot(time_series["time"], time_series["base_min_rel"],
         color=BASE_COLOR, lw=2.5, linestyle="--",
         label=f"Baseline (mean={base_min_rel:.3f})")
ax3.set_xlabel("Simulation Timestep")
ax3.set_ylabel("Min Link Reliability (Bottleneck)")
ax3.set_title("Bottleneck Link Reliability over Time",
              fontsize=13, fontweight="bold")
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_ylim(0, 1.05)

# ── Plot 4: Hop count over time ───────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_series["time"], time_series["ml_hops"],
         color=ML_COLOR,   lw=2.5, label=f"ML (mean={ml_hops:.2f})")
ax4.plot(time_series["time"], time_series["base_hops"],
         color=BASE_COLOR, lw=2.5, linestyle="--",
         label=f"Baseline (mean={base_hops:.2f})")
ax4.set_xlabel("Simulation Timestep")
ax4.set_ylabel("Average Hop Count")
ax4.set_title("Average Hop Count over Time", fontsize=13, fontweight="bold")
ax4.legend()
ax4.grid(alpha=0.3)

# ── Plot 5: Reliability distribution histogram ────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(results["ml_avg_reliability"],   bins=40, alpha=0.65,
         color=ML_COLOR,   label="ML Routing", density=True)
ax5.hist(results["base_avg_reliability"], bins=40, alpha=0.65,
         color=BASE_COLOR, label="Baseline",   density=True)
ax5.axvline(ml_avg_rel,   color=ML_COLOR,   lw=2, linestyle="--")
ax5.axvline(base_avg_rel, color=BASE_COLOR, lw=2, linestyle="--")
ax5.set_xlabel("Average Route Reliability")
ax5.set_ylabel("Density")
ax5.set_title("Reliability Distribution\n(all routing decisions)",
              fontsize=13, fontweight="bold")
ax5.legend()
ax5.grid(alpha=0.3)

plt.suptitle(
    f"Routing Evaluation Results  |  "
    f"ML avg_rel={ml_avg_rel:.3f}  Baseline avg_rel={base_avg_rel:.3f}  "
    f"Improvement={rel_improvement:+.1f}%",
    fontsize=14, fontweight="bold", y=1.01
)

plot_path = "assets/routing_evaluation.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Plots saved to {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Save results CSV
# ═══════════════════════════════════════════════════════════════════════════════
results_path = "dataset/routing_results.csv"
results.to_csv(results_path, index=False)
print(f"  Detailed results saved to {results_path}")

print("\n" + "=" * 65)
print("EVALUATION COMPLETE")
print("=" * 65)
print(f"  Total routing decisions : {len(results)}")
print(f"  Avg reliability — ML    : {ml_avg_rel:.4f}")
print(f"  Avg reliability — Base  : {base_avg_rel:.4f}")
print(f"  Improvement             : {rel_improvement:+.2f}%")
print(f"  Min reliability — ML    : {ml_min_rel:.4f}")
print(f"  Min reliability — Base  : {base_min_rel:.4f}")
print(f"  Avg hops — ML           : {ml_hops:.3f}")
print(f"  Avg hops — Baseline     : {base_hops:.3f}")
print(f"  Statistical significance: p={p_value:.6f}")
