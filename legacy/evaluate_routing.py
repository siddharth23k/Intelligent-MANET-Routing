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

DATASET_PATH   = "dataset/manet_featured_dataset.csv"
TEST_RUNS      = None                        # chosen reproducibly from dataset run_ids
PAIRS_PER_STEP = 5                           # source-dest pairs per timestep
RADIUS         = 150.0                       # communication radius (metres)
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs("assets", exist_ok=True)


router = DatasetRouter()

df = pd.read_csv(DATASET_PATH)

run_ids = sorted(df["run_id"].dropna().unique().astype(int).tolist())
rng = random.Random(RANDOM_SEED)
test_runs = sorted(rng.sample(run_ids, k=min(6, len(run_ids))))

test_df    = df[df["run_id"].isin(test_runs)].copy()
all_times  = sorted(test_df["time"].unique())
all_runs   = sorted(test_df["run_id"].unique())

records = []

for run_id in all_runs:
    run_df = test_df[test_df["run_id"] == run_id]
    times  = sorted(run_df["time"].unique())

    for t in times:
        snapshot = run_df[run_df["time"] == t].copy().reset_index(drop=True)

        try:
            G_ml, G_base, pos = router.build_graph(snapshot, radius=RADIUS)
        except Exception as e:
            # This captures the shape mismatch if predict.py isn't fixed yet
            print(f"    t={t}: graph build failed ({e}), skipping.")
            continue

        if G_ml.number_of_edges() == 0:
            continue

        nodes = list(G_ml.nodes())
        if len(nodes) < 2:
            continue

        # Try random source-destination pairs until we get PAIRS_PER_STEP valid routes
        pairs_tried = 0
        pairs_done  = 0

        while pairs_done < PAIRS_PER_STEP and pairs_tried < 50:
            pairs_tried += 1
            src, dst = random.sample(nodes, 2)

            ml_path   = router.find_ml_path(G_ml, src, dst)
            base_path = router.find_baseline_path(G_base, src, dst)

            # Only record if BOTH paths exist 
            if ml_path is None or base_path is None:
                continue

            ml_m   = router.compute_route_metrics(G_ml, ml_path)
            base_m = router.compute_route_metrics(G_ml, base_path)

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

        results = pd.DataFrame(records)

ml_avg_rel   = results["ml_avg_reliability"].mean()
base_avg_rel = results["base_avg_reliability"].mean()
ml_min_rel   = results["ml_min_reliability"].mean()
base_min_rel = results["base_min_reliability"].mean()
ml_hops      = results["ml_hops"].mean()
base_hops    = results["base_hops"].mean()

rel_improvement = (ml_avg_rel - base_avg_rel) / base_avg_rel * 100
min_improvement = (ml_min_rel - base_min_rel) / base_min_rel * 100

t_by_run = results.groupby("run_id")[["ml_avg_reliability", "base_avg_reliability"]].mean()
t_stat, p_value = stats.ttest_rel(t_by_run["ml_avg_reliability"], t_by_run["base_avg_reliability"])


# 4. Plots (Section unchanged but ensured output path exists)
time_series = results.groupby("time").agg(
    ml_avg_rel   = ("ml_avg_reliability",  "mean"),
    base_avg_rel = ("base_avg_reliability","mean"),
    ml_min_rel   = ("ml_min_reliability",  "mean"),
    base_min_rel = ("base_min_reliability","mean"),
    ml_hops      = ("ml_hops",             "mean"),
    base_hops    = ("base_hops",           "mean"),
).reset_index()

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)
ML_COLOR, BASE_COLOR = "#2196F3", "#9C27B0"

ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(time_series["time"], time_series["ml_avg_rel"], color=ML_COLOR, lw=2.5, label="ML Routing")
ax1.plot(time_series["time"], time_series["base_avg_rel"], color=BASE_COLOR, lw=2.5, linestyle="--", label="Baseline")
ax1.set_title("Average Route Reliability over Time", fontweight="bold")
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
ax2.bar([str(r) for r in per_run.index], per_run["improvement_%"], color=ML_COLOR)
ax2.set_title("Per-Run Improvement (%)", fontweight="bold")

plot_path = "assets/routing_evaluation.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
results.to_csv("dataset/routing_results.csv", index=False)