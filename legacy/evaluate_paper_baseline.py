import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "baseline_paper"))
from frlfp_router import FRLFPRouter

DATASET_PATH = "dataset/paper/processed/paper_lfp_dataset.csv"
OUTPUT_PATH = "results/paper_baseline_results.csv"
PAIRS_PER_STEP = 5
RADIUS = 150.0
SEED = 42


def main():
    os.makedirs("results", exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(DATASET_PATH)
    run_ids = sorted(df["run_id"].unique().tolist())
    test_runs = sorted(random.Random(SEED).sample(run_ids, k=min(6, len(run_ids))))
    test = df[df["run_id"].isin(test_runs)]

    router = FRLFPRouter()
    rec = []

    for run_id in sorted(test["run_id"].unique()):
        run_df = test[test["run_id"] == run_id]
        for t in sorted(run_df["time"].unique()):
            s = run_df[run_df["time"] == t].reset_index(drop=True)
            g_full, g_filtered, _, _ = router.build_graphs(s, radius=RADIUS)
            nodes = list(g_full.nodes())
            if len(nodes) < 2 or g_full.number_of_edges() == 0:
                continue

            tried = 0
            done = 0
            while done < PAIRS_PER_STEP and tried < 50:
                tried += 1
                src, dst = random.sample(nodes, 2)
                path = router.route(g_full, g_filtered, src, dst)
                if path is None:
                    continue

                try:
                    base_path = nx.shortest_path(g_full, src, dst)
                except Exception:
                    continue

                m = router.route_metrics(g_full, path)
                b = router.route_metrics(g_full, base_path)
                rec.append(
                    {
                        "method": "paper_baseline",
                        "run_id": run_id,
                        "time": t,
                        "ml_avg_reliability": m["avg_reliability"],
                        "ml_min_reliability": m["min_reliability"],
                        "ml_hops": m["hop_count"],
                        "base_avg_reliability": b["avg_reliability"],
                        "base_min_reliability": b["min_reliability"],
                        "base_hops": b["hop_count"],
                    }
                )
                done += 1

    out = pd.DataFrame(rec)
    out.to_csv(OUTPUT_PATH, index=False)
    

if __name__ == "__main__":
    main()
