import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from routing_from_dataset import DatasetRouter

DATASET_PATH = "dataset/manet_featured_dataset.csv"
OUTPUT_PATH = "results/ours_results.csv"
PAIRS_PER_STEP = 5
RADIUS = 150.0
SEED = 42


def main():
    os.makedirs("results", exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(DATASET_PATH)
    run_ids = sorted(df["run_id"].dropna().unique().astype(int).tolist())
    test_runs = sorted(random.Random(SEED).sample(run_ids, k=min(6, len(run_ids))))
    test = df[df["run_id"].isin(test_runs)]

    router = DatasetRouter()
    rec = []

    for run_id in sorted(test["run_id"].unique()):
        run_df = test[test["run_id"] == run_id]
        for t in sorted(run_df["time"].unique()):
            s = run_df[run_df["time"] == t].reset_index(drop=True)
            g_ml, g_base, _ = router.build_graph(s, radius=RADIUS)
            nodes = list(g_ml.nodes())
            if len(nodes) < 2 or g_ml.number_of_edges() == 0:
                continue

            tried = 0
            done = 0
            while done < PAIRS_PER_STEP and tried < 50:
                tried += 1
                src, dst = random.sample(nodes, 2)
                ml_path = router.find_ml_path(g_ml, src, dst)
                base_path = router.find_baseline_path(g_base, src, dst)
                if ml_path is None or base_path is None:
                    continue
                m = router.compute_route_metrics(g_ml, ml_path)
                b = router.compute_route_metrics(g_ml, base_path)
                rec.append(
                    {
                        "method": "ours",
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
    print(f"Saved {OUTPUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
