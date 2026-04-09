import os
import random

import networkx as nx
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/paper/processed/paper_lfp_dataset.csv"
OUTPUT_PATH = "results/classic_baseline_results.csv"
PAIRS_PER_STEP = 5
RADIUS = 150.0
SEED = 42


def build_graph(snapshot, radius):
    rows = snapshot.to_dict("records")
    g = nx.Graph()
    for r in rows:
        g.add_node(int(r["node_id"]))
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            u, v = int(rows[i]["node_id"]), int(rows[j]["node_id"])
            d = np.sqrt((rows[i]["x"] - rows[j]["x"]) ** 2 + (rows[i]["y"] - rows[j]["y"]) ** 2)
            if d <= radius:
                g.add_edge(u, v, weight=1.0)
    return g


def main():
    os.makedirs("results", exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(DATASET_PATH)
    run_ids = sorted(df["run_id"].unique().tolist())
    test_runs = sorted(random.Random(SEED).sample(run_ids, k=min(6, len(run_ids))))
    test = df[df["run_id"].isin(test_runs)]

    rec = []
    for run_id in sorted(test["run_id"].unique()):
        run_df = test[test["run_id"] == run_id]
        for t in sorted(run_df["time"].unique()):
            s = run_df[run_df["time"] == t].reset_index(drop=True)
            g = build_graph(s, RADIUS)
            nodes = list(g.nodes())
            if len(nodes) < 2 or g.number_of_edges() == 0:
                continue

            tried = 0
            done = 0
            while done < PAIRS_PER_STEP and tried < 50:
                tried += 1
                src, dst = random.sample(nodes, 2)
                try:
                    path = nx.shortest_path(g, src, dst)
                except Exception:
                    continue
                rec.append({"method": "classic_baseline", "run_id": run_id, "time": t, "hop_count": len(path) - 1})
                done += 1

    out = pd.DataFrame(rec)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
