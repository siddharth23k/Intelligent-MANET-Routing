import os
import subprocess

import numpy as np
import pandas as pd
from scipy import stats


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


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("assets/comparison", exist_ok=True)

    # Build paper-style dataset artifacts
    _run("python scripts/paper_build_dataset.py")
    _run("python scripts/paper_feature_engineering.py")
    _run("python scripts/paper_compute_lfp.py")

    # Ensure our model artifacts exist
    if not os.path.exists("models/random_forest.pkl") or not os.path.exists("models/xgboost_model.pkl"):
        _run("python experiments/training.py")

    # Evaluate all methods
    _run("python experiments/evaluate_ours.py")
    _run("python experiments/evaluate_paper_baseline.py")
    _run("python experiments/evaluate_classic_baseline.py")

    ours = pd.read_csv("results/ours_results.csv")
    paper = pd.read_csv("results/paper_baseline_results.csv")
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
