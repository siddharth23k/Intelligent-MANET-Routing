"""End to end smoke test.

Runs every stage of the pipeline on a deliberately tiny configuration and
asserts that each one produced the artifact the next stage depends on. The point
is to prove the chain executes and the contracts between stages hold, not to
produce meaningful numbers. Every stage is capped so the whole thing finishes in
seconds, which makes it usable as a pre commit check and in CI.

Numbers produced by a smoke run are not results. The model sees a couple of
simulation runs and a handful of epochs. Anything printed here is a liveness
signal.

    python pipeline/smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from config_loader import get_config  # noqa: E402

CFG = get_config()
SM = CFG.smoke

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "results" / "models"
RESULTS = ROOT / "results"


class SmokeFailure(RuntimeError):
    pass


def run_stage(name: str, argv: list[str], expect: list[Path], timeout: int = 300) -> float:
    print(f"\n=== {name} ===", flush=True)
    started = time.perf_counter()
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONWARNINGS="ignore")
    proc = subprocess.run(
        [sys.executable, *argv], cwd=str(ROOT), env=env, timeout=timeout,
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    for line in proc.stdout.splitlines():
        if line.startswith("[") or line.startswith("    "):
            print("   " + line.rstrip())
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-4000:])
        raise SmokeFailure(f"{name} exited with code {proc.returncode}")

    missing = [p for p in expect if not Path(p).exists()]
    if missing:
        raise SmokeFailure(f"{name} did not produce: {[str(m) for m in missing]}")
    print(f"   ok in {elapsed:.1f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole pipeline in miniature.")
    parser.add_argument("--max-runs", type=int, default=int(SM["max_runs"]))
    parser.add_argument("--max-rows-per-run", type=int, default=int(SM["max_rows_per_run"]))
    parser.add_argument("--budget-seconds", type=float, default=30.0)
    args = parser.parse_args()

    test_runs = str(SM["test_run_count"])
    val_runs = str(SM["val_run_count"])
    split_args = ["--test-run-count", test_runs, "--val-run-count", val_runs]

    total = 0.0
    total += run_stage(
        "1/6 generate_data",
        ["pipeline/generate_data.py",
         "--max-runs", str(args.max_runs),
         "--max-rows-per-run", str(args.max_rows_per_run)],
        [PROC / "paper_raw_dataset.csv", PROC / "dataset_manifest.json"],
    )
    total += run_stage(
        "2/6 engineer_features",
        ["pipeline/engineer_features.py", *split_args],
        [PROC / "paper_featured_dataset.csv", MODELS / "feature_norm_stats.json"],
    )
    total += run_stage(
        "3/6 validate_dataset",
        ["pipeline/validate_dataset.py", "--smoke", "--allow-run-level-traffic", *split_args],
        [RESULTS / "data_quality_report.json"],
    )
    total += run_stage(
        "4/6 train_predictor (ours)",
        ["pipeline/train_predictor.py", "--smoke", *split_args],
        [MODELS / "random_forest.pkl", MODELS / "xgboost_model.pkl",
         MODELS / "scaler.pkl", MODELS / "ensemble_weights.pkl",
         MODELS / "predictor_schema.json", RESULTS / "predictor_metrics.json"],
    )
    total += run_stage(
        "5/6 train_models (paper baseline)",
        ["pipeline/train_models.py", "--smoke", "--retrain", *split_args],
        [MODELS / "sfrnnr_paper.keras", MODELS / "sfrnnr_meta.json",
         PROC / "paper_lfp_dataset.csv"],
    )
    total += run_stage(
        "6/6 compare_methods",
        ["pipeline/compare_methods.py", "--smoke", *split_args],
        [RESULTS / "comparison_metrics.csv", RESULTS / "path_survival.csv",
         RESULTS / "routing_decisions.csv", RESULTS / "comparison_summary.md",
         RESULTS / "comparison_diagnostics.json"],
    )

    # Contract checks that a stage returning zero could still violate.
    with open(RESULTS / "predictor_metrics.json", encoding="utf-8") as f:
        pm = json.load(f)
    integrity = pm["integrity"]
    if integrity["rf_class"] == integrity["xgb_class"]:
        raise SmokeFailure("ensemble members are the same model class")
    if integrity["max_abs_probability_difference"] <= 0:
        raise SmokeFailure("ensemble members are indistinguishable; the blend is a no op")

    with open(RESULTS / "comparison_diagnostics.json", encoding="utf-8") as f:
        diag = json.load(f)
    if diag["decisions"] < 1:
        raise SmokeFailure("comparison recorded no routing decisions")
    sp = diag["split"]
    if set(sp["train_runs"]) & set(sp["test_runs"]):
        raise SmokeFailure("train and test runs overlap")

    print("\n" + "=" * 62)
    print(f"SMOKE TEST PASSED in {total:.1f}s "
          f"(budget {args.budget_seconds:.0f}s)")
    print(f"  ensemble: {integrity['rf_class']} + {integrity['xgb_class']}, "
          f"max prob diff {integrity['max_abs_probability_difference']:.4f}")
    print(f"  split: train={sp['train_runs']} val={sp['val_runs']} test={sp['test_runs']}")
    print(f"  routing decisions: {diag['decisions']}")
    print(f"  FRLFP fallback rate: {diag['frlfp']['fallback_rate']:.1%}")
    print("=" * 62)
    print("These numbers are a liveness signal, not results. Run the full")
    print("pipeline without --smoke for anything you intend to quote.")

    if total > args.budget_seconds:
        print(f"\nNOTE: {total:.1f}s exceeded the {args.budget_seconds:.0f}s budget.")


if __name__ == "__main__":
    main()
