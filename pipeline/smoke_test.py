"""End to end smoke test.

Runs every pipeline stage on a tiny configuration and asserts each one produced
what the next depends on. It proves the chain executes and the contracts between
stages hold; it does not produce meaningful numbers.

Stage output is streamed live rather than captured, so a slow or stuck stage is
visible while it happens instead of after a timeout.

    python pipeline/smoke_test.py
    python pipeline/smoke_test.py --stages 4 5      # run a subset
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
SMOKE = CFG.smoke

PROC = ROOT / "data" / "processed"
MODELS = ROOT / "results" / "models"
RESULTS = ROOT / "results"

DEFAULT_STAGE_TIMEOUT = int(os.environ.get("MANET_SMOKE_STAGE_TIMEOUT", "600"))
DEFAULT_BUDGET_SECONDS = 60.0


class SmokeFailure(RuntimeError):
    """A stage failed, timed out, or did not produce its expected output."""


def run_stage(name: str, argv: list[str], expect: list[Path], timeout: int) -> float:
    """Run one stage as a subprocess, streaming its output."""
    print(f"\n=== {name} ===", flush=True)
    started = time.perf_counter()
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")

    process = subprocess.Popen(
        [sys.executable, "-u", *argv],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    try:
        for line in process.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            tail.append(line)
            del tail[:-40]
            if line.startswith("[") or line.startswith("    "):
                print("   " + line, flush=True)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        raise SmokeFailure(
            f"{name} exceeded {timeout}s. Last output:\n" + "\n".join(tail[-15:])
        ) from None

    if process.returncode != 0:
        raise SmokeFailure(
            f"{name} exited with code {process.returncode}. Last output:\n"
            + "\n".join(tail[-25:])
        )

    missing = [str(p) for p in expect if not Path(p).exists()]
    if missing:
        raise SmokeFailure(f"{name} did not produce: {missing}")

    elapsed = time.perf_counter() - started
    print(f"   ok in {elapsed:.1f}s", flush=True)
    return elapsed


def check_contracts() -> tuple[dict, dict]:
    """Assertions a stage returning zero could still violate."""
    with open(RESULTS / "predictor_metrics.json", encoding="utf-8") as handle:
        predictor = json.load(handle)
    integrity = predictor["integrity"]
    if integrity["rf_class"] == integrity["xgb_class"]:
        raise SmokeFailure("ensemble members are the same model class")
    if integrity["max_abs_probability_difference"] <= 0:
        raise SmokeFailure("ensemble members are indistinguishable; the blend is a no op")
    if predictor["ensemble_weights"]["source"] == "test_auc":
        raise SmokeFailure("blend weights were selected on the test set")

    with open(RESULTS / "comparison_diagnostics.json", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    if diagnostics["decisions"] < 1:
        raise SmokeFailure("comparison recorded no routing decisions")
    split = diagnostics["split"]
    if set(split["train_runs"]) & set(split["test_runs"]):
        raise SmokeFailure("train and test runs overlap")
    return integrity, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole pipeline in miniature.")
    parser.add_argument("--max-runs", type=int, default=int(SMOKE["max_runs"]))
    parser.add_argument("--max-rows-per-run", type=int, default=int(SMOKE["max_rows_per_run"]))
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--stage-timeout", type=int, default=DEFAULT_STAGE_TIMEOUT)
    parser.add_argument("--stages", type=int, nargs="*", default=None,
                        help="stage numbers to run, e.g. --stages 4 5. Default: all.")
    args = parser.parse_args()

    split_args = [
        "--test-run-count", str(SMOKE["test_run_count"]),
        "--val-run-count", str(SMOKE["val_run_count"]),
    ]

    stages = [
        ("generate_data",
         ["pipeline/generate_data.py",
          "--max-runs", str(args.max_runs),
          "--max-rows-per-run", str(args.max_rows_per_run)],
         [PROC / "paper_raw_dataset.csv", PROC / "dataset_manifest.json"]),
        ("engineer_features",
         ["pipeline/engineer_features.py", *split_args],
         [PROC / "paper_featured_dataset.csv", MODELS / "feature_norm_stats.json"]),
        ("validate_dataset",
         ["pipeline/validate_dataset.py", "--smoke", "--allow-run-level-traffic", *split_args],
         [RESULTS / "data_quality_report.json"]),
        ("train_predictor (ours)",
         ["pipeline/train_predictor.py", "--smoke", *split_args],
         [MODELS / "random_forest.pkl", MODELS / "xgboost_model.pkl", MODELS / "scaler.pkl",
          MODELS / "ensemble_weights.pkl", MODELS / "predictor_schema.json",
          RESULTS / "predictor_metrics.json"]),
        ("train_models (paper baseline)",
         ["pipeline/train_models.py", "--smoke", "--retrain", *split_args],
         [MODELS / "sfrnnr_paper.keras", MODELS / "sfrnnr_meta.json",
          PROC / "paper_lfp_dataset.csv"]),
        ("compare_methods",
         ["pipeline/compare_methods.py", "--smoke", *split_args],
         [RESULTS / "comparison_metrics.csv", RESULTS / "path_survival.csv",
          RESULTS / "routing_decisions.csv", RESULTS / "comparison_summary.md",
          RESULTS / "comparison_diagnostics.json"]),
    ]

    selected = args.stages or list(range(1, len(stages) + 1))
    total = 0.0
    for number, (name, argv, expect) in enumerate(stages, start=1):
        if number not in selected:
            continue
        total += run_stage(f"{number}/{len(stages)} {name}", argv, expect, args.stage_timeout)

    integrity, diagnostics = check_contracts()

    print("\n" + "=" * 62)
    print(f"SMOKE TEST PASSED in {total:.1f}s (budget {args.budget_seconds:.0f}s)")
    print(f"  ensemble: {integrity['rf_class']} + {integrity['xgb_class']}, "
          f"max prob diff {integrity['max_abs_probability_difference']:.4f}")
    split = diagnostics["split"]
    print(f"  split: train={split['train_runs']} val={split['val_runs']} test={split['test_runs']}")
    print(f"  routing decisions: {diagnostics['decisions']}")
    print(f"  FRLFP fallback rate: {diagnostics['frlfp']['fallback_rate']:.1%}")
    print("=" * 62)
    print("These numbers are a liveness signal, not results. Run the full")
    print("pipeline without --smoke for anything you intend to quote.")

    if total > args.budget_seconds:
        print(f"\nNOTE: {total:.1f}s exceeded the {args.budget_seconds:.0f}s budget.")


if __name__ == "__main__":
    main()
