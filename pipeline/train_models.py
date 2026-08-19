"""Stage 5b: attach the paper baseline's outputs to the shared dataset.

Despite the historical name this does not train our predictor; that is
pipeline/train_predictor.py. This trains the SFRNNR if no model exists, runs it
over the featured dataset, and writes paper_lfp_dataset.csv with the per node
link failure probability and adaptive threshold attached.

Keeping both methods' inputs in one file is what makes stage 6 a genuinely
shared evaluation rather than two pipelines that merely resemble each other.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from config_loader import get_config  # noqa: E402
from label_utils import add_link_failure_labels, drop_label_aux_columns  # noqa: E402
from sfrnnr_infer import apply_sfrnnr  # noqa: E402

CFG = get_config()
INPUT_FILE = ROOT / "data" / "processed" / "paper_featured_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "paper_lfp_dataset.csv"
MODEL_FILE = ROOT / "results" / "models" / "sfrnnr_paper.keras"
META_FILE = ROOT / "results" / "models" / "sfrnnr_meta.json"

HIGH_FLAG_RATE = 0.75
LOW_FLAG_RATE = 0.02


def ensure_sfrnnr_trained(smoke: bool, extra_args: list[str]) -> None:
    """Train the baseline if no model is on disk.

    Called in process rather than as a subprocess, so TensorFlow is imported
    once per pipeline run instead of twice.
    """
    if MODEL_FILE.is_file() and META_FILE.is_file():
        return
    sys.path.insert(0, str(ROOT / "pipeline"))
    import train_sfrnnr_paper

    argv = ["train_sfrnnr_paper.py"] + (["--smoke"] if smoke else []) + list(extra_args)
    saved_argv = sys.argv
    try:
        sys.argv = argv
        train_sfrnnr_paper.main()
    finally:
        sys.argv = saved_argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SFRNNR baseline over the shared dataset and attach its outputs."
    )
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="train even if a model exists")
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(f"Missing {args.input}. Run pipeline/engineer_features.py first.")

    extra: list[str] = []
    if args.test_run_count is not None:
        extra += ["--test-run-count", str(args.test_run_count)]
    if args.val_run_count is not None:
        extra += ["--val-run-count", str(args.val_run_count)]

    if args.retrain:
        for path in (MODEL_FILE, META_FILE):
            if path.exists():
                path.unlink()
    ensure_sfrnnr_trained(args.smoke, extra)

    print("[train_models] running SFRNNR inference over the dataset", flush=True)
    started = time.perf_counter()
    df = pd.read_csv(args.input).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = drop_label_aux_columns(add_link_failure_labels(df))
    df = apply_sfrnnr(df, repo_root=ROOT, model_path=MODEL_FILE, meta_path=META_FILE, verbose=True)
    df["paper_predicted_failure"] = (df["lfp"] > df["lfp_threshold"]).astype(int)
    print(f"[train_models] inference done in {time.perf_counter() - started:.1f}s", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    flagged = float(df["paper_predicted_failure"].mean())
    print(f"[train_models] wrote {len(df)} rows to {args.output}")
    print(f"[train_models] mean lfp {df['lfp'].mean():.4f}, "
          f"mean threshold {df['lfp_threshold'].mean():.4f}")
    print(f"[train_models] SFRNNR flags {flagged:.1%} of node snapshots as failing "
          f"(true label rate {df['link_failure'].mean():.1%})")

    if flagged > HIGH_FLAG_RATE:
        print("[train_models] WARNING: the baseline flags most of the network, so its "
              "filtered graph disconnects and FRLFP falls back to hop count routing on "
              "almost every decision. Do not quote a comparison against it until the "
              "fallback rate from compare_methods is low.")
    elif flagged < LOW_FLAG_RATE:
        print("[train_models] WARNING: the baseline flags almost nothing, so its filtered "
              "graph equals the full graph and FRLFP again reduces to hop count routing. "
              "The SFRNNR is probably undertrained; run without --smoke.")


if __name__ == "__main__":
    main()
