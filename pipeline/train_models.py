"""Stage 5b: attach the paper baseline's outputs to the shared dataset.

Despite the historical name, this script does not train our predictor. Ours is
trained by pipeline/train_predictor.py. This one trains the SFRNNR if needed,
runs it over the featured dataset, and writes paper_lfp_dataset.csv with the
per node link failure probability and adaptive threshold attached.

Keeping both methods' inputs in one file is what makes the comparison in
pipeline/compare_methods.py a genuinely shared evaluation rather than two
pipelines that merely resemble each other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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


def ensure_sfrnnr_trained(smoke: bool, extra_args: list[str]) -> None:
    """Train the baseline if no model is on disk.

    Called in process rather than as a subprocess so TensorFlow is imported once
    per pipeline run instead of twice, which is most of the smoke test budget.
    """
    if MODEL_FILE.is_file() and META_FILE.is_file():
        return
    sys.path.insert(0, str(ROOT / "pipeline"))
    import train_sfrnnr_paper

    argv = ["train_sfrnnr_paper.py"] + (["--smoke"] if smoke else []) + list(extra_args)
    saved = sys.argv
    try:
        sys.argv = argv
        train_sfrnnr_paper.main()
    finally:
        sys.argv = saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the SFRNNR baseline over the shared dataset and attach its outputs."
    )
    parser.add_argument("--input", default=str(INPUT_FILE))
    parser.add_argument("--output", default=str(OUTPUT_FILE))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--retrain", action="store_true", help="train the SFRNNR even if a model exists")
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError(
            f"Missing {args.input}. Run pipeline/engineer_features.py first."
        )

    extra: list[str] = []
    if args.test_run_count is not None:
        extra += ["--test-run-count", str(args.test_run_count)]
    if args.val_run_count is not None:
        extra += ["--val-run-count", str(args.val_run_count)]

    if args.retrain:
        for p in (MODEL_FILE, META_FILE):
            if p.exists():
                p.unlink()
    ensure_sfrnnr_trained(args.smoke, extra)

    df = pd.read_csv(args.input).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = drop_label_aux_columns(add_link_failure_labels(df))
    df = apply_sfrnnr(df, repo_root=ROOT, model_path=MODEL_FILE, meta_path=META_FILE)
    df["paper_predicted_failure"] = (df["lfp"] > df["lfp_threshold"]).astype(int)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    flagged = float(df["paper_predicted_failure"].mean())
    print(f"[train_models] wrote {len(df)} rows to {args.output}")
    print(f"[train_models] mean lfp {df['lfp'].mean():.4f}, "
          f"mean threshold {df['lfp_threshold'].mean():.4f}")
    print(f"[train_models] SFRNNR flags {flagged:.1%} of node snapshots as failing "
          f"(true label rate {df['link_failure'].mean():.1%})")
    if flagged > 0.75:
        print("[train_models] WARNING: the baseline flags most of the network. Its filtered "
              "routing graph will disconnect and FRLFP will fall back to shortest path on "
              "almost every decision, which makes it indistinguishable from hop count "
              "routing. Do not quote a comparison against this baseline until the fallback "
              "rate reported by compare_methods is low.")
    elif flagged < 0.02:
        print("[train_models] WARNING: the baseline flags almost nothing, so its filtered "
              "graph equals the full graph and FRLFP again reduces to hop count routing. "
              "The SFRNNR is probably undertrained; run without --smoke.")


if __name__ == "__main__":
    main()
