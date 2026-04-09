"""
Paper baseline: supervised labels + trained SFRNNR (fuzzification, fuzzy RNN, consequent, threshold).

If no trained model is present, trains SFRNNR automatically (can take several minutes).

Run from repository root:
  python scripts/paper_compute_lfp.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "baseline_paper"))

from label_utils import add_link_failure_labels, drop_label_aux_columns  # noqa: E402
from sfrnnr_infer import apply_sfrnnr  # noqa: E402

INPUT_FILE = "dataset/paper/processed/paper_featured_dataset.csv"
OUTPUT_FILE = "dataset/paper/processed/paper_lfp_dataset.csv"
MODEL_FILE = ROOT / "models" / "sfrnnr_paper.keras"


def ensure_sfrnnr_trained():
    if MODEL_FILE.is_file():
        return
    print("No SFRNNR checkpoint found; training paper baseline model...")
    cmd = [sys.executable, str(ROOT / "experiments" / "train_sfrnnr_paper.py")]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    os.chdir(ROOT)
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run paper_feature_engineering.py first.")

    df = pd.read_csv(INPUT_FILE).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = add_link_failure_labels(df)
    df = drop_label_aux_columns(df)

    ensure_sfrnnr_trained()
    df = apply_sfrnnr(df, repo_root=ROOT)
    df["paper_predicted_failure"] = (df["lfp"] > df["lfp_threshold"]).astype(int)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} ({len(df)} rows)")


if __name__ == "__main__":
    main()
