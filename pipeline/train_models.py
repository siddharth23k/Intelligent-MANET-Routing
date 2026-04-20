"""Train SFRNNR and apply inference."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "methods/baseline"))

from label_utils import add_link_failure_labels, drop_label_aux_columns  # noqa: E402
from sfrnnr_infer import apply_sfrnnr  # noqa: E402

INPUT_FILE = "data/processed/paper_featured_dataset.csv"
OUTPUT_FILE = "data/processed/paper_lfp_dataset.csv"
MODEL_FILE = ROOT / "results/models" / "sfrnnr_paper.keras"


def ensure_sfrnnr_trained():
    if MODEL_FILE.is_file():
        return
        cmd = [sys.executable, str(ROOT / "pipeline" / "train_sfrnnr_paper.py")]
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main():
    os.chdir(ROOT)
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run paper_feature_engineering.py first.")

    df = pd.read_csv(INPUT_FILE).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = add_link_failure_labels(df)
    df = drop_label_aux_columns(df)

    ensure_sfrnnr_trained()
    df = apply_sfrnnr(df, repo_root=ROOT, 
                   model_path=ROOT / "results/models" / "sfrnnr_paper.keras",
                   meta_path=ROOT / "results/models" / "sfrnnr_meta.json")
    df["paper_predicted_failure"] = (df["lfp"] > df["lfp_threshold"]).astype(int)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    

if __name__ == "__main__":
    main()
