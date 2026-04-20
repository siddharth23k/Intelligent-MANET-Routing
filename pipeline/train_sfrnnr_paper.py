"""
Train the paper SFRNNR baseline (fuzzification + fuzzy RNN + consequent + threshold head).

Run from repository root:
  python pipeline/train_sfrnnr_paper.py --dataset data/processed/paper_featured_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "methods/baseline"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras  # noqa: E402

from label_utils import add_link_failure_labels, drop_label_aux_columns  # noqa: E402
from sfrnnr_model import (  # noqa: E402
    FACTOR_COLS,
    N_FACTORS,
    N_MFS,
    build_sfrnnr_model,
)
from threshold_model import AdaptiveThresholdModel  # noqa: E402


def _minmax_global(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if np.isclose(hi - lo, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def pad_sequences(items: list[tuple], max_len: int):
    """items: list of (X [T, F], y [T], thr [T])"""
    Xs, ys, thrs, wts = [], [], [], []
    for X, y, thr in items:
        t = X.shape[0]
        pad = max_len - t
        X_pad = np.pad(X, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
        y_pad = np.pad(y, (0, pad), mode="constant", constant_values=0.0)
        thr_pad = np.pad(thr, (0, pad), mode="constant", constant_values=0.0)
        wts_pad = np.concatenate([np.ones(t), np.zeros(pad)])
        Xs.append(X_pad)
        ys.append(y_pad)
        thrs.append(thr_pad)
        wts.append(wts_pad)
    return np.stack(Xs), np.stack(ys), np.stack(thrs), np.stack(wts)


def main():
    parser = argparse.ArgumentParser(description="Train SFRNNR paper baseline")
    parser.add_argument("--dataset", default="data/processed/paper_featured_dataset.csv")
    parser.add_argument("--gru-units", type=int, default=16)
    parser.add_argument("--rule-units", type=int, default=8)
    parser.add_argument("--n-mfs", type=int, default=N_MFS)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train-sequences", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print(f"Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = add_link_failure_labels(df)
    df = drop_label_aux_columns(df)

    # Normalize factor columns globally
    for col in FACTOR_COLS:
        df[col] = _minmax_global(df[col])

    # Prepare sequences by (run_id, node_id)
    sequences_with_runs = []
    for (run_id, node_id), group in df.groupby(["run_id", "node_id"]):
        if len(group) < 3:  # Skip very short sequences
            continue
        X = group[FACTOR_COLS].values.astype(np.float32)
        y = group["link_failure"].values.astype(np.float32)
        
        # Teacher threshold signal - use static method
        thr = np.array([AdaptiveThresholdModel.predict_threshold(row.to_dict()) 
                       for _, row in group.iterrows()]).astype(np.float32)
        
        sequences_with_runs.append((run_id, X, y, thr))

    # Split by run_id
    run_ids = sorted(df["run_id"].unique())
    train_runs = set(run_ids[:20])  # First 20 runs for training
    test_runs = set(run_ids[20:])   # Remaining runs for testing

    train_sequences = [(X, y, thr) for run_id, X, y, thr in sequences_with_runs if run_id in train_runs]
    test_sequences = [(X, y, thr) for run_id, X, y, thr in sequences_with_runs if run_id in test_runs]

    # Limit training sequences if requested
    if args.max_train_sequences > 0:
        train_sequences = train_sequences[:args.max_train_sequences]

    print(f"Train sequences: {len(train_sequences)}, Test sequences: {len(test_sequences)}")

    # Pad sequences
    max_len = max(max(len(seq[0]) for seq in train_sequences), 
                  max(len(seq[0]) for seq in test_sequences))
    
    X_train, y_train, thr_train, wts_train = pad_sequences(train_sequences, max_len)
    X_test, y_test, thr_test, wts_test = pad_sequences(test_sequences, max_len)

    # Build model
    model = build_sfrnnr_model(
        gru_units=args.gru_units,
        rule_units=args.rule_units,
        n_mfs=args.n_mfs,
        seq_len=max_len,
        n_factors=N_FACTORS,
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "lfp": "binary_crossentropy",
            "lfp_threshold": "mse",
        },
        loss_weights={"lfp": 1.0, "lfp_threshold": 0.1},
        metrics={"lfp": "auc"},
    )

    print(f"Training SFRNNR for {args.epochs} epochs...")
    history = model.fit(
        X_train, {"lfp": y_train, "lfp_threshold": thr_train},
        sample_weight={"lfp": wts_train, "lfp_threshold": wts_train},
        validation_data=(X_test, {"lfp": y_test, "lfp_threshold": thr_test}),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
    )

    # Save model and metadata
    model_path = ROOT / "results/models" / "sfrnnr_paper.keras"
    meta_path = ROOT / "results/models" / "sfrnnr_meta.json"
    
    os.makedirs(model_path.parent, exist_ok=True)
    model.save(model_path)
    
    metadata = {
        "gru_units": args.gru_units,
        "rule_units": args.rule_units,
        "n_mfs": args.n_mfs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": max_len,
        "train_sequences": len(train_sequences),
        "test_sequences": len(test_sequences),
        "train_runs": train_runs,
        "test_runs": test_runs,
        "final_train_auc": float(history.history["lfp_output_auc"][-1]),
        "final_val_auc": float(history.history["val_lfp_output_auc"][-1]),
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {meta_path}")
    print(f"Final train AUC: {metadata['final_train_auc']:.4f}")
    print(f"Final val AUC: {metadata['final_val_auc']:.4f}")


if __name__ == "__main__":
    main()
