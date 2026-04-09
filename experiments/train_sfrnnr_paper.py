"""
Train the paper SFRNNR baseline (fuzzification + fuzzy RNN + consequent + threshold head).

Run from repository root:
  python experiments/train_sfrnnr_paper.py --dataset dataset/paper/processed/paper_featured_dataset.csv
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
sys.path.insert(0, str(ROOT / "baseline_paper"))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

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
        if pad > 0:
            last = X[-1:]
            Xp = np.vstack([X, np.repeat(last, pad, axis=0)])
            yp = np.concatenate([y, np.zeros(pad, dtype=np.float32)])
            trp = np.concatenate([thr, np.repeat(thr[-1], pad).astype(np.float32)])
            wt = np.concatenate([np.ones(t, dtype=np.float32), np.zeros(pad, dtype=np.float32)])
        else:
            Xp = X[:max_len]
            yp = y[:max_len]
            trp = thr[:max_len]
            wt = np.ones(Xp.shape[0], dtype=np.float32)
        Xs.append(Xp)
        ys.append(yp)
        thrs.append(trp)
        wts.append(wt)
    return (
        np.stack(Xs, axis=0),
        np.stack(ys, axis=0)[..., np.newaxis],
        np.stack(thrs, axis=0)[..., np.newaxis],
        np.stack(wts, axis=0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="dataset/paper/processed/paper_featured_dataset.csv",
        help="Featured dataset path (repo root relative).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Fast default: 2. Use more for quality.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Large batch = fewer steps per epoch (faster wall clock).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.05,
        help="Validation fraction of training sequences (small = faster).",
    )
    parser.add_argument(
        "--max-train-sequences",
        type=int,
        default=600,
        help="Cap training sequences after excluding test runs (0 = use all). Speeds up quick runs.",
    )
    parser.add_argument("--gru-units", type=int, default=8)
    parser.add_argument("--rule-units", type=int, default=4)
    parser.add_argument("--thr-hidden", type=int, default=4)
    parser.add_argument("--n-mfs", type=int, default=N_MFS, help="Membership functions per factor.")
    args = parser.parse_args()

    os.chdir(ROOT)
    path = ROOT / args.dataset
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = drop_label_aux_columns(add_link_failure_labels(df))

    n = {c: _minmax_global(df[c]) for c in FACTOR_COLS}
    df["thr_teacher"] = AdaptiveThresholdModel.predict_threshold_batch(
        n["LQ_mean"].to_numpy(dtype=np.float32),
        n["RSSI"].to_numpy(dtype=np.float32),
        n["LS"].to_numpy(dtype=np.float32),
        n["LET"].to_numpy(dtype=np.float32),
        n["LL_d"].to_numpy(dtype=np.float32),
        n["ND"].to_numpy(dtype=np.float32),
    )
    for c in FACTOR_COLS:
        df[c] = n[c]

    run_ids = sorted(df["run_id"].dropna().unique().astype(int).tolist())
    rng = random.Random(args.seed)
    test_runs = set(rng.sample(run_ids, k=min(6, len(run_ids))))

    train_items: list = []
    test_items: list = []
    for (rid, nid), g in df.groupby(["run_id", "node_id"], sort=False):
        g = g.sort_values("time")
        X = g[FACTOR_COLS].values.astype(np.float32)
        y = g["link_failure"].values.astype(np.float32)
        thr = g["thr_teacher"].values.astype(np.float32)
        rid = int(rid)
        tup = (X, y, thr)
        if rid in test_runs:
            test_items.append(tup)
        else:
            train_items.append(tup)

    if not train_items:
        raise RuntimeError("No training sequences: increase runs or shrink test set.")

    if args.max_train_sequences and len(train_items) > args.max_train_sequences:
        train_items = rng.sample(train_items, k=args.max_train_sequences)

    max_len = max(len(x[0]) for x in train_items)
    if test_items:
        max_len = max(max_len, max(len(x[0]) for x in test_items))

    rng_np = np.random.RandomState(args.seed)
    n_train = len(train_items)
    perm = rng_np.permutation(n_train)
    n_val = max(1, int(n_train * args.val_frac))
    val_ix = set(perm[:n_val].tolist())
    train_only = [train_items[i] for i in range(n_train) if i not in val_ix]
    val_only = [train_items[i] for i in range(n_train) if i in val_ix]

    X_tv, y_tv, thr_tv, w_tv = pad_sequences(train_only, max_len)
    X_va, y_va, thr_va, w_va = pad_sequences(val_only, max_len)

    model = build_sfrnnr_model(
        seq_len=max_len,
        n_factors=N_FACTORS,
        n_mfs=args.n_mfs,
        gru_units=args.gru_units,
        rule_units=args.rule_units,
        thr_hidden=args.thr_hidden,
        dropout=0.0,
    )

    sw_tr = {"lfp": w_tv, "lfp_threshold": w_tv}
    sw_va = {"lfp": w_va, "lfp_threshold": w_va}
    model.fit(
        X_tv,
        {"lfp": y_tv, "lfp_threshold": thr_tv},
        sample_weight=sw_tr,
        validation_data=(X_va, {"lfp": y_va, "lfp_threshold": thr_va}, sw_va),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "sfrnnr_paper.keras"
    model.save(model_path)

    meta = {
        "factor_cols": FACTOR_COLS,
        "seq_len": int(max_len),
        "test_run_ids": sorted(test_runs),
        "seed": args.seed,
        "n_train_seq": int(X_tv.shape[0]),
        "n_val_seq": int(X_va.shape[0]),
        "n_test_seq": len(test_items),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_train_sequences": args.max_train_sequences,
        "gru_units": args.gru_units,
        "rule_units": args.rule_units,
        "n_mfs": args.n_mfs,
    }
    with open(out_dir / "sfrnnr_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if test_items:
        X_te, y_te, thr_te, w_te = pad_sequences(test_items, max_len)
        pred = model.predict(X_te, batch_size=args.batch_size, verbose=0)
        p = pred["lfp"].reshape(-1)
        y_flat = y_te.reshape(-1)
        w_flat = w_te.reshape(-1)
        m = w_flat > 0.5
        if m.sum() > 0:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_flat[m], p[m])
            print(f"Held-out run ROC-AUC (SFRNNR lfp): {auc:.4f}")

    print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
