"""Stage 5a: train the SFRNNR paper baseline.

The point of this script is a comparison that is worth publishing, which means
the baseline has to be trained honestly. The previous defaults were one epoch
over two hundred of three thousand sequences, roughly seven percent of the data,
and the resulting model flagged over ninety percent of nodes as failing. Beating
that says nothing.

What changed:
  - real epoch budget with early stopping on a validation split
  - the validation split comes from the shared run level splitter, so the
    baseline is never validated or selected on the runs it will be scored on
  - factor normalisation statistics fitted on training runs only and persisted
    into the meta file, so inference reuses exactly the same scaling
  - --smoke keeps the whole thing under a few seconds for CI
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from config_loader import get_config  # noqa: E402
from label_utils import add_link_failure_labels, drop_label_aux_columns  # noqa: E402
from normalization import MinMaxStats  # noqa: E402
from schema import FACTOR_COLS  # noqa: E402
from sfrnnr_model import N_FACTORS, N_MFS, build_sfrnnr_model  # noqa: E402
from splits import make_run_split  # noqa: E402
from threshold_model import AdaptiveThresholdModel  # noqa: E402

CFG = get_config()
MODEL_PATH = ROOT / "results" / "models" / "sfrnnr_paper.keras"
META_PATH = ROOT / "results" / "models" / "sfrnnr_meta.json"
DEFAULT_DATASET = ROOT / "data" / "processed" / "paper_featured_dataset.csv"


def build_sequences(df: pd.DataFrame, seq_len: int):
    """One fixed length window per (run_id, node_id).

    Tracks shorter than seq_len are dropped rather than zero padded. Padding
    plus per timestep sample weights was the old approach and it made the loss
    masking fragile across Keras versions; since seq_len is clamped to the
    shortest track in the caller, nothing is actually lost.
    """
    Xs, ys, thrs, runs = [], [], [], []
    df = df.assign(_thr=AdaptiveThresholdModel.predict_threshold_frame(df))

    dropped = 0
    for (run_id, node_id), g in df.groupby(["run_id", "node_id"], sort=False):
        if len(g) < seq_len:
            dropped += 1
            continue
        g = g.sort_values("time")
        Xs.append(g[FACTOR_COLS].to_numpy(dtype=np.float32)[-seq_len:])
        ys.append(g["link_failure"].to_numpy(dtype=np.float32)[-seq_len:].reshape(-1, 1))
        thrs.append(g["_thr"].to_numpy(dtype=np.float32)[-seq_len:].reshape(-1, 1))
        runs.append(int(run_id))

    if not Xs:
        raise ValueError(
            f"no (run, node) track reaches seq_len={seq_len}; lower --seq-len"
        )
    if dropped:
        print(f"[train_sfrnnr] dropped {dropped} track(s) shorter than seq_len={seq_len}")
    return (
        np.stack(Xs),
        np.stack(ys),
        np.stack(thrs),
        np.asarray(runs, dtype=int),
    )


def main() -> None:
    tr_cfg = CFG.training
    sm = CFG.smoke

    parser = argparse.ArgumentParser(description="Train the SFRNNR paper baseline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--gru-units", type=int, default=int(tr_cfg["sfrnnr_gru_units"]))
    parser.add_argument("--rule-units", type=int, default=int(tr_cfg["sfrnnr_rule_units"]))
    parser.add_argument("--n-mfs", type=int, default=int(tr_cfg["sfrnnr_n_mfs"]))
    parser.add_argument("--batch-size", type=int, default=int(tr_cfg["sfrnnr_batch_size"]))
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    args = parser.parse_args()

    smoke = args.smoke
    epochs = args.epochs if args.epochs is not None else int(sm["sfrnnr_epochs"] if smoke else tr_cfg["sfrnnr_epochs"])
    max_train_sequences = (
        args.max_train_sequences
        if args.max_train_sequences is not None
        else (int(sm["sfrnnr_max_train_sequences"]) if smoke else 0)
    )
    test_run_count = args.test_run_count if args.test_run_count is not None else int(
        sm["test_run_count"] if smoke else CFG.test_run_count
    )
    val_run_count = args.val_run_count if args.val_run_count is not None else int(
        sm["val_run_count"] if smoke else CFG.val_run_count
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = pd.read_csv(args.dataset).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = drop_label_aux_columns(add_link_failure_labels(df))

    run_ids = sorted(df["run_id"].unique().astype(int).tolist())
    split = make_run_split(run_ids, seed=args.seed, test_run_count=test_run_count, val_run_count=val_run_count)

    # Fit factor normalisation on training runs only, then persist it so
    # inference cannot silently use different statistics.
    stats = MinMaxStats.fit(df[df["run_id"].isin(split.train_runs)], FACTOR_COLS)
    df = stats.transform(df, FACTOR_COLS)

    track_lengths = df.groupby(["run_id", "node_id"]).size()
    timesteps = int(track_lengths.min())
    seq_len = args.seq_len or (int(sm["sfrnnr_seq_len"]) if smoke else timesteps)
    seq_len = max(3, min(seq_len, timesteps))

    X, y, thr, runs = build_sequences(df, seq_len)

    train_mask = np.isin(runs, split.train_runs)
    val_mask = np.isin(runs, split.val_runs) if split.val_runs else np.zeros_like(train_mask)
    if not val_mask.any():
        # No dedicated validation runs (smoke). Hold out a slice of the training
        # sequences instead. The test runs are still never touched.
        idx = np.where(train_mask)[0]
        cut = max(1, int(0.2 * len(idx)))
        val_idx = idx[:cut]
        val_mask = np.zeros_like(train_mask)
        val_mask[val_idx] = True
        train_mask[val_idx] = False

    tr_idx = np.where(train_mask)[0]
    if max_train_sequences > 0:
        tr_idx = tr_idx[:max_train_sequences]
    va_idx = np.where(val_mask)[0]

    model = build_sfrnnr_model(
        seq_len=seq_len,
        n_factors=N_FACTORS,
        n_mfs=args.n_mfs or N_MFS,
        gru_units=args.gru_units,
        rule_units=args.rule_units,
    )

    callbacks = []
    if not smoke:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_lfp_auc",
                mode="max",
                patience=int(tr_cfg["sfrnnr_patience"]),
                restore_best_weights=True,
                verbose=1,
            )
        )
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
            )
        )

    history = model.fit(
        X[tr_idx],
        {"lfp": y[tr_idx], "lfp_threshold": thr[tr_idx]},
        validation_data=(
            X[va_idx],
            {"lfp": y[va_idx], "lfp_threshold": thr[va_idx]},
        ),
        epochs=epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2 if not smoke else 0,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    def _last(key, default=float("nan")):
        v = history.history.get(key)
        return float(v[-1]) if v else default

    meta = {
        "smoke": bool(smoke),
        "seed": int(args.seed),
        "gru_units": args.gru_units,
        "rule_units": args.rule_units,
        "n_mfs": args.n_mfs,
        "epochs_requested": int(epochs),
        "epochs_run": int(len(history.history.get("loss", []))),
        "batch_size": int(args.batch_size),
        "seq_len": int(seq_len),
        "max_len": int(seq_len),
        "train_sequences": int(len(tr_idx)),
        "val_sequences": int(len(va_idx)),
        "total_sequences": int(len(runs)),
        "sequence_coverage": float(len(tr_idx) / max(1, int(train_mask.sum() + len(tr_idx)))),
        "split": split.as_dict(),
        "factor_cols": FACTOR_COLS,
        "factor_norm_stats": stats.stats,
        "final_train_auc": _last("lfp_auc"),
        "final_val_auc": _last("val_lfp_auc"),
        "early_stopping": not smoke,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[train_sfrnnr] seq_len={seq_len} train_seq={len(tr_idx)} val_seq={len(va_idx)} "
          f"of {len(runs)} total")
    print(f"[train_sfrnnr] epochs run {meta['epochs_run']}/{epochs} "
          f"(early stopping {'on' if meta['early_stopping'] else 'off'})")
    print(f"[train_sfrnnr] train auc {meta['final_train_auc']:.4f} "
          f"val auc {meta['final_val_auc']:.4f}")
    print(f"[train_sfrnnr] split -> {split.describe()}")
    print(f"[train_sfrnnr] saved {MODEL_PATH.name} and {META_PATH.name}")


if __name__ == "__main__":
    main()
