"""Stage 5a: train the SFRNNR paper baseline.

The comparison is only worth reporting if the baseline is trained honestly, so
the defaults are a real epoch budget with early stopping on a validation split
drawn from the shared run level splitter. --smoke keeps it under a few seconds.

Progress is printed at each phase, because TensorFlow start up dominates the
wall clock and a silent script looks like a hang.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_t0 = time.perf_counter()
print("[train_sfrnnr] importing tensorflow", flush=True)
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

print(f"[train_sfrnnr] tensorflow ready in {time.perf_counter() - _t0:.1f}s", flush=True)

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
    plus per timestep sample weights made loss masking fragile across Keras
    versions, and seq_len is clamped to the shortest track by the caller.
    """
    windows, labels, thresholds, runs = [], [], [], []
    df = df.assign(_thr=AdaptiveThresholdModel.predict_threshold_frame(df))

    dropped = 0
    for (run_id, _node_id), group in df.groupby(["run_id", "node_id"], sort=False):
        if len(group) < seq_len:
            dropped += 1
            continue
        group = group.sort_values("time")
        windows.append(group[FACTOR_COLS].to_numpy(dtype=np.float32)[-seq_len:])
        labels.append(group["link_failure"].to_numpy(dtype=np.float32)[-seq_len:].reshape(-1, 1))
        thresholds.append(group["_thr"].to_numpy(dtype=np.float32)[-seq_len:].reshape(-1, 1))
        runs.append(int(run_id))

    if not windows:
        raise ValueError(f"no track reaches seq_len={seq_len}; lower --seq-len")
    if dropped:
        print(f"[train_sfrnnr] dropped {dropped} track(s) shorter than {seq_len}", flush=True)
    return np.stack(windows), np.stack(labels), np.stack(thresholds), np.asarray(runs, dtype=int)


def _history_value(history, *keys, default=float("nan")) -> float:
    for key in keys:
        values = history.history.get(key)
        if values:
            return float(values[-1])
    return default


def main() -> None:
    training, smoke_cfg = CFG.training, CFG.smoke

    parser = argparse.ArgumentParser(description="Train the SFRNNR paper baseline.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--gru-units", type=int, default=int(training["sfrnnr_gru_units"]))
    parser.add_argument("--rule-units", type=int, default=int(training["sfrnnr_rule_units"]))
    parser.add_argument("--n-mfs", type=int, default=int(training["sfrnnr_n_mfs"]))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    args = parser.parse_args()

    smoke = args.smoke
    def pick(explicit, smoke_key, training_key):
        if explicit is not None:
            return int(explicit)
        return int(smoke_cfg[smoke_key] if smoke else training[training_key])

    epochs = pick(args.epochs, "sfrnnr_epochs", "sfrnnr_epochs")
    batch_size = pick(args.batch_size, "sfrnnr_batch_size", "sfrnnr_batch_size")
    max_train_sequences = (
        int(args.max_train_sequences)
        if args.max_train_sequences is not None
        else (int(smoke_cfg["sfrnnr_max_train_sequences"]) if smoke else 0)
    )
    test_run_count = (
        int(args.test_run_count)
        if args.test_run_count is not None
        else int(smoke_cfg["test_run_count"] if smoke else CFG.test_run_count)
    )
    val_run_count = (
        int(args.val_run_count)
        if args.val_run_count is not None
        else int(smoke_cfg["val_run_count"] if smoke else CFG.val_run_count)
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    df = pd.read_csv(args.dataset).sort_values(["run_id", "node_id", "time"]).reset_index(drop=True)
    df = drop_label_aux_columns(add_link_failure_labels(df))

    run_ids = sorted(df["run_id"].unique().astype(int).tolist())
    split = make_run_split(
        run_ids, seed=args.seed, test_run_count=test_run_count, val_run_count=val_run_count
    )

    # Fit factor normalisation on training runs only and persist it, so
    # inference cannot silently use different statistics.
    stats = MinMaxStats.fit(df[df["run_id"].isin(split.train_runs)], FACTOR_COLS)
    df = stats.transform(df, FACTOR_COLS)

    shortest_track = int(df.groupby(["run_id", "node_id"]).size().min())
    seq_len = args.seq_len or (int(smoke_cfg["sfrnnr_seq_len"]) if smoke else shortest_track)
    seq_len = max(3, min(int(seq_len), shortest_track))

    X, y, thresholds, runs = build_sequences(df, seq_len)

    train_mask = np.isin(runs, split.train_runs)
    val_mask = np.isin(runs, split.val_runs) if split.val_runs else np.zeros_like(train_mask)
    if not val_mask.any():
        # No dedicated validation runs. Hold out a slice of the training
        # sequences instead; the test runs are still never touched.
        indices = np.where(train_mask)[0]
        cut = max(1, int(0.2 * len(indices)))
        val_mask = np.zeros_like(train_mask)
        val_mask[indices[:cut]] = True
        train_mask[indices[:cut]] = False

    train_idx = np.where(train_mask)[0]
    if max_train_sequences > 0:
        train_idx = train_idx[:max_train_sequences]
    val_idx = np.where(val_mask)[0]

    print(f"[train_sfrnnr] seq_len={seq_len} train_seq={len(train_idx)} "
          f"val_seq={len(val_idx)} of {len(runs)} total", flush=True)

    model = build_sfrnnr_model(
        seq_len=seq_len,
        n_factors=N_FACTORS,
        n_mfs=args.n_mfs or N_MFS,
        gru_units=args.gru_units,
        rule_units=args.rule_units,
    )

    callbacks = []
    if not smoke:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_lfp_auc",
                mode="max",
                patience=int(training["sfrnnr_patience"]),
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0
            ),
        ]

    print(f"[train_sfrnnr] fitting for up to {epochs} epoch(s)", flush=True)
    fit_started = time.perf_counter()
    history = model.fit(
        X[train_idx],
        {"lfp": y[train_idx], "lfp_threshold": thresholds[train_idx]},
        validation_data=(X[val_idx], {"lfp": y[val_idx], "lfp_threshold": thresholds[val_idx]}),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2 if not smoke else 0,
    )
    print(f"[train_sfrnnr] fit done in {time.perf_counter() - fit_started:.1f}s", flush=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    meta = {
        "smoke": bool(smoke),
        "seed": int(args.seed),
        "gru_units": args.gru_units,
        "rule_units": args.rule_units,
        "n_mfs": args.n_mfs,
        "epochs_requested": int(epochs),
        "epochs_run": int(len(history.history.get("loss", []))),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "train_sequences": int(len(train_idx)),
        "val_sequences": int(len(val_idx)),
        "total_sequences": int(len(runs)),
        "split": split.as_dict(),
        "factor_cols": FACTOR_COLS,
        "factor_norm_stats": stats.stats,
        "final_train_auc": _history_value(history, "lfp_auc", "auc"),
        "final_val_auc": _history_value(history, "val_lfp_auc", "val_auc"),
        "early_stopping": not smoke,
    }
    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"[train_sfrnnr] epochs run {meta['epochs_run']}/{epochs} "
          f"(early stopping {'on' if meta['early_stopping'] else 'off'})", flush=True)
    print(f"[train_sfrnnr] train auc {meta['final_train_auc']:.4f} "
          f"val auc {meta['final_val_auc']:.4f}", flush=True)
    print(f"[train_sfrnnr] split -> {split.describe()}", flush=True)
    print(f"[train_sfrnnr] saved {MODEL_PATH.name} and {META_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
