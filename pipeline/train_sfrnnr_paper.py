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

from sklearn.metrics import roc_auc_score  # noqa: E402

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


AUC_SAMPLE_LIMIT = 512


class TrainingHistory:
    """Minimal stand in for a Keras History object."""

    def __init__(self):
        self.history: dict[str, list] = {}

    def record(self, **values) -> None:
        for key, value in values.items():
            self.history.setdefault(key, []).append(float(value))


def _batched(n: int, batch_size: int, rng: np.random.Generator):
    """Shuffled index batches, all exactly batch_size.

    Every batch having the same shape means graph mode traces the train step
    once instead of once per distinct batch size, which is what made Keras
    retrace on the ragged final batch. Reshuffling each epoch means the dropped
    remainder is a different handful of sequences every time.
    """
    order = rng.permutation(n)
    usable = (n // batch_size) * batch_size
    for start in range(0, usable, batch_size):
        yield order[start : start + batch_size]


def predict_lfp(model, X: np.ndarray, batch_size: int) -> np.ndarray:
    """Forward pass in batches, calling the model directly.

    model.predict builds a tf.data pipeline even for a plain numpy array. That
    adapter is the component that stalls on some TensorFlow builds, and calling
    the model as a function skips it entirely: no data adapter, no predict
    function, just ops.
    """
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    chunks = []
    for start in range(0, len(X), batch_size):
        outputs = model(X[start : start + batch_size], training=False)
        chunks.append(np.asarray(outputs["lfp"])[:, :, 0])
    return np.concatenate(chunks, axis=0)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    flat_true = y_true.reshape(-1)
    flat_score = y_score.reshape(-1)
    if len(np.unique(flat_true)) < 2:
        return float("nan")
    return float(roc_auc_score(flat_true, flat_score))


def train_with_loop(
    model,
    X_train, y_train, thr_train,
    X_val, y_val,
    epochs: int,
    batch_size: int,
    patience: int,
    seed: int,
    verbose: bool = True,
) -> TrainingHistory:
    """Explicit training loop over train_on_batch.

    Written out rather than calling model.fit because fit's data adapter and
    epoch iterator sit between the caller and the first gradient step, and a
    stall in there is invisible. Here every batch is timed and reported, early
    stopping is explicit, and only one tensor shape ever reaches the model.
    """
    rng = np.random.default_rng(seed)
    history = TrainingHistory()
    batches = max(1, len(X_train) // batch_size)

    # Training AUC is a progress signal, not a selection criterion, so it is
    # measured on a fixed subsample. Early stopping uses validation AUC, which
    # is always computed on the full validation set.
    auc_sample = np.sort(
        rng.choice(len(X_train), size=min(len(X_train), AUC_SAMPLE_LIMIT), replace=False)
    )

    if verbose:
        print(f"[train_sfrnnr] starting first batch (shape {X_train[:batch_size].shape})",
              flush=True)

    best_auc, best_weights, since_best = -np.inf, None, 0
    for epoch in range(epochs):
        started = time.perf_counter()
        losses = []
        for index, batch in enumerate(_batched(len(X_train), batch_size, rng)):
            outputs = model.train_on_batch(
                X_train[batch],
                {"lfp": y_train[batch], "lfp_threshold": thr_train[batch]},
                return_dict=True,
            )
            losses.append(float(outputs.get("loss", np.nan)))
            if verbose and epoch == 0 and index < 3:
                print(f"[train_sfrnnr]   batch {index + 1}/{batches} "
                      f"loss={losses[-1]:.4f} ({time.perf_counter() - started:.1f}s)", flush=True)

        train_auc = _auc(y_train[auc_sample], predict_lfp(model, X_train[auc_sample], batch_size))
        val_auc = float("nan")
        if len(X_val):
            val_auc = _auc(y_val, predict_lfp(model, X_val, batch_size))

        history.record(loss=np.nanmean(losses), lfp_auc=train_auc, val_lfp_auc=val_auc)
        if verbose:
            print(f"[train_sfrnnr] epoch {epoch + 1}/{epochs} "
                  f"loss={np.nanmean(losses):.4f} auc={train_auc:.4f} "
                  f"val_auc={val_auc:.4f} ({time.perf_counter() - started:.1f}s)", flush=True)

        score = val_auc if not np.isnan(val_auc) else train_auc
        if score > best_auc:
            best_auc, best_weights, since_best = score, model.get_weights(), 0
        else:
            since_best += 1
            if patience > 0 and since_best >= patience:
                if verbose:
                    print(f"[train_sfrnnr] early stopping at epoch {epoch + 1} "
                          f"(best score {best_auc:.4f})", flush=True)
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return history


class EpochProgress(keras.callbacks.Callback):
    """Print one line per epoch with elapsed seconds.

    Keras' own progress bar is suppressed in scripted runs, and a silent fit is
    indistinguishable from a stalled one.
    """

    def on_train_begin(self, logs=None):
        self._started = time.perf_counter()
        print("[train_sfrnnr] epoch 0 starting", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        auc = logs.get("lfp_auc", logs.get("auc", float("nan")))
        print(f"[train_sfrnnr] epoch {epoch + 1} done in "
              f"{time.perf_counter() - self._started:.1f}s "
              f"loss={logs.get('loss', float('nan')):.4f} auc={auc:.4f}", flush=True)


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
    parser.add_argument("--run-eagerly", dest="run_eagerly", action="store_true", default=None,
                        help="skip graph compilation; default on for --smoke")
    parser.add_argument("--no-run-eagerly", dest="run_eagerly", action="store_false",
                        help="force graph mode even in --smoke")
    parser.add_argument("--fit-backend", choices=("loop", "keras"), default="loop",
                        help="loop: explicit train_on_batch loop (default). "
                             "keras: model.fit, useful for comparison.")
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
    configured_seq_len = int(
        smoke_cfg["sfrnnr_seq_len"] if smoke else training["sfrnnr_seq_len"]
    )
    seq_len = args.seq_len or configured_seq_len
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

    # Eager by default. The model has ~2k parameters, and the explicit loop
    # already feeds a single fixed batch shape, so graph mode's main advantage
    # is gone. Tracing a several hundred step recurrent graph has been observed
    # to stall on some TensorFlow builds, and correctness beats a speedup on a
    # model this small. Opt into graph mode with --no-run-eagerly.
    run_eagerly = True if args.run_eagerly is None else bool(args.run_eagerly)
    model = build_sfrnnr_model(
        seq_len=seq_len,
        n_factors=N_FACTORS,
        n_mfs=args.n_mfs or N_MFS,
        gru_units=args.gru_units,
        rule_units=args.rule_units,
        run_eagerly=run_eagerly,
    )
    print(f"[train_sfrnnr] model compiled (run_eagerly={run_eagerly}, "
          f"params={model.count_params()})", flush=True)

    patience = 0 if smoke else int(training["sfrnnr_patience"])
    batches_per_epoch = max(1, len(train_idx) // batch_size)
    dropped_per_epoch = len(train_idx) - batches_per_epoch * batch_size
    print(f"[train_sfrnnr] fitting for up to {epochs} epoch(s) with the "
          f"'{args.fit_backend}' backend: X{X[train_idx].shape} batch={batch_size} "
          f"({batches_per_epoch} batches per epoch, {dropped_per_epoch} sequence(s) "
          f"dropped per epoch by reshuffling)", flush=True)

    fit_started = time.perf_counter()
    if args.fit_backend == "loop":
        history = train_with_loop(
            model,
            X[train_idx], y[train_idx], thresholds[train_idx],
            X[val_idx], y[val_idx],
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seed=args.seed,
        )
    else:
        callbacks: list = [EpochProgress()]
        if patience > 0:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_lfp_auc", mode="max", patience=patience,
                    restore_best_weights=True, verbose=1,
                )
            )
        history = model.fit(
            X[train_idx],
            {"lfp": y[train_idx], "lfp_threshold": thresholds[train_idx]},
            validation_data=(X[val_idx], {"lfp": y[val_idx], "lfp_threshold": thresholds[val_idx]}),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
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
        "early_stopping": patience > 0,
        "fit_backend": args.fit_backend,
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
