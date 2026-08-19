"""Isolate which TensorFlow component is slow or stalling on this machine.

Builds up from a trivial model to the full SFRNNR, timing each step, in graph
mode and in eager mode. Run it when stage 5 of the smoke test does not finish:
the last line printed names the component that hung.

    python pipeline/diagnose_tf.py
    python pipeline/diagnose_tf.py --timeout 30
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

print(f"python  {platform.python_version()} on {platform.system()} {platform.machine()}", flush=True)
print("importing tensorflow ...", flush=True)
_t0 = time.perf_counter()
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

print(f"tensorflow {tf.__version__}, keras {keras.__version__} "
      f"(import took {time.perf_counter() - _t0:.1f}s)", flush=True)
print(f"visible devices: {[d.device_type for d in tf.config.list_physical_devices()]}", flush=True)

from sfrnnr_model import FuzzificationLayer, build_sfrnnr_model  # noqa: E402

BATCH = 32
SEQ_LEN = 24
N_FACTORS = 9
N_SAMPLES = 64


def _data():
    rng = np.random.default_rng(0)
    X = rng.random((N_SAMPLES, SEQ_LEN, N_FACTORS)).astype("float32")
    y = rng.integers(0, 2, size=(N_SAMPLES, SEQ_LEN, 1)).astype("float32")
    return X, y


def timed(label: str, fn, timeout: float) -> bool:
    """Run one probe, report the elapsed time, and flag anything over budget."""
    print(f"  {label} ...", end="", flush=True)
    started = time.perf_counter()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - a probe failing is a result
        print(f" FAILED after {time.perf_counter() - started:.1f}s: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return False
    elapsed = time.perf_counter() - started
    flag = "  <-- SLOW" if elapsed > timeout else ""
    print(f" {elapsed:.1f}s{flag}", flush=True)
    return True


def probe_dense(eager: bool):
    X, y = _data()
    model = keras.Sequential([
        keras.Input(shape=(SEQ_LEN, N_FACTORS)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", run_eagerly=eager, jit_compile=False)
    model.fit(X, y, epochs=1, batch_size=BATCH, verbose=0)


def probe_gru(eager: bool):
    X, y = _data()
    model = keras.Sequential([
        keras.Input(shape=(SEQ_LEN, N_FACTORS)),
        layers.GRU(16, return_sequences=True),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", run_eagerly=eager, jit_compile=False)
    model.fit(X, y, epochs=1, batch_size=BATCH, verbose=0)


def probe_fuzzification(eager: bool):
    X, y = _data()
    model = keras.Sequential([
        keras.Input(shape=(SEQ_LEN, N_FACTORS)),
        FuzzificationLayer(n_inputs=N_FACTORS, n_mfs=2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", run_eagerly=eager, jit_compile=False)
    model.fit(X, y, epochs=1, batch_size=BATCH, verbose=0)


def probe_auc_metric(eager: bool):
    X, y = _data()
    model = keras.Sequential([
        keras.Input(shape=(SEQ_LEN, N_FACTORS)),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc"],
                  run_eagerly=eager, jit_compile=False)
    model.fit(X, y, epochs=1, batch_size=BATCH, verbose=0)


def probe_full_sfrnnr(eager: bool):
    X, y = _data()
    thresholds = np.full_like(y, 0.5)
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    model.fit(
        X,
        {"lfp": y, "lfp_threshold": thresholds},
        validation_data=(X, {"lfp": y, "lfp_threshold": thresholds}),
        epochs=1,
        batch_size=BATCH,
        verbose=0,
    )


def probe_train_on_batch(eager: bool):
    """What the explicit training loop uses. Bypasses fit's data adapter."""
    X, y = _data()
    thresholds = np.full_like(y, 0.5)
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    for start in range(0, N_SAMPLES, BATCH):
        stop = start + BATCH
        model.train_on_batch(
            X[start:stop],
            {"lfp": y[start:stop], "lfp_threshold": thresholds[start:stop]},
            return_dict=True,
        )


def probe_fit_ragged_validation(eager: bool):
    """fit with a validation set that does not divide evenly by the batch size.

    This is the shape stage 5 used to pass to model.fit: 64 training sequences
    and 100 validation sequences at batch 32, so the final validation batch has
    4 rows. It is the probe that reproduces the stall. Nothing in the pipeline
    calls fit this way any more; it is kept so a regression is detectable.
    """
    X, y = _data()
    thresholds = np.full_like(y, 0.5)
    rng = np.random.default_rng(1)
    X_val = rng.random((100, SEQ_LEN, N_FACTORS)).astype("float32")
    y_val = rng.integers(0, 2, size=(100, SEQ_LEN, 1)).astype("float32")
    thr_val = np.full_like(y_val, 0.5)
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    model.fit(
        X,
        {"lfp": y, "lfp_threshold": thresholds},
        validation_data=(X_val, {"lfp": y_val, "lfp_threshold": thr_val}),
        epochs=2,
        batch_size=BATCH,
        verbose=0,
    )


def probe_predict(eager: bool):
    X, _ = _data()
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    model.predict(X, batch_size=BATCH, verbose=0)


def probe_predict_after_training(eager: bool):
    """model.predict on a model that has already run a training step.

    This ordering is what the pipeline does, and it is the one that stalls.
    A fresh model predicts fine.
    """
    X, y = _data()
    thresholds = np.full_like(y, 0.5)
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    model.train_on_batch(
        X[:BATCH], {"lfp": y[:BATCH], "lfp_threshold": thresholds[:BATCH]}, return_dict=True
    )
    model.predict(X, batch_size=BATCH, verbose=0)


def probe_direct_call_after_training(eager: bool):
    """What the pipeline uses instead: a direct forward pass, no data adapter."""
    X, y = _data()
    thresholds = np.full_like(y, 0.5)
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    model.train_on_batch(
        X[:BATCH], {"lfp": y[:BATCH], "lfp_threshold": thresholds[:BATCH]}, return_dict=True
    )
    for start in range(0, len(X), BATCH):
        model(X[start : start + BATCH], training=False)


def probe_save_load(eager: bool):
    model = build_sfrnnr_model(seq_len=SEQ_LEN, run_eagerly=eager)
    path = ROOT / "results" / "models" / "_diagnose_tmp.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    from sfrnnr_model import load_sfrnnr

    load_sfrnnr(str(path))
    path.unlink(missing_ok=True)


PROBES = [
    ("dense only", probe_dense),
    ("dense + auc metric", probe_auc_metric),
    ("gru", probe_gru),
    ("fuzzification layer", probe_fuzzification),
    ("full sfrnnr fit", probe_full_sfrnnr),
    ("train_on_batch loop", probe_train_on_batch),
    ("fit, ragged validation", probe_fit_ragged_validation),
    ("full sfrnnr predict", probe_predict),
    ("predict after training", probe_predict_after_training),
    ("direct call after training", probe_direct_call_after_training),
    ("save and load", probe_save_load),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the slow TensorFlow component.")
    parser.add_argument("--timeout", type=float, default=20.0,
                        help="seconds above which a probe is flagged as slow")
    parser.add_argument("--mode", choices=("both", "eager", "graph"), default="both")
    args = parser.parse_args()

    modes = {"both": [True, False], "eager": [True], "graph": [False]}[args.mode]
    for eager in modes:
        print(f"\n=== {'eager' if eager else 'graph'} mode ===", flush=True)
        for label, probe in PROBES:
            if not timed(label, lambda p=probe, e=eager: p(e), args.timeout):
                break

    print("\nIf a probe never printed its time, that is the component that hangs.")
    print("The pipeline trains through 'train_on_batch loop', not 'fit'. If only the")
    print("fit probes stall, the pipeline is unaffected. Force either backend with")
    print("  python pipeline/train_sfrnnr_paper.py --smoke --fit-backend loop")
    print("  python pipeline/train_sfrnnr_paper.py --smoke --fit-backend keras")


if __name__ == "__main__":
    main()
