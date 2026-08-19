"""Run a trained SFRNNR over a dataframe and attach lfp / lfp_threshold.

Inference is batched instead of one Keras call per track, and factor
normalisation reuses the statistics persisted at training time rather than
recomputing min and max over whatever frame is being scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from normalization import MinMaxStats  # noqa: E402
from schema import FACTOR_COLS, assert_columns  # noqa: E402
from sfrnnr_model import load_sfrnnr  # noqa: E402

DEFAULT_MODEL = ROOT / "results" / "models" / "sfrnnr_paper.keras"
DEFAULT_META = ROOT / "results" / "models" / "sfrnnr_meta.json"


def _tiled_windows(work: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, List[tuple]]:
    """Cover every row of every track with fixed length windows.

    Tracks are longer than the model's input window, so one window per track
    would leave most rows unscored. Windows tile the track and the last one is
    clamped to end at the final sample; each window only claims the rows it is
    responsible for.
    """
    windows: List[np.ndarray] = []
    blocks: List[tuple] = []

    for _, group in work.groupby(["run_id", "node_id"], sort=False):
        group = group.sort_values("time")
        X = group[FACTOR_COLS].to_numpy(dtype=np.float32)
        rows = group.index.to_numpy()
        length = len(X)
        if length == 0:
            continue

        if length < seq_len:
            pad = seq_len - length
            windows.append(np.vstack([np.repeat(X[:1], pad, axis=0), X]))
            blocks.append((rows, np.arange(pad, seq_len)))
            continue

        start = 0
        while start < length:
            offset = min(start, length - seq_len)
            windows.append(X[offset : offset + seq_len])
            claim_lo = max(start, offset)
            claim_hi = min(start + seq_len, length)
            blocks.append((rows[claim_lo:claim_hi], np.arange(claim_lo - offset, claim_hi - offset)))
            start += seq_len

    stacked = (
        np.stack(windows)
        if windows
        else np.zeros((0, seq_len, len(FACTOR_COLS)), dtype=np.float32)
    )
    return stacked, blocks


def apply_sfrnnr(
    df: pd.DataFrame,
    repo_root: Optional[Path] = None,
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    batch_size: int = 128,
    verbose: bool = False,
) -> pd.DataFrame:
    """Return a copy of `df` with lfp and lfp_threshold columns attached."""
    root = Path(repo_root or ROOT)
    model_path = Path(model_path or (root / "results" / "models" / "sfrnnr_paper.keras"))
    meta_path = Path(meta_path or (root / "results" / "models" / "sfrnnr_meta.json"))

    assert_columns(df, ["run_id", "node_id", "time"] + FACTOR_COLS, "apply_sfrnnr")

    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    seq_len = int(meta["seq_len"])

    if "factor_norm_stats" not in meta:
        raise KeyError(
            f"{meta_path} has no 'factor_norm_stats'. Retrain with "
            "pipeline/train_sfrnnr_paper.py so training run statistics are persisted."
        )
    stats = MinMaxStats(meta["factor_norm_stats"])

    model = load_sfrnnr(str(model_path))
    work = stats.transform(df.reset_index(drop=True), FACTOR_COLS)

    X, blocks = _tiled_windows(work, seq_len)
    if verbose:
        print(f"[sfrnnr_infer] scoring {len(X)} windows of length {seq_len}", flush=True)

    lfp_values = np.full(len(work), np.nan, dtype=np.float64)
    threshold_values = np.full(len(work), np.nan, dtype=np.float64)

    if len(X):
        prediction = model.predict(X, batch_size=batch_size, verbose=0)
        lfp_sequence = np.asarray(prediction["lfp"])[:, :, 0]
        threshold_sequence = np.asarray(prediction["lfp_threshold"])[:, :, 0]
        for index, (rows, valid) in enumerate(blocks):
            count = min(len(rows), len(valid))
            if count:
                lfp_values[rows[:count]] = lfp_sequence[index, valid[:count]]
                threshold_values[rows[:count]] = threshold_sequence[index, valid[:count]]

    unscored = int(np.isnan(lfp_values).sum())
    if unscored:
        raise RuntimeError(
            f"{unscored} of {len(work)} rows were never scored. Every row must be "
            "covered, otherwise the baseline is compared on default values."
        )

    out = df.copy()
    out["lfp"] = lfp_values
    out["lfp_threshold"] = threshold_values
    return out
