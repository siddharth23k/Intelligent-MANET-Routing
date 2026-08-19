"""Run a trained SFRNNR over a dataframe and attach lfp / lfp_threshold.

Two changes from the original. Inference is batched instead of one Keras call
per (run, node) sequence, which took thousands of calls and dominated the whole
pipeline's wall clock. And the factor normalisation uses the statistics that
were fitted on the training runs and saved at training time, rather than
recomputing min and max over whatever frame is being scored, which leaked the
held out runs' extremes into the scaling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

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


def _tiled_windows(work: pd.DataFrame, seq_len: int):
    """Cover every row of every (run, node) track with fixed length windows.

    A track is longer than the model's input window, so a single window would
    leave most rows unscored. This tiles consecutive windows across the track,
    clamping the last one so it ends at the final sample. Overlapping rows are
    written by the later window, which carries more recurrent context.

    Returns (windows, row_index_blocks) where row_index_blocks[i] holds the
    dataframe row positions that window i is responsible for.
    """
    n_factors = len(FACTOR_COLS)
    windows, blocks = [], []

    for _, g in work.groupby(["run_id", "node_id"], sort=False):
        g = g.sort_values("time")
        X = g[FACTOR_COLS].to_numpy(dtype=np.float32)
        rows = g.index.to_numpy()
        t = len(X)
        if t == 0:
            continue

        if t < seq_len:
            pad = seq_len - t
            window = np.vstack([np.repeat(X[:1], pad, axis=0), X])
            windows.append(window)
            blocks.append((rows, np.arange(pad, seq_len)))
            continue

        start = 0
        while start < t:
            s0 = min(start, t - seq_len)
            windows.append(X[s0 : s0 + seq_len])
            # Only claim the rows this window is responsible for, so earlier
            # windows are not overwritten by the clamped final one.
            claim_lo = max(start, s0)
            claim_hi = min(start + seq_len, t)
            blocks.append(
                (rows[claim_lo:claim_hi], np.arange(claim_lo - s0, claim_hi - s0))
            )
            start += seq_len

    stacked = (
        np.stack(windows)
        if windows
        else np.zeros((0, seq_len, n_factors), dtype=np.float32)
    )
    return stacked, blocks


def apply_sfrnnr(
    df: pd.DataFrame,
    repo_root: Optional[Path] = None,
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    root = Path(repo_root or ROOT)
    model_path = Path(model_path or (root / "results" / "models" / "sfrnnr_paper.keras"))
    meta_path = Path(meta_path or (root / "results" / "models" / "sfrnnr_meta.json"))

    assert_columns(df, ["run_id", "node_id", "time"] + FACTOR_COLS, "apply_sfrnnr")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["seq_len"])

    if "factor_norm_stats" in meta:
        stats = MinMaxStats(meta["factor_norm_stats"])
    else:
        raise KeyError(
            f"{meta_path} has no 'factor_norm_stats'. Retrain with "
            "pipeline/train_sfrnnr_paper.py so the training-run statistics are persisted."
        )

    model = load_sfrnnr(str(model_path))

    work = df.reset_index(drop=True).copy()
    work = stats.transform(work, FACTOR_COLS)

    X, blocks = _tiled_windows(work, seq_len)
    lfp_vals = np.full(len(work), np.nan, dtype=np.float64)
    thr_vals = np.full(len(work), np.nan, dtype=np.float64)

    if len(X):
        pred = model.predict(X, batch_size=batch_size, verbose=0)
        lfp_seq = np.asarray(pred["lfp"])[:, :, 0]
        thr_seq = np.asarray(pred["lfp_threshold"])[:, :, 0]
        for i, (rows, valid) in enumerate(blocks):
            n = min(len(rows), len(valid))
            if n:
                lfp_vals[rows[:n]] = lfp_seq[i, valid[:n]]
                thr_vals[rows[:n]] = thr_seq[i, valid[:n]]

    unscored = int(np.isnan(lfp_vals).sum())
    if unscored:
        raise RuntimeError(
            f"{unscored} of {len(work)} rows were never scored by the SFRNNR. "
            "Every row must be covered, otherwise the baseline is silently "
            "compared on default values."
        )

    out = df.copy()
    out["lfp"] = lfp_vals
    out["lfp_threshold"] = thr_vals
    return out
