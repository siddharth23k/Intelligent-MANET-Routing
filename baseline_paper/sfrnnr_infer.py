"""Run trained SFRNNR over a dataframe; fill lfp and lfp_threshold columns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sfrnnr_model import FACTOR_COLS, load_sfrnnr


def _minmax_global(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if np.isclose(hi - lo, 0.0):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def apply_sfrnnr(
    df: pd.DataFrame,
    repo_root: Path,
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> pd.DataFrame:
    """Apply SFRNNR; returns copy with lfp, lfp_threshold (same row order as df)."""
    repo_root = Path(repo_root)
    model_path = model_path or (repo_root / "models" / "sfrnnr_paper.keras")
    meta_path = meta_path or (repo_root / "models" / "sfrnnr_meta.json")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["seq_len"])

    model = load_sfrnnr(str(model_path))

    work = df.reset_index(drop=True).copy()
    n = {c: _minmax_global(work[c]) for c in FACTOR_COLS}
    for c in FACTOR_COLS:
        work[c] = n[c]

    lfp_vals = np.zeros(len(work), dtype=np.float64)
    thr_vals = np.zeros(len(work), dtype=np.float64)

    for (rid, nid), g in work.groupby(["run_id", "node_id"], sort=False):
        g = g.sort_values("time")
        pos = g.index.to_numpy()
        X = g[FACTOR_COLS].values.astype(np.float32)
        t = X.shape[0]
        pad = seq_len - t
        if pad > 0:
            last = X[-1:]
            Xp = np.vstack([X, np.repeat(last, pad, axis=0)])
        else:
            Xp = X[:seq_len]
        pred = model.predict(Xp[np.newaxis, ...], batch_size=1, verbose=0)
        lf = pred["lfp"][0, :, 0]
        th = pred["lfp_threshold"][0, :, 0]
        for k in range(t):
            lfp_vals[pos[k]] = lf[k]
            thr_vals[pos[k]] = th[k]

    out = df.copy()
    out["lfp"] = lfp_vals
    out["lfp_threshold"] = thr_vals
    return out
