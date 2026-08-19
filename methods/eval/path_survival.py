"""Ground truth routing metric: does the chosen path still exist H seconds later?

Why this module exists. The original evaluation scored a route by the mean and
minimum of the very reliabilities that Dijkstra had just minimised over. The
optimiser and the judge were the same function, so the method could not lose:
across nine thousand routing decisions it was never once worse than shortest
path. That is a property of the algebra, not evidence that the predictor works.

This metric is computed from the mobility trace alone. A path chosen at time t
survives if every one of its hops is still within the communication radius at
t + horizon. No model output enters the calculation, so all three routing
methods can be compared on a number none of them can influence.

It answers the question the project actually claims to answer: did routing
around the predicted failures keep the route up longer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

NodeId = int
Position = Tuple[float, float]


class PositionLookup:
    """Fast (run_id, time, node_id) -> (x, y) lookup over the mobility trace."""

    def __init__(self, frame: pd.DataFrame):
        needed = {"run_id", "time", "node_id", "x", "y"}
        missing = needed - set(frame.columns)
        if missing:
            raise KeyError(f"PositionLookup needs columns {sorted(missing)}")
        self._by_run_time: Dict[Tuple[int, float], Dict[NodeId, Position]] = {}
        cols = frame[["run_id", "time", "node_id", "x", "y"]].to_numpy()
        for run_id, t, nid, x, y in cols:
            key = (int(run_id), float(t))
            self._by_run_time.setdefault(key, {})[int(nid)] = (float(x), float(y))
        self._times_by_run: Dict[int, np.ndarray] = {}
        for run_id, grp in frame.groupby("run_id"):
            self._times_by_run[int(run_id)] = np.sort(grp["time"].unique().astype(float))

    def snapshot(self, run_id: int, time: float) -> Optional[Dict[NodeId, Position]]:
        return self._by_run_time.get((int(run_id), float(time)))

    def step_size(self, run_id: int) -> float:
        times = self._times_by_run.get(int(run_id))
        if times is None or len(times) < 2:
            return 1.0
        return float(np.median(np.diff(times)))

    def future_time(self, run_id: int, time: float, horizon_steps: int) -> Optional[float]:
        """The timestamp `horizon_steps` samples after `time`, if it exists."""
        times = self._times_by_run.get(int(run_id))
        if times is None:
            return None
        idx = int(np.searchsorted(times, float(time)))
        if idx >= len(times) or not np.isclose(times[idx], float(time)):
            return None
        target = idx + int(horizon_steps)
        if target >= len(times):
            return None
        return float(times[target])


def path_hops(path: Sequence[NodeId]) -> list:
    if not path or len(path) < 2:
        return []
    return list(zip(path[:-1], path[1:]))


def evaluate_path_survival(
    path: Optional[Sequence[NodeId]],
    lookup: PositionLookup,
    run_id: int,
    time: float,
    radius: float,
    horizon_steps: int,
) -> dict:
    """Return survival facts for one chosen path.

    survived            1 if every hop is still within radius at t + horizon
    broken_hops         how many hops exceeded the radius
    surviving_fraction  fraction of hops that held
    first_break_index   position of the first broken hop, or -1
    evaluable           0 when the future snapshot does not exist (end of run)
    """
    hops = path_hops(path or [])
    base = {
        "survived": 0,
        "broken_hops": 0,
        "hop_count": len(hops),
        "surviving_fraction": 0.0,
        "first_break_index": -1,
        "evaluable": 0,
        "max_hop_distance_future": float("nan"),
    }
    if not hops:
        return base

    t_future = lookup.future_time(run_id, time, horizon_steps)
    if t_future is None:
        return base
    future = lookup.snapshot(run_id, t_future)
    if not future:
        return base

    broken = 0
    first_break = -1
    max_d = 0.0
    for i, (u, v) in enumerate(hops):
        pu, pv = future.get(int(u)), future.get(int(v))
        if pu is None or pv is None:
            broken += 1
            if first_break < 0:
                first_break = i
            continue
        d = float(np.hypot(pu[0] - pv[0], pu[1] - pv[1]))
        max_d = max(max_d, d)
        if d > radius:
            broken += 1
            if first_break < 0:
                first_break = i

    base.update(
        survived=int(broken == 0),
        broken_hops=int(broken),
        surviving_fraction=float((len(hops) - broken) / len(hops)),
        first_break_index=int(first_break),
        evaluable=1,
        max_hop_distance_future=max_d,
    )
    return base


def summarise_survival(rows: Iterable[dict]) -> dict:
    """Aggregate survival facts, ignoring decisions that had no future snapshot."""
    rows = [r for r in rows if r.get("evaluable")]
    if not rows:
        return {
            "n_evaluable": 0,
            "survival_rate": float("nan"),
            "mean_surviving_fraction": float("nan"),
            "mean_broken_hops": float("nan"),
        }
    return {
        "n_evaluable": len(rows),
        "survival_rate": float(np.mean([r["survived"] for r in rows])),
        "mean_surviving_fraction": float(np.mean([r["surviving_fraction"] for r in rows])),
        "mean_broken_hops": float(np.mean([r["broken_hops"] for r in rows])),
    }
