"""Ground truth routing metric: does the chosen path still exist H steps later?

Scoring a route by the same reliabilities Dijkstra minimised over makes the
method mathematically incapable of losing. This metric is computed from the
mobility trace alone, so no model can influence it, and all three routers are
comparable on it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

setup_paths()

NodeId = int
Position = Tuple[float, float]

REQUIRED_COLUMNS = ("run_id", "time", "node_id", "x", "y")


class PositionLookup:
    """(run_id, time, node_id) -> (x, y), indexed once for repeated queries."""

    def __init__(self, frame: pd.DataFrame):
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise KeyError(f"PositionLookup needs columns {sorted(missing)}")

        self._by_run_time: Dict[Tuple[int, float], Dict[NodeId, Position]] = {}
        for run_id, t, node_id, x, y in frame[list(REQUIRED_COLUMNS)].to_numpy():
            self._by_run_time.setdefault((int(run_id), float(t)), {})[int(node_id)] = (
                float(x),
                float(y),
            )

        self._times_by_run: Dict[int, np.ndarray] = {
            int(run_id): np.sort(group["time"].unique().astype(float))
            for run_id, group in frame.groupby("run_id")
        }

    def snapshot(self, run_id: int, time: float) -> Optional[Dict[NodeId, Position]]:
        return self._by_run_time.get((int(run_id), float(time)))

    def step_size(self, run_id: int) -> float:
        times = self._times_by_run.get(int(run_id))
        if times is None or len(times) < 2:
            return 1.0
        return float(np.median(np.diff(times)))

    def future_time(self, run_id: int, time: float, horizon_steps: int) -> Optional[float]:
        """Timestamp `horizon_steps` samples after `time`, or None past the end."""
        times = self._times_by_run.get(int(run_id))
        if times is None:
            return None
        index = int(np.searchsorted(times, float(time)))
        if index >= len(times) or not np.isclose(times[index], float(time)):
            return None
        target = index + int(horizon_steps)
        return float(times[target]) if target < len(times) else None


def path_hops(path: Sequence[NodeId]) -> List[Tuple[NodeId, NodeId]]:
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
    """Survival facts for one chosen path.

    `evaluable` is 0 when there is no future snapshot, for example near the end
    of a run. Those decisions are excluded from the summary rather than counted
    as failures.
    """
    hops = path_hops(path or [])
    result = {
        "survived": 0,
        "broken_hops": 0,
        "hop_count": len(hops),
        "surviving_fraction": 0.0,
        "first_break_index": -1,
        "evaluable": 0,
        "max_hop_distance_future": float("nan"),
    }
    if not hops:
        return result

    future_time = lookup.future_time(run_id, time, horizon_steps)
    if future_time is None:
        return result
    future = lookup.snapshot(run_id, future_time)
    if not future:
        return result

    broken, first_break, max_distance = 0, -1, 0.0
    for index, (u, v) in enumerate(hops):
        pu, pv = future.get(int(u)), future.get(int(v))
        if pu is None or pv is None:
            broken += 1
            first_break = index if first_break < 0 else first_break
            continue
        distance = float(np.hypot(pu[0] - pv[0], pu[1] - pv[1]))
        max_distance = max(max_distance, distance)
        if distance > radius:
            broken += 1
            first_break = index if first_break < 0 else first_break

    result.update(
        survived=int(broken == 0),
        broken_hops=int(broken),
        surviving_fraction=float((len(hops) - broken) / len(hops)),
        first_break_index=int(first_break),
        evaluable=1,
        max_hop_distance_future=max_distance,
    )
    return result


def summarise_survival(rows: Iterable[dict]) -> dict:
    """Aggregate survival facts, ignoring decisions with no future snapshot."""
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
