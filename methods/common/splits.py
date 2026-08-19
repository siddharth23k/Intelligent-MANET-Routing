"""Run level train / validation / test splitting.

Rows inside one simulation run are heavily autocorrelated: consecutive samples
for a node differ by one step of motion and share four of five elements of every
rolling window, and the label looks five steps into the future. Splitting rows
at random would put near duplicates on both sides of the split and inflate every
metric. The only unit that is genuinely independent is the simulation run, since
each run uses its own NS-3 RNG substream.

Both the predictor training and the routing evaluation import this module, so
the held out runs are guaranteed to be the same set. That guarantee used to be
an unchecked coincidence of two scripts using the same seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class RunSplit:
    train_runs: List[int]
    val_runs: List[int]
    test_runs: List[int]

    def as_dict(self) -> Dict[str, List[int]]:
        return asdict(self)

    def assert_disjoint(self) -> "RunSplit":
        t, v, s = set(self.train_runs), set(self.val_runs), set(self.test_runs)
        if t & v:
            raise ValueError(f"train and val runs overlap: {sorted(t & v)}")
        if t & s:
            raise ValueError(f"train and test runs overlap: {sorted(t & s)}")
        if v & s:
            raise ValueError(f"val and test runs overlap: {sorted(v & s)}")
        if not t:
            raise ValueError("empty training split")
        if not s:
            raise ValueError("empty test split")
        return self

    def describe(self) -> str:
        return (
            f"train={self.train_runs} "
            f"val={self.val_runs} "
            f"test={self.test_runs}"
        )


def make_run_split(
    run_ids: Iterable[int],
    seed: int,
    test_run_count: int,
    val_run_count: int = 0,
) -> RunSplit:
    """Deterministically partition simulation runs.

    Test runs are drawn first so that adding or removing a validation split can
    never move a run into or out of the test set. Validation runs are then drawn
    from what remains, so the test set is touched exactly once, at the end.
    """
    runs = sorted(int(r) for r in run_ids)
    if len(runs) < 2:
        raise ValueError(f"need at least 2 runs to split, got {runs}")

    k_test = max(1, min(int(test_run_count), len(runs) - 1))
    test = sorted(random.Random(seed).sample(runs, k=k_test))

    remaining = [r for r in runs if r not in set(test)]
    k_val = max(0, min(int(val_run_count), max(0, len(remaining) - 1)))
    val = sorted(random.Random(seed + 1).sample(remaining, k=k_val)) if k_val else []

    train = [r for r in remaining if r not in set(val)]
    return RunSplit(train_runs=train, val_runs=val, test_runs=test).assert_disjoint()


def split_frame(df, split: RunSplit, column: str = "run_id"):
    """Return (train_df, val_df, test_df) sliced by run id."""
    tr = df[df[column].isin(split.train_runs)]
    va = df[df[column].isin(split.val_runs)] if split.val_runs else df.iloc[0:0]
    te = df[df[column].isin(split.test_runs)]
    return tr, va, te


def assert_no_row_overlap(train_df, test_df, keys: Sequence[str] = ("run_id", "node_id", "time")) -> None:
    """Belt and braces: prove no identical (run, node, time) row is on both sides."""
    a = set(map(tuple, train_df[list(keys)].to_numpy().tolist()))
    b = set(map(tuple, test_df[list(keys)].to_numpy().tolist()))
    overlap = a & b
    if overlap:
        raise ValueError(f"{len(overlap)} rows appear in both train and test, e.g. {list(overlap)[:3]}")
