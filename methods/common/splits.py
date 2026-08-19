"""Run level train / validation / test splitting.

Rows inside one simulation run are heavily autocorrelated: consecutive samples
differ by one step of motion, share four of five rolling window elements, and
overlap in label horizon. The only independent unit is the run, since each uses
its own NS-3 RNG substream. Training and evaluation both import this module, so
the held out runs are provably the same set.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class RunSplit:
    train_runs: List[int]
    val_runs: List[int]
    test_runs: List[int]

    def as_dict(self) -> Dict[str, List[int]]:
        return asdict(self)

    def assert_disjoint(self) -> "RunSplit":
        train, val, test = set(self.train_runs), set(self.val_runs), set(self.test_runs)
        for a, b, names in ((train, val, "train/val"), (train, test, "train/test"), (val, test, "val/test")):
            if a & b:
                raise ValueError(f"{names} runs overlap: {sorted(a & b)}")
        if not train:
            raise ValueError("empty training split")
        if not test:
            raise ValueError("empty test split")
        return self

    def describe(self) -> str:
        return f"train={self.train_runs} val={self.val_runs} test={self.test_runs}"


def make_run_split(
    run_ids: Iterable[int],
    seed: int,
    test_run_count: int,
    val_run_count: int = 0,
) -> RunSplit:
    """Partition runs deterministically.

    Test runs are drawn first so changing the validation size can never move a
    run into or out of the test set.
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
    """Slice a frame into (train, val, test) by run id."""
    train = df[df[column].isin(split.train_runs)]
    val = df[df[column].isin(split.val_runs)] if split.val_runs else df.iloc[0:0]
    test = df[df[column].isin(split.test_runs)]
    return train, val, test


def assert_no_row_overlap(train_df, test_df, keys: Sequence[str] = ("run_id", "node_id", "time")) -> None:
    """Prove no identical (run, node, time) row sits on both sides."""
    left = set(map(tuple, train_df[list(keys)].to_numpy().tolist()))
    right = set(map(tuple, test_df[list(keys)].to_numpy().tolist()))
    overlap = left & right
    if overlap:
        raise ValueError(f"{len(overlap)} rows in both train and test, e.g. {list(overlap)[:3]}")
