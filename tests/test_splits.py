"""The split is the project's main defence against leakage, so it gets tests."""

import pandas as pd
import pytest

from splits import RunSplit, assert_no_row_overlap, make_run_split, split_frame


def test_train_val_test_are_disjoint():
    s = make_run_split(range(1, 31), seed=42, test_run_count=6, val_run_count=4)
    assert not set(s.train_runs) & set(s.test_runs)
    assert not set(s.val_runs) & set(s.test_runs)
    assert not set(s.train_runs) & set(s.val_runs)
    assert len(s.train_runs) + len(s.val_runs) + len(s.test_runs) == 30


def test_split_is_deterministic_for_a_seed():
    a = make_run_split(range(1, 31), seed=42, test_run_count=6, val_run_count=4)
    b = make_run_split(range(1, 31), seed=42, test_run_count=6, val_run_count=4)
    assert a == b


def test_adding_a_validation_split_does_not_move_the_test_runs():
    """Test runs are drawn first on purpose, so tuning the validation size can
    never quietly change which runs the final numbers are measured on."""
    without = make_run_split(range(1, 31), seed=42, test_run_count=6, val_run_count=0)
    with_val = make_run_split(range(1, 31), seed=42, test_run_count=6, val_run_count=4)
    assert without.test_runs == with_val.test_runs


def test_disjointness_is_enforced():
    with pytest.raises(ValueError):
        RunSplit(train_runs=[1, 2], val_runs=[], test_runs=[2]).assert_disjoint()


def test_no_row_appears_on_both_sides():
    df = pd.DataFrame({
        "run_id": [1, 1, 2, 2],
        "node_id": [0, 1, 0, 1],
        "time": [1.0, 1.0, 1.0, 1.0],
    })
    split = make_run_split([1, 2], seed=0, test_run_count=1, val_run_count=0)
    train, _, test = split_frame(df, split)
    assert_no_row_overlap(train, test)


def test_row_overlap_is_detected():
    df = pd.DataFrame({"run_id": [1], "node_id": [0], "time": [1.0]})
    with pytest.raises(ValueError):
        assert_no_row_overlap(df, df)
