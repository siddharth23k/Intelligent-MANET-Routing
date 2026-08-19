"""The ground truth metric. If this is wrong, the honest number is wrong."""

import numpy as np
import pandas as pd
import pytest

from path_survival import PositionLookup, evaluate_path_survival, path_hops, summarise_survival


def _trace():
    """Two snapshots. Between them node 2 teleports far away, so any path that
    uses the 1-2 link should be recorded as broken."""
    return pd.DataFrame({
        "run_id": [1] * 8,
        "time": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        "node_id": [0, 1, 2, 3] * 2,
        "x": [0.0, 100.0, 200.0, 300.0, 0.0, 100.0, 900.0, 300.0],
        "y": [0.0] * 8,
    })


def test_hops_are_consecutive_pairs():
    assert path_hops([0, 1, 2]) == [(0, 1), (1, 2)]
    assert path_hops([0]) == []
    assert path_hops([]) == []


def test_an_intact_path_survives():
    lk = PositionLookup(_trace())
    r = evaluate_path_survival([0, 1], lk, run_id=1, time=1.0, radius=150.0, horizon_steps=1)
    assert r["evaluable"] == 1
    assert r["survived"] == 1
    assert r["broken_hops"] == 0
    assert r["surviving_fraction"] == 1.0


def test_a_path_whose_hop_moves_out_of_range_does_not_survive():
    lk = PositionLookup(_trace())
    r = evaluate_path_survival([0, 1, 2], lk, run_id=1, time=1.0, radius=150.0, horizon_steps=1)
    assert r["survived"] == 0
    assert r["broken_hops"] == 1
    assert r["first_break_index"] == 1
    assert np.isclose(r["surviving_fraction"], 0.5)


def test_survival_is_independent_of_any_model_output():
    """The whole point of this metric: it is a function of positions only."""
    lk = PositionLookup(_trace())
    a = evaluate_path_survival([0, 1], lk, 1, 1.0, 150.0, 1)
    b = evaluate_path_survival([0, 1], lk, 1, 1.0, 150.0, 1)
    assert a == b


def test_radius_boundary_is_inclusive():
    df = pd.DataFrame({
        "run_id": [1] * 4, "time": [1.0, 1.0, 2.0, 2.0],
        "node_id": [0, 1, 0, 1], "x": [0.0, 0.0, 0.0, 150.0], "y": [0.0] * 4,
    })
    lk = PositionLookup(df)
    assert evaluate_path_survival([0, 1], lk, 1, 1.0, 150.0, 1)["survived"] == 1
    assert evaluate_path_survival([0, 1], lk, 1, 1.0, 149.9, 1)["survived"] == 0


def test_decisions_without_a_future_snapshot_are_not_evaluable():
    """Routes chosen near the end of a run must be excluded, not counted as
    failures, otherwise the metric is biased against every method equally but
    for the wrong reason."""
    lk = PositionLookup(_trace())
    r = evaluate_path_survival([0, 1], lk, run_id=1, time=2.0, radius=150.0, horizon_steps=1)
    assert r["evaluable"] == 0


def test_empty_and_single_node_paths_are_not_evaluable():
    lk = PositionLookup(_trace())
    for p in (None, [], [0]):
        assert evaluate_path_survival(p, lk, 1, 1.0, 150.0, 1)["evaluable"] == 0


def test_summary_ignores_non_evaluable_rows():
    rows = [
        {"survived": 1, "evaluable": 1, "surviving_fraction": 1.0, "broken_hops": 0},
        {"survived": 0, "evaluable": 1, "surviving_fraction": 0.5, "broken_hops": 1},
        {"survived": 0, "evaluable": 0, "surviving_fraction": 0.0, "broken_hops": 0},
    ]
    s = summarise_survival(rows)
    assert s["n_evaluable"] == 2
    assert np.isclose(s["survival_rate"], 0.5)


def test_lookup_requires_position_columns():
    with pytest.raises(KeyError):
        PositionLookup(pd.DataFrame({"run_id": [1], "time": [1.0]}))
