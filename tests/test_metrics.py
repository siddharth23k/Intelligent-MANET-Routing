"""Route metrics and the statistics used to report them."""

import networkx as nx
import numpy as np
import pandas as pd

from metrics import paired_run_test, proportion_test, route_metrics, win_loss_tie


def _graph():
    g = nx.Graph()
    g.add_edge(0, 1, weight=0.1, reliability=0.9)
    g.add_edge(1, 2, weight=0.5, reliability=0.6)
    return g


def test_route_metrics_on_a_known_path():
    m = route_metrics(_graph(), [0, 1, 2])
    assert m["hop_count"] == 2
    assert np.isclose(m["min_reliability"], 0.6)
    assert np.isclose(m["avg_reliability"], 0.75)


def test_route_metrics_on_a_degenerate_path():
    assert route_metrics(_graph(), None)["hop_count"] == 0
    assert route_metrics(_graph(), [0])["hop_count"] == 0


def test_win_loss_tie_sums_to_one():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 1.0, 5.0, 4.0]})
    w = win_loss_tie(df, "a", "b")
    assert np.isclose(w["win_rate"] + w["tie_rate"] + w["loss_rate"], 1.0)
    assert np.isclose(w["win_rate"], 0.25)
    assert np.isclose(w["tie_rate"], 0.5)
    assert np.isclose(w["loss_rate"], 0.25)


def test_significance_is_tested_at_the_run_level():
    """Two runs with many correlated decisions each must count as two units of
    evidence, not as many."""
    df = pd.DataFrame({
        "run_id": [1] * 50 + [2] * 50,
        "a": [0.8] * 50 + [0.9] * 50,
        "b": [0.7] * 50 + [0.8] * 50,
    })
    r = paired_run_test(df, "a", "b")
    assert r["n_runs"] == 2
    assert np.isclose(r["mean_delta"], 0.1)


def test_paired_test_handles_a_single_run_without_crashing():
    df = pd.DataFrame({"run_id": [1, 1], "a": [0.5, 0.6], "b": [0.4, 0.5]})
    r = paired_run_test(df, "a", "b")
    assert r["n_runs"] == 1
    assert np.isnan(r["t_p_value"])


def test_identical_columns_produce_no_effect():
    df = pd.DataFrame({"run_id": [1, 1, 2, 2], "a": [0.5] * 4, "b": [0.5] * 4})
    r = paired_run_test(df, "a", "b")
    assert np.isclose(r["mean_delta"], 0.0)
    w = win_loss_tie(df, "a", "b")
    assert np.isclose(w["tie_rate"], 1.0)


def test_proportion_test_direction():
    r = proportion_test(80, 100, 50, 100)
    assert r["delta"] > 0
    assert 0.0 <= r["p_value"] <= 1.0
