"""Edge weights, and the property that makes Dijkstra the right algorithm."""

import numpy as np
import pandas as pd

from graph_build import (
    build_snapshot_graphs,
    clip_reliability,
    geometric_edges,
    reliability_to_weight,
)


def _snapshot():
    return pd.DataFrame({
        "node_id": [0, 1, 2, 3],
        "x": [0.0, 100.0, 240.0, 1000.0],
        "y": [0.0, 0.0, 0.0, 1000.0],
    })


def test_edges_respect_the_radius():
    edges = set(geometric_edges(_snapshot(), radius=150.0))
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert (0, 2) not in edges       # 240 m apart, out of range
    assert not any(3 in e for e in edges)


def test_grid_index_agrees_with_brute_force():
    rng = np.random.default_rng(0)
    n = 200
    snap = pd.DataFrame({
        "node_id": np.arange(n),
        "x": rng.uniform(0, 1000, n),
        "y": rng.uniform(0, 1000, n),
    })
    radius = 150.0
    fast = set(geometric_edges(snap, radius))
    xs, ys = snap["x"].to_numpy(), snap["y"].to_numpy()
    brute = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2 <= radius ** 2
    }
    assert fast == brute


def test_weights_are_strictly_positive():
    """Dijkstra is only correct for non negative weights, and w = -log(r) goes
    negative the moment r exceeds 1. The clip is load bearing, not cosmetic."""
    for r in (-5.0, 0.0, 0.5, 1.0, 3.0):
        assert reliability_to_weight(r) > 0
    assert 0.0 < clip_reliability(0.0) < 1.0
    assert 0.0 < clip_reliability(1.0) < 1.0


def test_minus_log_weights_turn_addition_into_multiplication():
    rels = np.array([0.9, 0.8, 0.7])
    total_weight = float(np.sum(reliability_to_weight(rels)))
    assert np.isclose(np.exp(-total_weight), float(np.prod(rels)), atol=1e-9)


def test_both_graphs_share_one_edge_set():
    """The paired comparison is only valid if the topology is identical and only
    the weighting differs."""
    g = build_snapshot_graphs(_snapshot(), {0: 0.9, 1: 0.4, 2: 0.8, 3: 0.5}, radius=150.0)
    assert set(g.ml.edges()) == set(g.hop.edges())


def test_edge_reliability_takes_the_weaker_endpoint():
    g = build_snapshot_graphs(_snapshot(), {0: 0.9, 1: 0.4, 2: 0.8, 3: 0.5}, radius=150.0)
    assert np.isclose(g.ml[0][1]["reliability"], 0.4)
