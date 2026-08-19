"""The label definition, and the coupling it has with the features."""

import numpy as np
import pandas as pd

from label_utils import add_link_failure_labels, drop_label_aux_columns, label_diagnostics


def _track(neighbors, rssi=None, horizon_pad=6):
    n = len(neighbors)
    return pd.DataFrame({
        "run_id": [1] * n,
        "node_id": [0] * n,
        "time": np.arange(n, dtype=float),
        "neighbor_count": np.asarray(neighbors, dtype=float),
        "avg_rssi": np.asarray(rssi if rssi is not None else [-50.0] * n, dtype=float),
    })


def test_a_neighbour_collapse_is_labelled_a_failure():
    df = _track([10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2])
    out = add_link_failure_labels(df, horizon=5)
    assert out.loc[1, "link_failure"] == 1     # t=1 sees 2 neighbours at t=6


def test_a_stable_track_is_not_labelled_a_failure():
    df = _track([8] * 12)
    out = add_link_failure_labels(df, horizon=5)
    assert out.loc[0:5, "link_failure"].sum() == 0


def test_an_rssi_collapse_is_labelled_a_failure():
    df = _track([8] * 12, rssi=[-40.0] * 6 + [-70.0] * 6)
    out = add_link_failure_labels(df, horizon=5)
    assert out.loc[1, "link_failure"] == 1


def test_label_uses_the_future_not_the_past():
    """Reversing the track must change the labels; if it does not, the label is
    not actually forward looking."""
    rising = _track([2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10])
    falling = _track(list(reversed([2, 2, 2, 2, 2, 2, 10, 10, 10, 10, 10, 10])))
    a = add_link_failure_labels(rising, horizon=5)["link_failure"].to_numpy()
    b = add_link_failure_labels(falling, horizon=5)["link_failure"].to_numpy()
    assert not np.array_equal(a, b)


def test_aux_columns_are_dropped():
    out = drop_label_aux_columns(add_link_failure_labels(_track([5] * 10)))
    assert "f_neighbors" not in out.columns
    assert "f_rssi" not in out.columns


def test_diagnostics_expose_the_degree_coupling():
    """An isolated node cannot lose two neighbours, so the label is structurally
    easier for sparse nodes. The diagnostic makes that visible rather than
    letting it hide inside a feature importance number."""
    df = pd.concat([
        _track([0] * 12),
        _track([12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1]).assign(node_id=1),
    ], ignore_index=True)
    d = label_diagnostics(df, horizon=5)
    assert "label_rate" in d
    assert d["label_rate_when_isolated"] <= d["label_rate_when_dense"]
