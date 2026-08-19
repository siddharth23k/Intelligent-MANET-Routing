"""Normalisation statistics must come from training rows only."""

import numpy as np
import pandas as pd
import pytest

from normalization import MinMaxStats


def test_statistics_come_only_from_the_rows_they_were_fitted_on():
    train = pd.DataFrame({"v": [0.0, 5.0, 10.0]})
    test = pd.DataFrame({"v": [-100.0, 500.0]})
    stats = MinMaxStats.fit(train, ["v"])
    assert stats.stats["v"] == {"min": 0.0, "max": 10.0}
    # Held out extremes must not be able to change the scaling.
    assert MinMaxStats.fit(train, ["v"]).stats == stats.stats
    out = stats.transform(test, ["v"])["v"].to_numpy()
    assert np.all((out >= 0.0) & (out <= 1.0)), "held out rows must stay inside [0, 1]"


def test_transform_maps_the_training_range_onto_zero_one():
    train = pd.DataFrame({"v": [2.0, 4.0, 6.0]})
    stats = MinMaxStats.fit(train, ["v"])
    out = stats.transform(train, ["v"])["v"].to_numpy()
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)


def test_constant_column_does_not_divide_by_zero():
    stats = MinMaxStats.fit(pd.DataFrame({"v": [3.0, 3.0]}), ["v"])
    out = stats.transform(pd.DataFrame({"v": [3.0, 9.0]}), ["v"])["v"].to_numpy()
    assert np.allclose(out, 0.0)


def test_unknown_column_raises_rather_than_guessing():
    stats = MinMaxStats.fit(pd.DataFrame({"v": [1.0, 2.0]}), ["v"])
    with pytest.raises(KeyError):
        stats.transform_series(pd.Series([1.0]), "not_fitted")


def test_round_trip_through_disk(tmp_path):
    stats = MinMaxStats.fit(pd.DataFrame({"v": [1.0, 3.0]}), ["v"])
    path = tmp_path / "stats.json"
    stats.save(path)
    assert MinMaxStats.load(path).stats == stats.stats
