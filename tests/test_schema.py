"""The feature contract. A silent column order change here corrupts everything."""

import numpy as np
import pandas as pd
import pytest

from schema import (
    FACTOR_COLS,
    FEATURES,
    SchemaError,
    assert_matrix_shape,
    constant_columns,
    feature_matrix,
)


def test_feature_list_is_stable():
    assert len(FEATURES) == 14
    assert len(set(FEATURES)) == len(FEATURES), "duplicate feature name"
    assert FEATURES[0] == "neighbor_count"


def test_paper_factor_list_is_stable():
    assert len(FACTOR_COLS) == 9
    assert len(set(FACTOR_COLS)) == len(FACTOR_COLS)


def test_feature_matrix_preserves_declared_order():
    shuffled = list(reversed(FEATURES))
    df = pd.DataFrame({c: np.arange(3, dtype=float) + i for i, c in enumerate(shuffled)})
    X = feature_matrix(df)
    for j, name in enumerate(FEATURES):
        assert np.allclose(X[:, j], df[name].to_numpy())


def test_missing_columns_raise():
    df = pd.DataFrame({c: [0.0] for c in FEATURES[:-1]})
    with pytest.raises(SchemaError):
        feature_matrix(df)


def test_wrong_width_matrix_raises():
    with pytest.raises(SchemaError):
        assert_matrix_shape(np.zeros((4, 3)), len(FEATURES), "test")


def test_constant_columns_are_detected():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0], "c": [np.nan] * 3})
    assert set(constant_columns(df, ["a", "b", "c"])) == {"a", "c"}
