"""Regression tests for the bug that made the adaptive threshold a constant.

The original implementation looked up "RSSI_norm", "LS_norm", "LET_norm",
"LL_d_norm" and "ND_norm". No pipeline stage ever produced a column with a
_norm suffix, so `dict.get(key, 0.5)` returned the default for five of six
inputs and the threshold varied with one variable instead of six.
"""

import numpy as np
import pandas as pd
import pytest

from schema import THRESHOLD_INPUTS
from threshold_model import AdaptiveThresholdModel, MissingThresholdInput


def _row(**over):
    base = {k: 0.5 for k in THRESHOLD_INPUTS}
    base.update(over)
    return base


def test_missing_input_raises_instead_of_defaulting():
    bad = _row()
    bad.pop("RSSI")
    with pytest.raises(MissingThresholdInput):
        AdaptiveThresholdModel.predict_threshold(bad)


def test_the_old_norm_suffixed_keys_are_not_accepted():
    legacy = {f"{k}_norm": 0.5 for k in THRESHOLD_INPUTS}
    with pytest.raises(MissingThresholdInput):
        AdaptiveThresholdModel.predict_threshold(legacy)


@pytest.mark.parametrize("field", THRESHOLD_INPUTS)
def test_every_declared_input_actually_moves_the_threshold(field):
    low = AdaptiveThresholdModel.predict_threshold(_row(**{field: 0.0}))
    high = AdaptiveThresholdModel.predict_threshold(_row(**{field: 1.0}))
    assert not np.isclose(low, high), f"{field} has no effect on the threshold"


def test_threshold_stays_inside_the_paper_band():
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = AdaptiveThresholdModel.predict_threshold(_row(**{k: v for k in THRESHOLD_INPUTS}))
        assert 0.2 <= t <= 0.8


def test_frame_and_row_paths_agree():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.random((16, len(THRESHOLD_INPUTS))), columns=THRESHOLD_INPUTS)
    vec = AdaptiveThresholdModel.predict_threshold_frame(frame)
    one_by_one = np.array([
        AdaptiveThresholdModel.predict_threshold(r) for r in frame.to_dict("records")
    ])
    assert np.allclose(vec, one_by_one, atol=1e-5)


def test_frame_path_rejects_a_missing_column():
    frame = pd.DataFrame({k: [0.5] for k in THRESHOLD_INPUTS[:-1]})
    with pytest.raises(MissingThresholdInput):
        AdaptiveThresholdModel.predict_threshold_frame(frame)
