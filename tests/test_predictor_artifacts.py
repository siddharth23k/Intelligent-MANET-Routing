"""Guards against the failure that made the shipped 'ensemble' a single model.

Both artifact files once deserialised to the same RandomForestClassifier. The
blend is w*p + (1-w)*p, so it evaluated to p and nothing raised. These tests fail
that situation loudly.
"""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "results" / "models"

pytestmark = pytest.mark.skipif(
    not (MODELS / "predictor_schema.json").exists(),
    reason="no trained artifacts; run pipeline/train_predictor.py --smoke first",
)


@pytest.fixture(scope="module")
def predictor():
    from predict import LinkFailurePredictor

    return LinkFailurePredictor()


def test_members_are_different_model_classes(predictor):
    assert type(predictor.rf) is not type(predictor.xgb)
    assert "RandomForest" in type(predictor.rf).__name__
    assert "XGB" in type(predictor.xgb).__name__


def test_members_actually_disagree(predictor):
    """Two copies of one model pass every type check. This is the check that
    catches it."""
    rng = np.random.default_rng(0)
    centre = np.asarray(predictor.scaler.mean_, dtype=float)
    spread = np.asarray(predictor.scaler.scale_, dtype=float)
    probe = predictor.scaler.transform(centre + spread * rng.normal(size=(128, predictor.n_features)))
    p_rf, p_xgb = predictor._member_probabilities(probe)
    assert float(np.max(np.abs(p_rf - p_xgb))) > 1e-6


def test_blend_weights_are_canonical_and_normalised(predictor):
    assert np.isclose(predictor.w_rf + predictor.w_xgb, 1.0)
    assert 0.0 <= predictor.w_rf <= 1.0


def test_saved_schema_matches_the_code(predictor):
    from schema import FEATURES

    assert predictor.features == list(FEATURES)
    assert predictor.n_features == len(FEATURES)


def test_scaler_and_models_agree_on_width(predictor):
    assert predictor.scaler.n_features_in_ == predictor.n_features
    assert predictor.rf.n_features_in_ == predictor.n_features


def test_predictions_vary_across_the_scalers_own_input_distribution(predictor):
    """A scaler fitted on different data than the models collapses every real
    input into one leaf region and the output becomes a near constant band.
    That is exactly what happened, and it looked plausible in the results."""
    rng = np.random.default_rng(1)
    centre = np.asarray(predictor.scaler.mean_, dtype=float)
    spread = np.asarray(predictor.scaler.scale_, dtype=float)
    X = centre + spread * rng.normal(size=(256, predictor.n_features))
    reliability, failure = predictor.predict(X)
    assert np.ptp(failure) > 0.05, "predicted probability is effectively constant"
    assert np.allclose(reliability, 1.0 - failure)


def test_training_split_is_recorded_and_disjoint(predictor):
    split = predictor.schema["split"]
    assert not set(split["train_runs"]) & set(split["test_runs"])
    assert not set(split["val_runs"]) & set(split["test_runs"])


def test_reported_metrics_exist_and_are_sane():
    metrics_path = ROOT / "results" / "predictor_metrics.json"
    assert metrics_path.exists(), "train_predictor did not write metrics"
    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)
    for member in ("rf", "xgb", "ensemble"):
        r = m["test"][member]
        assert 0.0 <= r["roc_auc"] <= 1.0
        assert 0.0 <= r["precision"] <= 1.0
        assert 0.0 <= r["recall"] <= 1.0
    assert m["ensemble_weights"]["source"] in ("validation_auc", "equal_fallback_no_validation_data")
    assert m["ensemble_weights"]["source"] != "test_auc"
