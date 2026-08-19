"""Inference wrapper for the RF + XGBoost link failure predictor.

Everything defensive in this file exists because of a specific failure. The
model artifacts once drifted so far from the code that the two ensemble members
were the same serialised object and the scaler had been fitted on different data
than the models, and none of it raised. Probabilities came out in a narrow band,
routing degenerated towards hop count, and the results looked plausible.

So: load, then prove the artifacts are internally consistent, and refuse to run
if they are not. A loud failure at import time is cheap. A silent wrong answer
that ends up in a report is not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from schema import FEATURES, SchemaError, assert_matrix_shape  # noqa: E402

MODELS_DIR = ROOT / "results" / "models"
REQUIRED_ARTIFACTS = (
    "random_forest.pkl",
    "xgboost_model.pkl",
    "scaler.pkl",
    "ensemble_weights.pkl",
    "predictor_schema.json",
)


class ArtifactError(RuntimeError):
    """The saved models are missing, mismatched, or internally inconsistent."""


class LinkFailurePredictor:
    def __init__(self, models_dir: str | Path = MODELS_DIR, verify: bool = True):
        self.models_dir = Path(models_dir)

        missing = [a for a in REQUIRED_ARTIFACTS if not (self.models_dir / a).exists()]
        if missing:
            raise ArtifactError(
                f"missing model artifacts {missing} in {self.models_dir}. "
                "Run: python pipeline/train_predictor.py"
            )

        self.rf = joblib.load(self.models_dir / "random_forest.pkl")
        self.xgb = joblib.load(self.models_dir / "xgboost_model.pkl")
        self.scaler = joblib.load(self.models_dir / "scaler.pkl")
        weights = joblib.load(self.models_dir / "ensemble_weights.pkl")
        with open(self.models_dir / "predictor_schema.json", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.features: list = list(self.schema.get("features", FEATURES))
        self.n_features = int(self.schema.get("n_features", len(self.features)))

        # Canonical weight keys only. The previous version guessed between three
        # different key spellings and fell back to hardcoded constants, which is
        # how a weights file that no training run had produced went unnoticed.
        if not isinstance(weights, dict) or {"rf", "xgb"} - set(weights):
            raise ArtifactError(
                f"ensemble_weights.pkl must be a dict with keys 'rf' and 'xgb', got "
                f"{sorted(weights) if isinstance(weights, dict) else type(weights).__name__}. "
                "Retrain with pipeline/train_predictor.py."
            )
        self.w_rf = float(weights["rf"])
        self.w_xgb = float(weights["xgb"])
        total = self.w_rf + self.w_xgb
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ArtifactError(f"ensemble weights must sum to 1, got {total:.6f}")

        if verify:
            self._verify()

    # -- integrity ---------------------------------------------------------
    def _verify(self) -> None:
        if self.features != list(FEATURES):
            raise SchemaError(
                "saved feature schema does not match methods/common/schema.py.\n"
                f"  saved: {self.features}\n  code : {list(FEATURES)}\n"
                "Retrain so the artifacts and the code agree."
            )

        n_scaler = int(getattr(self.scaler, "n_features_in_", 0) or 0)
        if n_scaler != self.n_features:
            raise ArtifactError(
                f"scaler expects {n_scaler} features, schema declares {self.n_features}"
            )
        for name, model in (("random forest", self.rf), ("xgboost", self.xgb)):
            n_model = int(getattr(model, "n_features_in_", 0) or 0)
            if n_model and n_model != self.n_features:
                raise ArtifactError(
                    f"{name} expects {n_model} features, schema declares {self.n_features}"
                )

        if type(self.rf) is type(self.xgb):
            raise ArtifactError(
                f"both ensemble members deserialise to {type(self.rf).__name__}. "
                "The two artifact files are the same model, so the blend is a no op. "
                "Retrain with pipeline/train_predictor.py."
            )
        if self.rf is self.xgb:
            raise ArtifactError("both ensemble members are the same object")

        # Probe with data drawn from the scaler's own fitted distribution, so a
        # scaler that was fitted on different data than the models shows up here
        # rather than as a narrow, useless probability band at routing time.
        rng = np.random.default_rng(0)
        centre = np.asarray(getattr(self.scaler, "mean_", np.zeros(self.n_features)), dtype=float)
        spread = np.asarray(getattr(self.scaler, "scale_", np.ones(self.n_features)), dtype=float)
        probe = centre + spread * rng.normal(size=(256, self.n_features))

        p_rf, p_xgb = self._member_probabilities(probe)
        if float(np.max(np.abs(p_rf - p_xgb))) < 1e-9:
            raise ArtifactError(
                "ensemble members return identical probabilities on every probe row; "
                "the blend would be a no op"
            )

        blended = self.w_rf * p_rf + self.w_xgb * p_xgb
        if float(np.ptp(blended)) < 1e-6:
            raise ArtifactError(
                "predicted probability is effectively constant across the scaler's own "
                "input distribution. The scaler and the models were probably fitted on "
                "different data. Retrain with pipeline/train_predictor.py."
            )

    # -- inference ---------------------------------------------------------
    def _member_probabilities(self, X_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.rf.predict_proba(X_scaled)[:, 1],
            self.xgb.predict_proba(X_scaled)[:, 1],
        )

    def predict(self, X: Sequence[Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (reliability, failure_probability) for each row.

        Rows must be in the canonical feature order declared in
        methods/common/schema.py.
        """
        X = assert_matrix_shape(np.asarray(X, dtype=float), self.n_features, "LinkFailurePredictor.predict")
        X_scaled = self.scaler.transform(X)
        p_rf, p_xgb = self._member_probabilities(X_scaled)
        failure_prob = self.w_rf * p_rf + self.w_xgb * p_xgb
        return 1.0 - failure_prob, failure_prob

    def predict_frame(self, frame) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper that enforces the column contract."""
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise SchemaError(f"frame is missing feature columns {missing}")
        return self.predict(frame[self.features].to_numpy(dtype=float))

    def describe(self) -> dict:
        return {
            "models_dir": str(self.models_dir),
            "rf": type(self.rf).__name__,
            "xgb": type(self.xgb).__name__,
            "n_features": self.n_features,
            "weights": {"rf": self.w_rf, "xgb": self.w_xgb},
            "trained_split": self.schema.get("split"),
        }
