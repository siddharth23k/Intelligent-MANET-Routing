"""
predict.py
----------
Loads the trained ensemble (Random Forest + XGBoost + Neural Network)
and exposes a single predict() method used by all routing scripts.

Changes from v1:
  - Added XGBoost model
  - Feature set expanded from 4 to 10
  - Loads StandardScaler (fitted on train set) for NN preprocessing
  - Loads ensemble weights from ensemble_weights.pkl (not hardcoded)
  - RSSI sentinel -1000 replaced with -90 at inference time (same as training)
  - Robust error messages if any model file is missing

Feature order (MUST match training.py exactly):
  0  neighbor_count
  1  x
  2  y
  3  time
  4  avg_rssi
  5  dist_to_center
  6  rssi_velocity
  7  neighbor_velocity
  8  pdr
  9  log_delay
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")

# ── Feature list — must stay in sync with training.py ────────────────────────
FEATURE_NAMES = [
    "neighbor_count",
    "x",
    "y",
    "time",
    "avg_rssi",
    "dist_to_center",
    "rssi_velocity",
    "neighbor_velocity",
    "pdr",
    "log_delay",
]

RSSI_SENTINEL     = -1000.0
RSSI_SENTINEL_SUB = -90.0      # replacement used during training


class LinkFailurePredictor:
    """
    Loads all three models + scaler + weights and provides a single
    predict(X) interface that returns (reliability, failure_probability).

    X must be a 2D array of shape (n_samples, 10) with columns in the
    exact order defined by FEATURE_NAMES above.

    If you are calling predict() from routing_from_dataset.py, the
    DatasetRouter.build_graph() method constructs X in this order
    automatically — you do not need to do anything special.
    """

    def __init__(self):
        print("Loading ensemble models...")

        project_root = Path(__file__).resolve().parent.parent
        models_dir   = project_root / "models"

        # ── File paths ────────────────────────────────────────────────────────
        paths = {
            "rf"      : models_dir / "random_forest.pkl",
            "xgb"     : models_dir / "xgboost_model.pkl",
            "nn"      : models_dir / "neural_network.keras",
            "scaler"  : models_dir / "scaler.pkl",
            "weights" : models_dir / "ensemble_weights.pkl",
        }

        # ── Validate all files exist before loading ───────────────────────────
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}\n"
                f"Run notebooks/training.py first to generate all model files."
            )

        # ── Load models ───────────────────────────────────────────────────────
        self.rf              = joblib.load(paths["rf"])
        self.xgb             = joblib.load(paths["xgb"])
        self.nn              = keras.models.load_model(paths["nn"], compile=False)
        self.scaler          = joblib.load(paths["scaler"])
        self.ensemble_weights = joblib.load(paths["weights"])

        print(f"  Random Forest    : loaded")
        print(f"  XGBoost          : loaded")
        print(f"  Neural Network   : loaded")
        print(f"  Scaler           : loaded")
        print(f"  Ensemble weights : RF={self.ensemble_weights['rf']}, "
              f"XGB={self.ensemble_weights['xgb']}, "
              f"NN={self.ensemble_weights['nn']}")
        print("All models loaded successfully.\n")

    def _preprocess(self, X):
        """
        Converts input to DataFrame, replaces RSSI sentinel,
        and returns both raw array (for RF/XGB) and scaled array (for NN).
        """
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)

        # Replace sentinel value — same substitution used during training
        X_df["avg_rssi"] = X_df["avg_rssi"].replace(RSSI_SENTINEL, RSSI_SENTINEL_SUB)

        X_raw    = X_df.values
        X_scaled = self.scaler.transform(X_raw)

        return X_raw, X_scaled

    def predict(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, 10)
            Feature matrix. Columns must be in FEATURE_NAMES order.

        Returns
        -------
        reliability : np.ndarray, shape (n_samples,)
            Per-sample reliability score = 1 - failure_probability.
            Range [0, 1]. Higher = more stable link.

        failure_prob : np.ndarray, shape (n_samples,)
            Ensemble predicted probability of link failure.
            Range [0, 1]. Higher = more likely to fail.
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_raw, X_scaled = self._preprocess(X)

        # ── Individual model probabilities ────────────────────────────────────
        rf_probs  = self.rf.predict_proba(X_raw)[:, 1]
        xgb_probs = self.xgb.predict_proba(X_raw)[:, 1]
        nn_probs  = self.nn(X_scaled, training=False).numpy().flatten()

        # ── Weighted ensemble ─────────────────────────────────────────────────
        w_rf  = self.ensemble_weights["rf"]
        w_xgb = self.ensemble_weights["xgb"]
        w_nn  = self.ensemble_weights["nn"]

        failure_prob = w_rf * rf_probs + w_xgb * xgb_probs + w_nn * nn_probs
        reliability  = 1.0 - failure_prob

        return reliability, failure_prob

    def predict_single(self, neighbor_count, x, y, time, avg_rssi,
                       dist_to_center, rssi_velocity, neighbor_velocity,
                       pdr, log_delay):
        """
        Convenience wrapper for predicting a single link with named arguments.
        Useful for debugging and interactive testing.
        """
        X = np.array([[
            neighbor_count, x, y, time, avg_rssi,
            dist_to_center, rssi_velocity, neighbor_velocity,
            pdr, log_delay
        ]])
        reliability, failure_prob = self.predict(X)
        return float(reliability[0]), float(failure_prob[0])


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = LinkFailurePredictor()

    print("=" * 50)
    print("Standalone prediction test")
    print("=" * 50)

    # Three test cases:
    # 1 — well-connected node, good RSSI, centre of area  → should be low failure prob
    # 2 — isolated node (sentinel RSSI, 0 neighbours)     → should be high failure prob
    # 3 — border node, dropping RSSI, losing neighbours   → should be medium-high

    test_cases = np.array([
        # nc     x      y    time  rssi  d2c   rssi_v  nc_v  pdr   logd
        [  5,  500,   500,    10,  -55,    0,     0.0,   0,  1.0,  0.0],  # healthy
        [  0,   50,    50,    40, -1000, 660,   -15.0,  -3,  0.0,  0.0],  # failing
        [  2,  900,   900,    30,  -72,  566,    -8.0,  -1,  0.7,  2.5],  # degrading
    ])

    labels = ["Healthy node", "Isolated node", "Degrading node"]

    reliability, failure_prob = predictor.predict(test_cases)

    print(f"\n{'Case':<18} {'Fail Prob':>10} {'Reliability':>12}")
    print("-" * 42)
    for label, fp, r in zip(labels, failure_prob, reliability):
        print(f"{label:<18} {fp:>10.3f} {r:>12.3f}")

    print("\nExpected: Healthy < Degrading < Isolated in failure probability.")
