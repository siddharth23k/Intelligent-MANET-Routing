import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Suppress warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

class LinkFailurePredictor:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        models_dir = project_root / "models"

        # Load all components
        self.rf = joblib.load(models_dir / "random_forest.pkl")
        self.xgb = joblib.load(models_dir / "xgboost_model.pkl")
        self.scaler = joblib.load(models_dir / "scaler.pkl")
        # self.nn = keras.models.load_model(models_dir / "neural_network.keras")
        self.ensemble_weights = joblib.load(models_dir / "ensemble_weights.pkl")

        print("Ensemble Models Loaded: Random Forest and XGBoost.")

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # 1. Scale
        X_scaled = self.scaler.transform(X)

        # 2. Get probabilities from each model
        rf_p = self.rf.predict_proba(X_scaled)[:, 1]
        xgb_p = self.xgb.predict_proba(X_scaled)[:, 1]
        # nn_p = self.nn.predict(X_scaled, verbose=0).flatten()

        # 3. Weighted Ensemble
        w = self.ensemble_weights
        failure_prob = (w['rf'] * rf_p) + (w['xgb'] * xgb_p)
        
        reliability = 1.0 - failure_prob
        return reliability, failure_prob