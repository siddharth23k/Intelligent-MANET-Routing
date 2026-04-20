import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

class LinkFailurePredictor:
    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent.parent
        models_dir = project_root / "results/models"

        self.rf = joblib.load(models_dir / "random_forest.pkl")
        self.xgb = joblib.load(models_dir / "xgboost_model.pkl")
        self.scaler = joblib.load(models_dir / "scaler.pkl")
        self.ensemble_weights = joblib.load(models_dir / "ensemble_weights.pkl")

        
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        rf_p = self.rf.predict_proba(X_scaled)[:, 1]
        xgb_p = self.xgb.predict_proba(X_scaled)[:, 1]

        w = self.ensemble_weights
        if 'rf' in w and 'xgb' in w:
            rf_weight, xgb_weight = w['rf'], w['xgb']
        elif 'rf_weight' in w and 'xgb_weight' in w:
            rf_weight, xgb_weight = w['rf_weight'], w['xgb_weight']
        else:
            rf_weight, xgb_weight = 0.6, 0.4
        failure_prob = (rf_weight * rf_p) + (xgb_weight * xgb_p)
        
        reliability = 1.0 - failure_prob
        return reliability, failure_prob