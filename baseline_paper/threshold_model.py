import numpy as np


class AdaptiveThresholdModel:
    """
    Lightweight paper-inspired adaptive threshold estimator.
    """

    @staticmethod
    def predict_threshold(feature_row: dict) -> float:
        # Inputs are expected normalized or near-normalized.
        lq = float(feature_row.get("LQ_mean", 0.5))
        rssi = float(feature_row.get("RSSI_norm", 0.5))
        ls = float(feature_row.get("LS_norm", 0.5))
        let = float(feature_row.get("LET_norm", 0.5))
        ll = float(feature_row.get("LL_d_norm", 0.5))
        nd = float(feature_row.get("ND_norm", 0.5))
        z = 1.8 * lq + 1.2 * rssi + 1.0 * ls + 0.8 * let - 1.0 * ll - 0.8 * (1.0 - nd)
        return float(np.clip(0.35 + 0.3 * (1.0 / (1.0 + np.exp(-z))), 0.2, 0.8))
