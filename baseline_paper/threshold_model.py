import numpy as np


class AdaptiveThresholdModel:
    """
    Closed-form adaptive threshold used as a *teacher* target when training the
    SFRNNR threshold head (see experiments/train_sfrnnr_paper.py). Final routing
    uses thresholds produced by the trained network in paper_lfp_dataset.csv.
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

    @staticmethod
    def predict_threshold_batch(
        lq: np.ndarray,
        rssi: np.ndarray,
        ls: np.ndarray,
        let: np.ndarray,
        ll_d: np.ndarray,
        nd: np.ndarray,
    ) -> np.ndarray:
        """Vectorized teacher thresholds (same formula as predict_threshold)."""
        z = (
            1.8 * lq
            + 1.2 * rssi
            + 1.0 * ls
            + 0.8 * let
            - 1.0 * ll_d
            - 0.8 * (1.0 - nd)
        )
        s = 1.0 / (1.0 + np.exp(-z))
        return np.clip(0.35 + 0.3 * s, 0.2, 0.8).astype(np.float32)
