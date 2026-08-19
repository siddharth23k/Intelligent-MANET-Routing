"""Stage 4: train our RF + XGBoost link failure predictor.

Two correctness properties this script enforces.

1. Blend weights come from a validation split carved out of the training runs.
   Choosing them from test set AUC is model selection on the test set.
2. The two ensemble members are verified to be genuinely different models before
   anything is written, and again after reloading from disk. Two copies of one
   model make the blend w*p + (1-w)*p, which evaluates to p and raises nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "config"))
from bootstrap import setup_paths  # noqa: E402

ROOT = setup_paths()

from config_loader import get_config  # noqa: E402
from label_utils import add_link_failure_labels, drop_label_aux_columns, label_diagnostics  # noqa: E402
from schema import FEATURES, TARGET, assert_columns  # noqa: E402
from splits import assert_no_row_overlap, make_run_split, split_frame  # noqa: E402

CFG = get_config()
MODELS_DIR = ROOT / "results" / "models"
DEFAULT_DATASET = ROOT / "data" / "processed" / "paper_featured_dataset.csv"

RF_PATH = MODELS_DIR / "random_forest.pkl"
XGB_PATH = MODELS_DIR / "xgboost_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
WEIGHTS_PATH = MODELS_DIR / "ensemble_weights.pkl"
SCHEMA_PATH = MODELS_DIR / "predictor_schema.json"
METRICS_PATH = ROOT / "results" / "predictor_metrics.json"


class EnsembleIntegrityError(RuntimeError):
    """The two ensemble members are not two distinct, disagreeing models."""


def _classification_report(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "roc_auc": float("nan") if single_class else float(roc_auc_score(y_true, proba)),
        "average_precision": float("nan") if single_class else float(average_precision_score(y_true, proba)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)) if len(y_true) else float("nan"),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": float(threshold),
    }


def verify_distinct_members(rf, xgb_model, X_probe: np.ndarray) -> dict:
    """Check the class, the identity, and that predictions actually differ.

    The last check is the one that catches a duplicated artifact, because two
    copies of one forest pass any type check you care to write.
    """
    if type(rf) is type(xgb_model):
        raise EnsembleIntegrityError(
            f"both members are {type(rf).__name__}; expected a RandomForestClassifier "
            "and an XGBClassifier"
        )
    if rf is xgb_model:
        raise EnsembleIntegrityError("both ensemble members are the same object")

    p_rf = rf.predict_proba(X_probe)[:, 1]
    p_xgb = xgb_model.predict_proba(X_probe)[:, 1]
    max_diff = float(np.max(np.abs(p_rf - p_xgb))) if len(X_probe) else 0.0
    if max_diff < 1e-9:
        raise EnsembleIntegrityError(
            "members produce identical probabilities on every probe row; the blend is a no op"
        )
    return {
        "rf_class": type(rf).__name__,
        "xgb_class": type(xgb_model).__name__,
        "max_abs_probability_difference": max_diff,
        "mean_abs_probability_difference": float(np.mean(np.abs(p_rf - p_xgb))),
    }


def train(
    dataset_path: str | Path = DEFAULT_DATASET,
    smoke: bool = False,
    seed: int | None = None,
    test_run_count: int | None = None,
    val_run_count: int | None = None,
) -> dict:
    seed = CFG.random_seed if seed is None else seed
    training, smoke_cfg = CFG.training, CFG.smoke
    if test_run_count is None:
        test_run_count = smoke_cfg["test_run_count"] if smoke else CFG.test_run_count
    if val_run_count is None:
        val_run_count = smoke_cfg["val_run_count"] if smoke else CFG.val_run_count

    df = pd.read_csv(dataset_path)
    assert_columns(df, FEATURES + ["run_id", "node_id", "time"], "train_predictor")
    df = drop_label_aux_columns(add_link_failure_labels(df))

    run_ids = sorted(df["run_id"].unique().astype(int).tolist())
    split = make_run_split(
        run_ids, seed=seed, test_run_count=test_run_count, val_run_count=val_run_count
    )
    train_df, val_df, test_df = split_frame(df, split)
    assert_no_row_overlap(train_df, test_df)
    if len(val_df):
        assert_no_row_overlap(val_df, test_df)

    X_train = train_df[FEATURES].to_numpy(dtype=float)
    y_train = train_df[TARGET].to_numpy(dtype=int)
    X_val = val_df[FEATURES].to_numpy(dtype=float) if len(val_df) else X_train[:0]
    y_val = val_df[TARGET].to_numpy(dtype=int) if len(val_df) else y_train[:0]
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    y_test = test_df[TARGET].to_numpy(dtype=int)

    # Fitted on training rows only, applied unchanged to val and test.
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=int(smoke_cfg["rf_estimators"] if smoke else training["rf_estimators"]),
        max_depth=int(smoke_cfg["rf_max_depth"] if smoke else training["rf_max_depth"]),
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(Xtr, y_train)

    positives = max(1, int((y_train == 1).sum()))
    negatives = max(1, int((y_train == 0).sum()))
    use_early_stopping = bool(len(X_val)) and not smoke
    xgb_model = XGBClassifier(
        n_estimators=int(smoke_cfg["xgb_estimators"] if smoke else training["xgb_estimators"]),
        max_depth=int(smoke_cfg["xgb_max_depth"] if smoke else training["xgb_max_depth"]),
        learning_rate=float(training["xgb_learning_rate"]),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=negatives / positives,
        random_state=seed,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=(
            int(training["xgb_early_stopping_rounds"]) if use_early_stopping else None
        ),
    )
    if use_early_stopping:
        xgb_model.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
    else:
        xgb_model.fit(Xtr, y_train, verbose=False)

    probe = Xte[:512] if len(Xte) else Xtr[:512]
    integrity = verify_distinct_members(rf, xgb_model, probe)

    def _auc(model, X, y) -> float:
        if len(y) == 0 or len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    rf_val_auc, xgb_val_auc = _auc(rf, Xva, y_val), _auc(xgb_model, Xva, y_val)
    if np.isnan(rf_val_auc) or np.isnan(xgb_val_auc) or (rf_val_auc + xgb_val_auc) <= 0:
        w_rf, w_xgb, weight_source = 0.5, 0.5, "equal_fallback_no_validation_data"
    else:
        w_rf = float(rf_val_auc / (rf_val_auc + xgb_val_auc))
        w_xgb = float(1.0 - w_rf)
        weight_source = "validation_auc"

    blended_test = w_rf * rf.predict_proba(Xte)[:, 1] + w_xgb * xgb_model.predict_proba(Xte)[:, 1]

    metrics = {
        "dataset": str(dataset_path),
        "smoke": bool(smoke),
        "seed": int(seed),
        "split": split.as_dict(),
        "rows": {"train": int(len(y_train)), "val": int(len(y_val)), "test": int(len(y_test))},
        "features": FEATURES,
        "label_diagnostics": label_diagnostics(df),
        "ensemble_weights": {"rf": w_rf, "xgb": w_xgb, "source": weight_source},
        "validation_auc": {"rf": rf_val_auc, "xgb": xgb_val_auc},
        "integrity": integrity,
        "test": {
            "rf": _classification_report(y_test, rf.predict_proba(Xte)[:, 1]),
            "xgb": _classification_report(y_test, xgb_model.predict_proba(Xte)[:, 1]),
            "ensemble": _classification_report(y_test, blended_test),
        },
        "feature_importance_rf": {
            name: float(value)
            for name, value in sorted(
                zip(FEATURES, rf.feature_importances_), key=lambda kv: -kv[1]
            )
        },
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, RF_PATH)
    joblib.dump(xgb_model, XGB_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({"rf": w_rf, "xgb": w_xgb}, WEIGHTS_PATH)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "features": FEATURES,
                "n_features": len(FEATURES),
                "target": TARGET,
                "scaler": "StandardScaler fitted on training runs only",
                "split": split.as_dict(),
                "seed": int(seed),
                "artifacts": {
                    "rf": RF_PATH.name,
                    "xgb": XGB_PATH.name,
                    "scaler": SCALER_PATH.name,
                    "weights": WEIGHTS_PATH.name,
                },
            },
            handle,
            indent=2,
        )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Re-read from disk and re-verify, so a bad serialisation cannot slip through.
    verify_distinct_members(joblib.load(RF_PATH), joblib.load(XGB_PATH), probe[:256])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RF + XGBoost link failure predictor.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--smoke", action="store_true", help="tiny, fast configuration")
    parser.add_argument("--seed", type=int, default=CFG.random_seed)
    parser.add_argument("--test-run-count", type=int, default=None)
    parser.add_argument("--val-run-count", type=int, default=None)
    args = parser.parse_args()

    metrics = train(
        dataset_path=args.dataset,
        smoke=args.smoke,
        seed=args.seed,
        test_run_count=args.test_run_count,
        val_run_count=args.val_run_count,
    )
    split = metrics["split"]
    integrity = metrics["integrity"]
    print(f"[train_predictor] split -> train={len(split['train_runs'])} runs, "
          f"val={len(split['val_runs'])} runs, test={len(split['test_runs'])} runs")
    print(f"[train_predictor] members verified distinct ({integrity['rf_class']} vs "
          f"{integrity['xgb_class']}, max prob diff "
          f"{integrity['max_abs_probability_difference']:.4f})")
    print(f"[train_predictor] blend weights {metrics['ensemble_weights']}")
    for name in ("rf", "xgb", "ensemble"):
        report = metrics["test"][name]
        print(f"[train_predictor] test {name:9s} auc={report['roc_auc']:.4f} "
              f"ap={report['average_precision']:.4f} f1={report['f1']:.4f} "
              f"brier={report['brier']:.4f}")
    print(f"[train_predictor] metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    main()
