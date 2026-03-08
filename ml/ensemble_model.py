import json
import os

import joblib
from sklearn.metrics import accuracy_score, classification_report

from .data_utils import DATA_PATH, RESULTS_DIR, train_test_split_dataset


def load_models():
    """Load trained base models from disk."""
    rf_path = os.path.join(RESULTS_DIR, "random_forest_model.pkl")
    nn_path = os.path.join(RESULTS_DIR, "neural_network_model.pkl")

    rf = joblib.load(rf_path)
    nn = joblib.load(nn_path)
    return rf, nn


def evaluate_ensemble():
    """
    Evaluate the RF + NN ensemble on the same held-out test set
    used during training.

    Returns
    -------
    acc : float
        Ensemble accuracy on the test set.
    report : dict
        Dict-form classification report (for saving as JSON).
    y_test : array-like
        True labels for the test set.
    ensemble_pred : array-like
        Ensemble predictions for the test set.
    """
    _, X_test, _, y_test = train_test_split_dataset(DATA_PATH)

    rf, nn = load_models()

    rf_pred = rf.predict(X_test)
    nn_pred = nn.predict(X_test)

    # Simple OR-based ensemble: stable if any base model predicts stable.
    ensemble_pred = ((rf_pred + nn_pred) >= 1).astype(int)

    acc = accuracy_score(y_test, ensemble_pred)
    report = classification_report(y_test, ensemble_pred, output_dict=True)

    return acc, report, y_test, ensemble_pred


def save_ensemble_metrics(acc: float, report: dict) -> str:
    """Persist ensemble metrics to the results directory and return its path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "metrics_ensemble.json")
    with open(path, "w") as f:
        json.dump({"accuracy": acc, "classification_report": report}, f, indent=2, default=float)
    return path


def main() -> None:
    acc, report, y_test, ensemble_pred = evaluate_ensemble()

    print("Ensemble Accuracy:", acc)
    print("\nEnsemble Classification Report")
    print(classification_report(y_test, ensemble_pred))

    metrics_path = save_ensemble_metrics(acc, report)
    print("\nEnsemble metrics saved to:", metrics_path)


if __name__ == "__main__":
    main()
