import json
import os
from typing import Tuple

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from .data_utils import (
    DATA_PATH,
    RESULTS_DIR,
    RANDOM_STATE,
    load_raw_dataset,
    train_test_split_dataset,
)


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest classifier for link stability prediction."""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    return rf


def train_neural_network(X_train, y_train) -> MLPClassifier:
    """Train a simple feed-forward neural network for link stability."""
    nn = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=RANDOM_STATE)
    nn.fit(X_train, y_train)
    return nn


def evaluate_model(model, X_test, y_test) -> Tuple[float, dict]:
    """Return accuracy and a dict-form classification report for a model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report


def save_model(model, filename: str) -> str:
    """Persist a trained model under the results directory and return its path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    joblib.dump(model, path)
    return path


def save_metrics(metrics: dict, filename: str) -> str:
    """Save metrics as JSON under the results directory and return the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    return path


def main() -> None:
    # Load dataset and perform a deterministic train/test split.
    data = load_raw_dataset(DATA_PATH)
    print("Dataset path:", DATA_PATH)
    print("Dataset size:", data.shape)

    X_train, X_test, y_train, y_test = train_test_split_dataset(DATA_PATH)

    # -----------------------------
    # Random Forest Model
    # -----------------------------
    rf = train_random_forest(X_train, y_train)
    rf_acc, rf_report = evaluate_model(rf, X_test, y_test)

    print("\nRandom Forest Accuracy:", rf_acc)

    rf_model_path = save_model(rf, "random_forest_model.pkl")
    rf_metrics_path = save_metrics(
        {"accuracy": rf_acc, "classification_report": rf_report},
        "metrics_random_forest.json",
    )

    # -----------------------------
    # Neural Network Model
    # -----------------------------
    nn = train_neural_network(X_train, y_train)
    nn_acc, nn_report = evaluate_model(nn, X_test, y_test)

    print("Neural Network Accuracy:", nn_acc)

    nn_model_path = save_model(nn, "neural_network_model.pkl")
    nn_metrics_path = save_metrics(
        {"accuracy": nn_acc, "classification_report": nn_report},
        "metrics_neural_network.json",
    )

    print("\nClassification Report (Random Forest)")
    # Pretty-print a human-readable version of the RF classification report.
    print(classification_report(y_test, rf.predict(X_test)))

    print("\nArtifacts saved:")
    print("  Random Forest model:", rf_model_path)
    print("  Random Forest metrics:", rf_metrics_path)
    print("  Neural Network model:", nn_model_path)
    print("  Neural Network metrics:", nn_metrics_path)


if __name__ == "__main__":
    main()
