import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Common paths and configuration used across ML scripts and notebooks.
BASE_DATA_DIR = "dataset"
DATA_FILENAME = "link_dataset.csv"
DATA_PATH = os.path.join(BASE_DATA_DIR, DATA_FILENAME)

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

FEATURE_COLUMNS = ["distance", "time"]
TARGET_COLUMN = "link_status"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_raw_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the full link dataset CSV as a pandas DataFrame."""
    return pd.read_csv(path)


def get_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and label vector y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def load_features_and_labels(
    path: str = DATA_PATH,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience helper to load features and labels directly from CSV."""
    df = load_raw_dataset(path)
    return get_features_and_labels(df)


def train_test_split_dataset(
    path: str = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Load the dataset from disk and return a deterministic train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_raw_dataset(path)
    X, y = get_features_and_labels(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

