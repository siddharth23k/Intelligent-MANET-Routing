import os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb

# Import TensorFlow for the Neural Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# Feature list matches feature_engineering.py
FEATURES = [
    "neighbor_count", "x", "y", "time", "avg_rssi",
    "dist_to_center", "rssi_velocity", "neighbor_velocity",
    "pdr", "log_delay",
    "rssi_trend_3", "neighbor_trend_3", "rssi_std_5", "neighbor_std_5"
]
TARGET = "link_failure"
TEST_RUNS = [25, 26, 27, 28, 29, 30]

# def build_nn(input_dim):
#     model = keras.Sequential([
#         layers.Input(shape=(input_dim,)),
#         layers.Dense(64, activation="relu"),
#         layers.BatchNormalization(),
#         layers.Dropout(0.2),
#         layers.Dense(32, activation="relu"),
#         layers.Dense(1, activation="sigmoid")
#     ])
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
#     return model

def train():
    print("Step 1: Loading Data...")
    df = pd.read_csv("dataset/manet_featured_dataset.csv")
    df["avg_rssi"] = df["avg_rssi"].replace(-1000.0, -95.0)

    train_df = df[~df["run_id"].isin(TEST_RUNS)]
    test_df = df[df["run_id"].isin(TEST_RUNS)]

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Random Forest
    print("Step 2: Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1])

    # 2. XGBoost
    print("Step 3: Training XGBoost...")
    xgb_mod = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    xgb_mod.fit(X_train_scaled, y_train)
    xgb_auc = roc_auc_score(y_test, xgb_mod.predict_proba(X_test_scaled)[:, 1])

    # 3. Neural Network
    # print("Step 4: Training Neural Network...")
    # nn = build_nn(len(FEATURES))
    # nn.fit(X_train_scaled, y_train, epochs=5, batch_size=10000, verbose=1)
    # nn_probs = nn.predict(X_test_scaled, batch_size = 10000).flatten()
    # nn_auc = roc_auc_score(y_test, nn_probs)

    print("-" * 30)
    print(f"RF AUC: {rf_auc:.4f}")
    print(f"XGB AUC: {xgb_auc:.4f}")
    # print(f"NN AUC: {nn_auc:.4f}")
    print("-" * 30)

    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf, "models/random_forest.pkl")
    joblib.dump(xgb_mod, "models/xgboost_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    # nn.save("models/neural_network.keras")
    
    # Save ensemble weights (33% each)
    joblib.dump({"rf": 0.4, "xgb": 0.6}, "models/ensemble_weights.pkl")
    print("All models saved successfully.")

if __name__ == "__main__":
    train()