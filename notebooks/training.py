# ═══════════════════════════════════════════════════════════════════════════════
# training.py
# -----------
# Trains an ensemble of Random Forest + XGBoost + Neural Network on the
# featured MANET dataset to predict link failure.
#
# Key design decisions:
#   1. Train/test split by run_id — never by random row shuffle.
#      Rows from the same run are temporally correlated. A random split
#      would leak future timesteps into training, inflating AUC artificially.
#      We train on runs 1-24, test on runs 25-30.
#
#   2. Feature set — 10 features:
#      Original : neighbor_count, x, y, time, avg_rssi
#      Engineered: dist_to_center, rssi_velocity, neighbor_velocity, pdr, log_delay
#
#   3. Ensemble — weighted average of three model probabilities:
#      RF (0.3) + XGBoost (0.4) + NN (0.3)
#      XGBoost gets highest weight — it is typically the strongest on tabular data.
#
#   4. Evaluation — AUC, F1, precision, recall, confusion matrix,
#      ROC curve, feature importance (RF + XGBoost), NN training curves.
#
# Outputs (saved to models/):
#   random_forest.pkl
#   xgboost_model.pkl
#   neural_network.keras
#   scaler.pkl              ← StandardScaler fitted on train set only
#   ensemble_weights.pkl    ← weights dict for predict.py
# ═══════════════════════════════════════════════════════════════════════════════

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay, f1_score
)
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/manet_featured_dataset.csv"
MODELS_DIR   = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Feature config ────────────────────────────────────────────────────────────
FEATURES = [
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
TARGET = "link_failure"

# ── Ensemble weights (must sum to 1.0) ────────────────────────────────────────
ENSEMBLE_WEIGHTS = {"rf": 0.3, "xgb": 0.4, "nn": 0.3}

# ── Train/test split config ───────────────────────────────────────────────────
# Run IDs are integers. We hold out the last 6 runs for testing.
# Adjust these if your run IDs differ.
TEST_RUNS = [25, 26, 27, 28, 29, 30]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & Split
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 — Loading dataset")
print("=" * 65)

df = pd.read_csv(DATASET_PATH)
print(f"  Total rows    : {len(df)}")
print(f"  Total features: {len(FEATURES)}")
print(f"  Run IDs found : {sorted(df['run_id'].unique())}")

# Replace RSSI sentinel with a large negative value that is a real number
# (not -1000 which can confuse tree splits and NN gradients).
# -90 dBm is well below any real WiFi signal, so semantics are preserved.
df["avg_rssi"] = df["avg_rssi"].replace(-1000.0, -90.0)

# Split by run_id — NOT random shuffle
train_df = df[~df["run_id"].isin(TEST_RUNS)].copy()
test_df  = df[ df["run_id"].isin(TEST_RUNS)].copy()

print(f"\n  Train runs : {sorted(train_df['run_id'].unique())}")
print(f"  Test runs  : {sorted(test_df['run_id'].unique())}")
print(f"  Train rows : {len(train_df)}")
print(f"  Test rows  : {len(test_df)}")

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].values
X_test  = test_df[FEATURES].values
y_test  = test_df[TARGET].values

# Label distribution
train_pos = y_train.mean()
test_pos  = y_test.mean()
print(f"\n  Train failure rate : {train_pos:.3f}")
print(f"  Test  failure rate : {test_pos:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Scaling (for Neural Network only)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 — Fitting StandardScaler on train set")
print("=" * 65)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit ONLY on train
X_test_scaled  = scaler.transform(X_test)         # transform test with same params

joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("  Scaler fitted and saved.")
print("  Mean per feature :", np.round(scaler.mean_, 3))
print("  Std  per feature :", np.round(scaler.scale_, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Random Forest
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 — Training Random Forest")
print("=" * 65)

rf = RandomForestClassifier(
    n_estimators=200,       # 200 trees — better than default 100, diminishing returns past ~300
    max_depth=15,           # prevents overfitting on our ~50k row dataset
    min_samples_split=10,   # a node must have 10+ samples to be split
    min_samples_leaf=5,     # each leaf must have 5+ samples
    max_features="sqrt",    # sqrt(10) ≈ 3 features per split — standard for classification
    class_weight="balanced",# compensates for class imbalance automatically
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf.fit(X_train, y_train)

rf_train_probs = rf.predict_proba(X_train)[:, 1]
rf_test_probs  = rf.predict_proba(X_test)[:, 1]

rf_train_auc = roc_auc_score(y_train, rf_train_probs)
rf_test_auc  = roc_auc_score(y_test,  rf_test_probs)

print(f"  Train AUC : {rf_train_auc:.4f}")
print(f"  Test  AUC : {rf_test_auc:.4f}")

joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
print("  Random Forest saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. XGBoost
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 — Training XGBoost")
print("=" * 65)

# scale_pos_weight handles class imbalance for XGBoost:
# ratio of negative class to positive class in training set
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"  Neg/Pos ratio (scale_pos_weight): {scale_pos_weight:.2f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,         # slow learning rate + more trees = better generalisation
    subsample=0.8,              # use 80% of rows per tree (reduces overfitting)
    colsample_bytree=0.8,       # use 80% of features per tree
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Early stopping on a validation slice from train set
val_size = int(0.1 * len(X_train))
X_xgb_val, y_xgb_val = X_train[-val_size:], y_train[-val_size:]
X_xgb_tr,  y_xgb_tr  = X_train[:-val_size], y_train[:-val_size]

xgb_model.fit(
    X_xgb_tr, y_xgb_tr,
    eval_set=[(X_xgb_val, y_xgb_val)],
    verbose=False
)

xgb_train_probs = xgb_model.predict_proba(X_train)[:, 1]
xgb_test_probs  = xgb_model.predict_proba(X_test)[:, 1]

xgb_train_auc = roc_auc_score(y_train, xgb_train_probs)
xgb_test_auc  = roc_auc_score(y_test,  xgb_test_probs)

print(f"  Train AUC : {xgb_train_auc:.4f}")
print(f"  Test  AUC : {xgb_test_auc:.4f}")

joblib.dump(xgb_model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))
print("  XGBoost saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Neural Network
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 — Training Neural Network")
print("=" * 65)

def build_nn(input_dim):
    """
    3-layer MLP with BatchNorm and Dropout.
    BatchNorm: normalises activations between layers → faster, more stable training
    Dropout: randomly zeros 30% of neurons during training → prevents overfitting
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid")   # output: P(link_failure)
    ])
    return model

nn = build_nn(input_dim=len(FEATURES))
nn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["AUC"]
)
nn.summary()

# Callbacks
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_auc",
    patience=10,
    restore_best_weights=True,
    mode="max"
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    mode="max",
    verbose=0
)

# Class weights for NN (same imbalance correction as XGBoost)
class_weight_dict = {0: 1.0, 1: scale_pos_weight}

history = nn.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=512,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

nn_train_probs = nn.predict(X_train_scaled, verbose=0).flatten()
nn_test_probs  = nn.predict(X_test_scaled,  verbose=0).flatten()

nn_train_auc = roc_auc_score(y_train, nn_train_probs)
nn_test_auc  = roc_auc_score(y_test,  nn_test_probs)

print(f"\n  Train AUC : {nn_train_auc:.4f}")
print(f"  Test  AUC : {nn_test_auc:.4f}")

nn.save(os.path.join(MODELS_DIR, "neural_network.keras"))
print("  Neural Network saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Ensemble
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 — Building Ensemble")
print("=" * 65)

w_rf  = ENSEMBLE_WEIGHTS["rf"]
w_xgb = ENSEMBLE_WEIGHTS["xgb"]
w_nn  = ENSEMBLE_WEIGHTS["nn"]

ensemble_train_probs = (
    w_rf  * rf_train_probs  +
    w_xgb * xgb_train_probs +
    w_nn  * nn_train_probs
)
ensemble_test_probs = (
    w_rf  * rf_test_probs  +
    w_xgb * xgb_test_probs +
    w_nn  * nn_test_probs
)

ensemble_train_auc = roc_auc_score(y_train, ensemble_train_probs)
ensemble_test_auc  = roc_auc_score(y_test,  ensemble_test_probs)

# Threshold at 0.5 for classification metrics
ensemble_test_preds = (ensemble_test_probs >= 0.5).astype(int)

print(f"  Ensemble weights: RF={w_rf}, XGB={w_xgb}, NN={w_nn}")
print(f"\n  Individual model Test AUCs:")
print(f"    Random Forest : {rf_test_auc:.4f}")
print(f"    XGBoost       : {xgb_test_auc:.4f}")
print(f"    Neural Network: {nn_test_auc:.4f}")
print(f"    ── Ensemble ──: {ensemble_test_auc:.4f}")

print(f"\n  Classification Report (threshold=0.5):")
print(classification_report(y_test, ensemble_test_preds,
                             target_names=["Stable (0)", "Failure (1)"]))

joblib.dump(ENSEMBLE_WEIGHTS, os.path.join(MODELS_DIR, "ensemble_weights.pkl"))
print("  Ensemble weights saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Visualisations
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 — Generating plots")
print("=" * 65)

fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Plot 1: ROC Curves ────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for name, probs, color in [
    ("Random Forest",  rf_test_probs,       "#2196F3"),
    ("XGBoost",        xgb_test_probs,      "#FF9800"),
    ("Neural Network", nn_test_probs,       "#9C27B0"),
    ("Ensemble",       ensemble_test_probs, "#F44336"),
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc_val:.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves — All Models vs Ensemble", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

# ── Plot 2: AUC Comparison Bar ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
models_names = ["Random\nForest", "XGBoost", "Neural\nNetwork", "Ensemble"]
auc_values   = [rf_test_auc, xgb_test_auc, nn_test_auc, ensemble_test_auc]
bar_colors   = ["#2196F3", "#FF9800", "#9C27B0", "#F44336"]
bars = ax2.bar(models_names, auc_values, color=bar_colors, edgecolor="white", width=0.6)
ax2.set_ylim(0.5, 1.0)
ax2.set_ylabel("Test AUC")
ax2.set_title("AUC Comparison", fontsize=13, fontweight="bold")
ax2.axhline(0.5, color="gray", linestyle="--", lw=1, label="Random baseline")
for bar, val in zip(bars, auc_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

# ── Plot 3: Confusion Matrix ──────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, ensemble_test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Stable", "Failure"])
disp.plot(ax=ax3, colorbar=False, cmap="Blues")
ax3.set_title("Ensemble Confusion Matrix", fontsize=12, fontweight="bold")

# ── Plot 4: RF Feature Importance ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
rf_importances = rf.feature_importances_
sorted_idx = np.argsort(rf_importances)
ax4.barh(
    [FEATURES[i] for i in sorted_idx],
    rf_importances[sorted_idx],
    color="#2196F3", edgecolor="white"
)
ax4.set_xlabel("Importance")
ax4.set_title("Random Forest\nFeature Importance", fontsize=12, fontweight="bold")
ax4.grid(axis="x", alpha=0.3)

# ── Plot 5: XGBoost Feature Importance ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
xgb_importances = xgb_model.feature_importances_
sorted_idx_xgb  = np.argsort(xgb_importances)
ax5.barh(
    [FEATURES[i] for i in sorted_idx_xgb],
    xgb_importances[sorted_idx_xgb],
    color="#FF9800", edgecolor="white"
)
ax5.set_xlabel("Importance")
ax5.set_title("XGBoost\nFeature Importance", fontsize=12, fontweight="bold")
ax5.grid(axis="x", alpha=0.3)

# ── Plot 6: NN Training Curves ────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
epochs_ran = len(history.history["loss"])
ax6.plot(history.history["loss"],     label="Train Loss", color="#9C27B0")
ax6.plot(history.history["val_loss"], label="Val Loss",   color="#9C27B0",
         linestyle="--")
ax6.set_xlabel("Epoch")
ax6.set_ylabel("Binary Cross-Entropy Loss")
ax6.set_title(f"NN Training — Loss\n(stopped at epoch {epochs_ran})",
              fontsize=12, fontweight="bold")
ax6.legend()
ax6.grid(alpha=0.3)

ax7 = fig.add_subplot(gs[2, 1])
ax7.plot(history.history["auc"],     label="Train AUC", color="#9C27B0")
ax7.plot(history.history["val_auc"], label="Val AUC",   color="#9C27B0",
         linestyle="--")
ax7.set_xlabel("Epoch")
ax7.set_ylabel("AUC")
ax7.set_title("NN Training — AUC", fontsize=12, fontweight="bold")
ax7.legend()
ax7.grid(alpha=0.3)

# ── Plot 7: Ensemble probability distribution ─────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.hist(ensemble_test_probs[y_test == 0], bins=50, alpha=0.6,
         color="#4CAF50", label="Stable (0)", density=True)
ax8.hist(ensemble_test_probs[y_test == 1], bins=50, alpha=0.6,
         color="#F44336", label="Failure (1)", density=True)
ax8.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold=0.5")
ax8.set_xlabel("Predicted Failure Probability")
ax8.set_ylabel("Density")
ax8.set_title("Ensemble — Score Distribution", fontsize=12, fontweight="bold")
ax8.legend()
ax8.grid(alpha=0.3)

plt.suptitle("MANET Link Failure Prediction — Training Results",
             fontsize=16, fontweight="bold", y=1.01)

plot_path = os.path.join("assets", "training_results.png")
os.makedirs("assets", exist_ok=True)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"  Plots saved to {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Final Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("TRAINING COMPLETE — Summary")
print("=" * 65)
print(f"  Features used      : {len(FEATURES)}")
print(f"  Train rows         : {len(X_train)}")
print(f"  Test rows          : {len(X_test)}")
print(f"  Train/test split   : by run_id (no data leakage)")
print(f"")
print(f"  Model              AUC (Test)")
print(f"  ──────────────── ──────────")
print(f"  Random Forest    {rf_test_auc:.4f}")
print(f"  XGBoost          {xgb_test_auc:.4f}")
print(f"  Neural Network   {nn_test_auc:.4f}")
print(f"  Ensemble         {ensemble_test_auc:.4f}  ← used for routing")
print(f"")
print(f"  Ensemble F1 (failure class) : "
      f"{f1_score(y_test, ensemble_test_preds):.4f}")
print(f"")
print(f"  Saved models:")
print(f"    models/random_forest.pkl")
print(f"    models/xgboost_model.pkl")
print(f"    models/neural_network.keras")
print(f"    models/scaler.pkl")
print(f"    models/ensemble_weights.pkl")
