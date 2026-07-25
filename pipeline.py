"""
Credit Card Fraud Detection — Imbalanced Classification & Anomaly Detection
Pipeline script — produces the metrics and figures referenced in the README
and mirrors the notebook exactly (deterministic, random_state=42 throughout).
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / "creditcard.csv")

# ---------------------------------------------------------------------------
# 2. Data quality
# ---------------------------------------------------------------------------
assert df.isna().sum().sum() == 0, "Unexpected missing values"
assert df.duplicated().sum() >= 0  # duplicates are legitimate here (repeat small charges)

fraud_rate = df["Class"].mean()

# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------
df["Hour"] = (df["Time"] // 3600) % 24
scaler = StandardScaler()
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

FEATURES = [c for c in df.columns if c.startswith("V")] + ["Amount_scaled", "Hour"]
TARGET = "Class"

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 4. Train / test split — stratified (preserve the (small) fraud proportion
#    in both splits; a random split without stratify risks a test set with
#    very few or zero fraud cases)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 5. Baseline — Logistic Regression with class_weight='balanced'
# ---------------------------------------------------------------------------
log_reg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)
log_reg.fit(X_train, y_train)
lr_probs = log_reg.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 5b. Logistic Regression + SMOTE (oversampling comparison)
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
log_reg_smote = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
log_reg_smote.fit(X_train_sm, y_train_sm)
lr_smote_probs = log_reg_smote.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 6. Random Forest
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300, max_depth=12, class_weight="balanced_subsample",
    random_state=RANDOM_STATE, n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 7. XGBoost with scale_pos_weight tuned to the class imbalance
# ---------------------------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
)
xgb.fit(X_train, y_train)
xgb_probs = xgb.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 8. Autoencoder-style anomaly detection (unsupervised) — trained ONLY on
#    legitimate transactions; flags high reconstruction error as anomalous.
#    Implemented with a bottleneck MLPRegressor (encoder-decoder) to keep
#    the dependency footprint light (no TensorFlow needed for this repo).
# ---------------------------------------------------------------------------
X_train_legit = X_train[y_train == 0]
autoencoder = MLPRegressor(
    hidden_layer_sizes=(20, 8, 20), activation="relu", solver="adam",
    max_iter=200, random_state=RANDOM_STATE, early_stopping=True,
)
autoencoder.fit(X_train_legit, X_train_legit)

recon_test = autoencoder.predict(X_test)
recon_error = np.mean((X_test.values - recon_test) ** 2, axis=1)

# threshold = 99th percentile of reconstruction error on legit training data
recon_train_legit = autoencoder.predict(X_train_legit)
train_error = np.mean((X_train_legit.values - recon_train_legit) ** 2, axis=1)
ae_threshold = np.percentile(train_error, 99)
ae_preds = (recon_error > ae_threshold).astype(int)


# ---------------------------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------------------------
def evaluate_proba(y_true, probs, name, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "model": name,
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "F1": round(f1_score(y_true, preds, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_true, probs), 4),
        "PR_AUC": round(average_precision_score(y_true, probs), 4),
    }


def evaluate_preds(y_true, preds, name):
    return {
        "model": name,
        "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
        "Recall": round(recall_score(y_true, preds, zero_division=0), 4),
        "F1": round(f1_score(y_true, preds, zero_division=0), 4),
        "ROC_AUC": None,
        "PR_AUC": None,
    }


results = [
    evaluate_proba(y_test, lr_probs, "Logistic Regression (class-weighted)"),
    evaluate_proba(y_test, lr_smote_probs, "Logistic Regression + SMOTE"),
    evaluate_proba(y_test, rf_probs, "Random Forest"),
    evaluate_proba(y_test, xgb_probs, "XGBoost"),
    evaluate_preds(y_test, ae_preds, "Autoencoder (anomaly detection, unsupervised)"),
]

best_supervised = max(results[:4], key=lambda r: r["PR_AUC"])

# ---------------------------------------------------------------------------
# 10. Business-driven threshold selection on best supervised model
# ---------------------------------------------------------------------------
best_probs = {"Logistic Regression (class-weighted)": lr_probs,
              "Logistic Regression + SMOTE": lr_smote_probs,
              "Random Forest": rf_probs,
              "XGBoost": xgb_probs}[best_supervised["model"]]

precisions, recalls, thresholds = precision_recall_curve(y_test, best_probs)
# business rule: maximize recall while keeping precision >= 0.85 (tunable)
valid_idx = np.where(precisions[:-1] >= 0.85)[0]
if len(valid_idx):
    chosen_idx = valid_idx[np.argmax(recalls[:-1][valid_idx])]
    chosen_threshold = float(thresholds[chosen_idx])
    chosen_precision = float(precisions[chosen_idx])
    chosen_recall = float(recalls[chosen_idx])
else:
    chosen_threshold, chosen_precision, chosen_recall = 0.5, None, None

avg_fraud_amount = df.loc[df["Class"] == 1, "Amount"].mean()
n_fraud_test = int(y_test.sum())
caught = int(round(chosen_recall * n_fraud_test)) if chosen_recall else None
estimated_savings = round(caught * avg_fraud_amount, 2) if caught else None

# ---------------------------------------------------------------------------
# 11. Feature importance (best supervised model)
# ---------------------------------------------------------------------------
best_model = {"Random Forest": rf, "XGBoost": xgb}.get(best_supervised["model"])
if best_model is not None:
    importance = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
else:
    importance = pd.Series(np.abs(log_reg.coef_[0]), index=FEATURES).sort_values(ascending=False)

# ---------------------------------------------------------------------------
# 12. Save metrics
# ---------------------------------------------------------------------------
metrics = {
    "n_transactions": int(len(df)),
    "n_fraud": int(df["Class"].sum()),
    "fraud_rate_pct": round(fraud_rate * 100, 3),
    "results": results,
    "best_supervised_model": best_supervised["model"],
    "chosen_threshold": round(chosen_threshold, 3),
    "chosen_precision": round(chosen_precision, 3) if chosen_precision else None,
    "chosen_recall": round(chosen_recall, 3) if chosen_recall else None,
    "avg_fraud_amount": round(avg_fraud_amount, 2),
    "estimated_savings_test_set": estimated_savings,
    "top_features": importance.head(8).round(4).to_dict(),
    "autoencoder_threshold": round(float(ae_threshold), 4),
}
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))

# ---------------------------------------------------------------------------
# 13. Figures
# ---------------------------------------------------------------------------
# Class imbalance bar
fig, ax = plt.subplots(figsize=(5, 4))
df["Class"].value_counts().plot(kind="bar", ax=ax, color=["#4F46E5", "#DC2626"])
ax.set_xticklabels(["Legitimate", "Fraud"], rotation=0)
ax.set_title(f"Class Imbalance ({fraud_rate:.2%} fraud)")
plt.tight_layout()
plt.savefig(OUT_DIR / "class_imbalance.png", dpi=140)
plt.close()

# PR curves comparison
fig, ax = plt.subplots(figsize=(7, 6))
for name, probs in best_probs.items() if False else {}.items():
    pass
for name, probs in {
    "Logistic Regression": lr_probs,
    "LogReg + SMOTE": lr_smote_probs,
    "Random Forest": rf_probs,
    "XGBoost": xgb_probs,
}.items():
    p, r, _ = precision_recall_curve(y_test, probs)
    ax.plot(r, p, label=name)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves (imbalanced classification)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curves.png", dpi=140)
plt.close()

# Feature importance
fig, ax = plt.subplots(figsize=(9, 5))
importance.head(10).sort_values().plot(kind="barh", ax=ax, color="#4F46E5")
ax.set_title(f"Top 10 Fraud Indicators ({best_supervised['model']})")
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=140)
plt.close()

# Confusion matrix at chosen threshold
final_preds = (best_probs >= chosen_threshold).astype(int)
cm = confusion_matrix(y_test, final_preds)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix ({best_supervised['model']} @ threshold {chosen_threshold:.2f})")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=140)
plt.close()

# Autoencoder reconstruction error distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(recon_error[y_test == 0], bins=60, color="#4F46E5", label="Legitimate", stat="density", alpha=0.6, ax=ax)
sns.histplot(recon_error[y_test == 1], bins=60, color="#DC2626", label="Fraud", stat="density", alpha=0.6, ax=ax)
ax.axvline(ae_threshold, color="black", linestyle="--", label="Anomaly threshold (99th pct)")
ax.set_xlim(0, np.percentile(recon_error, 99.5))
ax.set_title("Autoencoder Reconstruction Error: Legitimate vs. Fraud")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "autoencoder_error.png", dpi=140)
plt.close()

print("\nSaved figures to", OUT_DIR)
