import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import joblib
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Fixed column order — must match MONOTONE_CONSTRAINTS below, position for position
FEATURE_ORDER = ["loan_amnt", "dti", "risk_score", "purpose"]

# 0 = no constraint, 1 = prediction must never decrease as the feature increases,
# -1 = prediction must never increase as the feature increases
MONOTONE_CONSTRAINTS = (0, -1, 1, 0)
# loan_amnt: no constraint | dti: higher DTI can never raise approval odds |
# risk_score: higher credit score can never lower approval odds | purpose: no constraint

train = pd.read_csv(os.path.join(PROCESSED_DIR, "approval_train.csv"))
test = pd.read_csv(os.path.join(PROCESSED_DIR, "approval_test.csv"))

X_train = train[FEATURE_ORDER]
y_train = train["approved"]
X_test = test[FEATURE_ORDER]
y_test = test["approved"]

X_train["dti"] = pd.to_numeric(X_train["dti"], errors="coerce")
X_test["dti"] = pd.to_numeric(X_test["dti"], errors="coerce")
dti_median = X_train["dti"].median()
X_train["dti"] = X_train["dti"].fillna(dti_median)
X_test["dti"] = X_test["dti"].fillna(dti_median)

print("Training approval model (with monotonic constraints + calibration)...")
base_model = XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    monotone_constraints=MONOTONE_CONSTRAINTS,
)
approval_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
approval_model.fit(X_train, y_train)

y_pred = approval_model.predict(X_test)
y_proba = approval_model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "AUC-ROC": roc_auc_score(y_test, y_proba),
}
print("\nApproval Model Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
print("\n", classification_report(y_test, y_pred))

pd.DataFrame([metrics]).to_csv(os.path.join(REPORTS_DIR, "approval_metrics.csv"), index=False)

cm = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title("Confusion Matrix — Approval Model")
plt.savefig(os.path.join(REPORTS_DIR, "approval_confusion_matrix.png"), bbox_inches="tight")
plt.close()

joblib.dump(approval_model, os.path.join(MODELS_DIR, "approval_model.pkl"))
print(f"\nModel saved to {MODELS_DIR}/approval_model.pkl")