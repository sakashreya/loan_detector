import os
import joblib
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(PROJECT_DIR, "models", "approval_model.pkl"))
encoders = joblib.load(os.path.join(PROJECT_DIR, "models", "approval_encoders.pkl"))
test = pd.read_csv(os.path.join(PROJECT_DIR, "data", "processed", "approval_test.csv"))

features = [str(c) for c in model.feature_names_in_]
X, y = test[features], test["approved"]

from sklearn.metrics import roc_auc_score
proba_all = model.predict_proba(X)[:, 1]
print("Test AUC:", round(roc_auc_score(y, proba_all), 4))
print("Median predicted probability among REAL approvals:",
      f"{pd.Series(proba_all)[y == 1].median():.1%}")
print("Median predicted probability among REAL rejections:",
      f"{pd.Series(proba_all)[y == 0].median():.1%}\n")

# Hold everything at a realistic "good applicant" baseline, vary credit score
purpose_code = encoders["purpose"].transform(["car"])[0]
base = {"loan_amnt": 6024, "dti": 3.0, "purpose": purpose_code}

print("Approval probability as credit score changes (loan_amnt=$6024, dti=3%, purpose=car):")
for score in [600, 650, 700, 730, 750, 780, 800, 820, 850]:
    row = pd.DataFrame([{**base, "risk_score": score}])[features]
    p = model.predict_proba(row)[0][1]
    print(f"  score {score}: {p:.1%}")