import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

COLUMN_MAP = {
    "Amount Requested": "loan_amnt",
    "Debt-To-Income Ratio": "dti",
    "Risk_Score": "risk_score",
    "Loan Title": "purpose_raw",
    "State": "addr_state",
    "Application Date": "app_date",
}

print("Loading raw CSVs (this can take a few minutes)...")
accepted = pd.read_csv(os.path.join(RAW_DIR, "accepted_2007_to_2018Q4.csv"), low_memory=False)
rejected = pd.read_csv(os.path.join(RAW_DIR, "rejected_2007_to_2018Q4.csv"), low_memory=False)
rejected = rejected.rename(columns=COLUMN_MAP)
print("Accepted shape:", accepted.shape)
print("Rejected shape:", rejected.shape)

APPROVAL_FEATURES = ["loan_amnt", "dti", "risk_score", "purpose"]


def normalize_purpose(text):
    t = str(text).lower()
    if "debt" in t:
        return "debt_consolidation"
    if "credit" in t and "card" in t:
        return "credit_card"
    if "home" in t and "improve" in t:
        return "home_improvement"
    if "major" in t and "purchase" in t:
        return "major_purchase"
    if "medical" in t:
        return "medical"
    if "car" in t or "auto" in t:
        return "car"
    if "business" in t:
        return "small_business"
    if "wedding" in t:
        return "wedding"
    if "vacation" in t or "travel" in t:
        return "vacation"
    if "house" in t:
        return "house"
    if "moving" in t or "relocation" in t:
        return "moving"
    if "educ" in t or "student" in t:
        return "educational"
    if "renewable" in t or "solar" in t or "green" in t:
        return "renewable_energy"
    return "other"


print("\nBuilding approval dataset...")

rej_common = rejected[["loan_amnt", "dti", "risk_score", "purpose_raw",
                       "addr_state", "app_date"]].dropna(subset=["risk_score", "dti"]).copy()
rej_common["purpose"] = rej_common["purpose_raw"].apply(normalize_purpose)
rej_common = rej_common.drop(columns=["purpose_raw"])
rej_common["approved"] = 0

acc_common = accepted[["loan_amnt", "dti", "fico_range_low", "fico_range_high",
                       "purpose", "addr_state", "issue_d"]].copy()
acc_common["risk_score"] = (acc_common["fico_range_low"] + acc_common["fico_range_high"]) / 2
acc_common["purpose"] = acc_common["purpose"].astype(str).str.lower().str.strip()
acc_common = acc_common.rename(columns={"issue_d": "app_date"})
acc_common = acc_common.drop(columns=["fico_range_low", "fico_range_high"])
acc_common = acc_common.dropna(subset=["risk_score", "dti"])
acc_common["approved"] = 1

rej_dates = pd.to_datetime(rej_common["app_date"], errors="coerce")
cutoff = rej_dates.max()
assert pd.notna(cutoff), "Could not read the rejected 'Application Date' column"
acc_dates = pd.to_datetime(acc_common["app_date"], format="%b-%Y", errors="coerce")
acc_common = acc_common[acc_dates <= cutoff].copy()
print("Credit scores in rejected file run up to:", cutoff.date())

for df_ in (acc_common, rej_common):
    df_["dti"] = pd.to_numeric(df_["dti"].astype(str).str.replace("%", "", regex=False),
                               errors="coerce")
    df_["risk_score"] = pd.to_numeric(df_["risk_score"], errors="coerce")
    df_["loan_amnt"] = pd.to_numeric(df_["loan_amnt"], errors="coerce")

max_funded = acc_common["loan_amnt"].max()
rej_common = rej_common[rej_common["loan_amnt"] <= max_funded]
print("Largest funded loan (training range cap):", max_funded)
print("Accepted kept:", len(acc_common), "| Rejected kept:", len(rej_common))

approval_df = pd.concat([acc_common, rej_common], ignore_index=True)
approval_df = approval_df.dropna(subset=["loan_amnt", "dti", "risk_score", "purpose"])
approval_df = approval_df[(approval_df["dti"] >= 0) & (approval_df["dti"] <= 100)]

top_purposes = approval_df["purpose"].value_counts().nlargest(8).index
approval_df["purpose"] = approval_df["purpose"].where(
    approval_df["purpose"].isin(top_purposes), "other")
print("\nPurpose categories:")
print(approval_df["purpose"].value_counts())

approved_rows = approval_df[approval_df["approved"] == 1]
rejected_rows = approval_df[approval_df["approved"] == 0]
n = min(len(approved_rows), len(rejected_rows))
approval_balanced = pd.concat([
    resample(approved_rows, n_samples=n, replace=False, random_state=42),
    resample(rejected_rows, n_samples=n, replace=False, random_state=42),
]).reset_index(drop=True)
print("\nBalanced:", approval_balanced["approved"].value_counts().to_dict())

approval_encoders = {}
le = LabelEncoder()
approval_balanced["purpose"] = le.fit_transform(approval_balanced["purpose"].astype(str))
approval_encoders["purpose"] = le
joblib.dump(approval_encoders, os.path.join(MODELS_DIR, "approval_encoders.pkl"))

audit_state = approval_balanced["addr_state"].astype(str)
X_app = approval_balanced[APPROVAL_FEATURES]
y_app = approval_balanced["approved"]

X_train_app, X_test_app, y_train_app, y_test_app, _, state_test = train_test_split(
    X_app, y_app, audit_state, test_size=0.2, stratify=y_app, random_state=42)

X_train_app.assign(approved=y_train_app).to_csv(
    os.path.join(PROCESSED_DIR, "approval_train.csv"), index=False)
X_test_app.assign(approved=y_test_app).to_csv(
    os.path.join(PROCESSED_DIR, "approval_test.csv"), index=False)
state_test.to_frame("addr_state").to_csv(
    os.path.join(PROCESSED_DIR, "approval_test_state.csv"), index=False)

print("\nDONE. Approval dataset saved.")
print("Train shape:", X_train_app.shape, "| Test shape:", X_test_app.shape)
print("Columns:", X_train_app.columns.tolist())