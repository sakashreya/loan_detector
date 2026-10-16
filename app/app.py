import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from sklearn.linear_model import LogisticRegression

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

# The model was trained on US dollar figures. We convert INR <-> USD at this
# rate so the model still sees the currency it was trained on, while every
# number the user types and sees stays in rupees. Update this if needed.
EXCHANGE_RATE = 83.0  # 1 USD = ~83 INR

TRAIN_LOAN_CAP_USD = 40000  # LendingClub's actual personal-loan product cap
TRAIN_LOAN_CAP_INR = TRAIN_LOAN_CAP_USD * EXCHANGE_RATE


def to_usd(inr_amount):
    return inr_amount / EXCHANGE_RATE


def to_inr(usd_amount):
    return usd_amount * EXCHANGE_RATE


def format_inr(amount):
    """Format a number using Indian digit grouping, e.g. 1234567 -> '12,34,567'."""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    s = str(amount)
    if len(s) <= 3:
        return sign + s
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return sign + ",".join(parts) + "," + last3


def rupees(amount):
    return f"\u20b9{format_inr(amount)}"


st.set_page_config(page_title="AI Loan Default Predictor", layout="centered")
st.title("AI Loan Default Predictor")
st.caption("A two-stage explainable AI system: approval likelihood, then default risk.")


@st.cache_resource
def load_model_file(name):
    return joblib.load(os.path.join(MODELS_DIR, name))


@st.cache_data
def build_default_reference(features):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "default_test.csv"))
    missing = [c for c in features if c not in df.columns]
    baseline = df[[c for c in features if c in df.columns]].median().reindex(features, fill_value=0)
    background = df.reindex(columns=features, fill_value=0).sample(
        n=min(1000, len(df)), random_state=42)
    return baseline, background, missing


approval_model = load_model_file("approval_model.pkl")
default_model = load_model_file("default_model.pkl")
approval_encoders = load_model_file("approval_encoders.pkl")
default_encoders = load_model_file("default_encoders.pkl")
default_scaler = load_model_file("default_scaler.pkl")


def get_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "estimator") and hasattr(model.estimator, "feature_names_in_"):
        return list(model.estimator.feature_names_in_)
    if hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
        if names:
            return names
    raise ValueError("Could not determine model's expected feature names.")


approval_features = [str(c) for c in get_feature_names(approval_model)]
default_features = [str(c) for c in get_feature_names(default_model)]

default_baseline, background, missing_in_test = build_default_reference(default_features)
if missing_in_test:
    print("WARNING - features model expects but not in default_test.csv:", missing_in_test)

scaler_mean = dict(zip(default_scaler.feature_names_in_, default_scaler.mean_))
scaler_scale = dict(zip(default_scaler.feature_names_in_, default_scaler.scale_))


def scale(col, raw_value):
    return (raw_value - scaler_mean[col]) / scaler_scale[col]


def raw_default(col):
    """Median of a feature, converted back from scaled units to real units (USD or %)."""
    v = float(default_baseline[col])
    if col in scaler_mean:
        v = v * scaler_scale[col] + scaler_mean[col]
    return v


CURRENCY_FIELDS = {"loan_amnt", "annual_inc", "installment"}

# ---------------------------------------------------------------------------
# Applicant details (shared by both stages) — all figures in rupees
# ---------------------------------------------------------------------------
st.header("Applicant Details")

c1, c2 = st.columns(2)
with c1:
    loan_amnt_inr = st.number_input("Loan Amount Requested (\u20b9)", min_value=5000,
                                     value=500000, step=50000)
    annual_inc_inr = st.number_input("Annual Income (\u20b9)", min_value=100000,
                                      value=600000, step=50000)
    purpose_label = st.selectbox("Reason for Loan",
                                  list(approval_encoders["purpose"].classes_))
with c2:
    monthly_debt_inr = st.number_input("Existing Monthly Debt Payments (\u20b9)", min_value=0,
                                        value=15000, step=1000)
    credit_score = st.slider("Credit Score (FICO)", 300, 850, 700, step=5)

monthly_income_inr = annual_inc_inr / 12
dti = min(monthly_debt_inr / monthly_income_inr * 100, 100.0)
loan_to_income = loan_amnt_inr / annual_inc_inr

m1, m2 = st.columns(2)
m1.metric("Debt-to-Income (existing debts)", f"{dti:.1f}%")
m2.metric("Loan-to-Income", f"{loan_to_income:.0%}")

is_jumbo = loan_amnt_inr > TRAIN_LOAN_CAP_INR
if is_jumbo:
    st.info(f"This dataset's lender never issues loans above about {rupees(TRAIN_LOAN_CAP_INR)} "
             f"(${TRAIN_LOAN_CAP_USD:,} — its actual product cap). That's a product limit, not "
             f"a risk decision, and there are zero approved examples above it to learn from. "
             f"For amounts above this, the app switches from the trained model to a clear, "
             f"stated policy rule instead of guessing.")

# Dollar equivalents, used only for talking to the model
loan_amnt = to_usd(loan_amnt_inr)
annual_inc = to_usd(annual_inc_inr)

# ---------------------------------------------------------------------------
# STAGE 1 - Approval
# ---------------------------------------------------------------------------
st.header("Stage 1 — Approval Likelihood")

if not is_jumbo:
    approval_row = {
        "loan_amnt": loan_amnt,
        "dti": dti,
        "risk_score": credit_score,
        "purpose": approval_encoders["purpose"].transform([purpose_label])[0],
    }

    if set(approval_row) != set(approval_features):
        st.error(f"The approval model was trained on {sorted(approval_features)}, but this "
                 f"form provides {sorted(approval_row)}. Re-run the notebook and "
                 f"approval_model.py.")
        st.stop()

    approval_input = pd.DataFrame([approval_row])[approval_features]
    approval_proba = float(approval_model.predict_proba(approval_input)[0][1])
    used_policy = False
else:
    # Rule-based fallback for amounts outside the training range.
    # Stated policy, not learned from data — good credit, manageable existing
    # debt, and a loan that isn't disproportionate to income.
    score_ok = credit_score >= 700
    dti_ok = dti <= 30
    ratio_ok = loan_to_income <= 3.0
    approval_proba = 0.85 if (score_ok and dti_ok and ratio_ok) else 0.10
    used_policy = True

    CHECK = "\u2705"
    CROSS = "\u274c"
    st.write(f"- Credit score \u2265 700: {CHECK if score_ok else CROSS} ({credit_score})")
    st.write(f"- Debt-to-income \u2264 30%: {CHECK if dti_ok else CROSS} ({dti:.1f}%)")
    st.write(f"- Loan-to-income \u2264 3x: {CHECK if ratio_ok else CROSS} ({loan_to_income:.1f}x)")

st.metric("Estimated Approval Probability", f"{approval_proba:.1%}")
st.progress(min(max(approval_proba, 0.0), 1.0))

if not used_policy:
    with st.expander("Why this approval prediction? (SHAP)"):
        if hasattr(approval_model, "calibrated_classifiers_"):
            tree_model = approval_model.calibrated_classifiers_[0].estimator
        else:
            tree_model = approval_model

        explainer_app = shap.TreeExplainer(tree_model)
        sv_app = explainer_app.shap_values(approval_input)
        base_val_app = explainer_app.expected_value
        if hasattr(base_val_app, "__len__"):
            base_val_app = base_val_app[0]

        # Show the waterfall with the loan amount labeled in rupees for readability
        display_row = approval_input.iloc[0].copy()
        display_row["loan_amnt"] = loan_amnt_inr

        fig1 = plt.figure()
        shap.plots.waterfall(
            shap.Explanation(values=sv_app[0], base_values=base_val_app,
                              data=display_row.values, feature_names=approval_features),
            show=False)
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption("Loan amount is shown in \u20b9 above; the model itself computed this in USD.")
else:
    st.caption("SHAP explanation isn't shown here because this decision came from the "
               "stated policy rule above, not the trained model.")

# ---------------------------------------------------------------------------
# STAGE 2 - Default risk (always shown)
# ---------------------------------------------------------------------------
st.header("Stage 2 — Default Risk")
if approval_proba >= 0.5:
    st.success("Stage 1 leans toward approval.")
else:
    st.info("Stage 1 leans toward rejection, but here is the default risk assessment anyway.")
st.caption("Loan amount, income, DTI and credit score carry over from above. "
           "Adjust the loan terms below.")

user_raw = {}
if "loan_amnt" in default_features:
    user_raw["loan_amnt"] = float(loan_amnt)
if "dti" in default_features:
    user_raw["dti"] = float(dti)
if "annual_inc" in default_features:
    user_raw["annual_inc"] = float(annual_inc)
if "fico_score" in default_features:
    user_raw["fico_score"] = float(credit_score)

stage2_fields = [f for f in ["int_rate", "term", "grade",
                              "home_ownership", "verification_status"]
                 if f in default_features]

kcols = st.columns(2)
for idx, col in enumerate(stage2_fields):
    with kcols[idx % 2]:
        label = col.replace("_", " ").title()
        if col in default_encoders:
            classes = list(default_encoders[col].classes_)
            default_idx = int(min(max(round(raw_default(col)), 0), len(classes) - 1))
            user_raw[col] = st.selectbox(label, classes, index=default_idx, key=f"def_{col}")
        else:
            default_val = raw_default(col)
            if col in CURRENCY_FIELDS:
                default_val = to_inr(default_val)
            user_raw[col] = st.number_input(f"{label} (\u20b9)" if col in CURRENCY_FIELDS else label,
                                             value=default_val, key=f"def_{col}")
            if col in CURRENCY_FIELDS:
                user_raw[col] = to_usd(user_raw[col])

if "installment" in default_features:
    digits = "".join(ch for ch in str(user_raw.get("term", "36")) if ch.isdigit())
    n_months = int(digits) if digits else 36
    r = user_raw.get("int_rate", 12.0) / 1200
    p = float(loan_amnt)
    installment_usd = p / n_months if r == 0 else p * r / (1 - (1 + r) ** -n_months)
    user_raw["installment"] = installment_usd
    installment_inr = to_inr(installment_usd)

    i1, i2 = st.columns(2)
    i1.metric("Monthly Installment", rupees(installment_inr))
    i2.metric("Payment-to-Income", f"{installment_inr * 12 / annual_inc_inr:.1%}")

row = default_baseline.copy()
for col, raw_value in user_raw.items():
    if col in default_encoders:
        encoded = default_encoders[col].transform([raw_value])[0]
        row[col] = scale(col, encoded) if col in scaler_mean else encoded
    else:
        row[col] = scale(col, raw_value) if col in scaler_mean else raw_value

default_input = pd.DataFrame([row]).reindex(columns=default_features, fill_value=0)
default_proba = float(default_model.predict_proba(default_input)[0][1])

st.metric("Estimated Default Risk", f"{default_proba:.1%}")
st.progress(min(max(default_proba, 0.0), 1.0))

if is_jumbo:
    st.caption("Note: this loan amount is above the trained default model's range too — "
               "treat this risk estimate as an extrapolation.")

st.subheader("Why this prediction? (SHAP)")
if isinstance(default_model, LogisticRegression):
    explainer_def = shap.LinearExplainer(default_model, background)
else:
    explainer_def = shap.TreeExplainer(default_model)
sv_def = explainer_def.shap_values(default_input)
base_val = explainer_def.expected_value
if hasattr(base_val, "__len__"):
    base_val = base_val[0]

fig2 = plt.figure()
shap.plots.waterfall(
    shap.Explanation(values=sv_def[0], base_values=base_val,
                      data=default_input.iloc[0], feature_names=default_features),
    show=False)
st.pyplot(fig2)
plt.close(fig2)
st.caption("Red bars push the applicant toward default, blue bars toward repayment. "
           "E[f(X)] is the average applicant's score and f(x) is this applicant's. "
           "Both are in log-odds, not percentages. Currency-based features were "
           "converted to USD before scoring, since the model was trained on USD figures.")