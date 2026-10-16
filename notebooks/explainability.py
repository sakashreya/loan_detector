import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap

from lime.lime_tabular import LimeTabularExplainer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

approval_model = joblib.load(
    os.path.join(
        MODELS_DIR,
        "approval_model.pkl"
    )
)

default_model = joblib.load(
    os.path.join(
        MODELS_DIR,
        "default_model.pkl"
    )
)

print("Approval model type:", type(approval_model))
print("Default model type:", type(default_model))


# ============================================================
# STAGE 1: APPROVAL MODEL EXPLAINABILITY
# ============================================================

print("\nExplaining approval model...")

approval_test = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "approval_test.csv"
    )
)

X_app_test = approval_test.drop(columns="approved")
X_app_test["dti"] = pd.to_numeric(X_app_test["dti"], errors="coerce")
X_app_test["dti"] = X_app_test["dti"].fillna(X_app_test["dti"].median())

print(
    "Approval test features:",
    X_app_test.shape[1]
)

print(
    "Approval test shape:",
    X_app_test.shape
)


# ------------------------------------------------------------
# Choose correct SHAP explainer for approval model
# ------------------------------------------------------------

if isinstance(approval_model, LogisticRegression):

    explainer_app = shap.LinearExplainer(
        approval_model,
        X_app_test
    )

else:

    explainer_app = shap.TreeExplainer(
        approval_model
    )


# To keep memory/time reasonable, explain a sample
SHAP_SAMPLE_SIZE = 2000

if len(X_app_test) > SHAP_SAMPLE_SIZE:
    X_app_sample = X_app_test.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=42
    )
else:
    X_app_sample = X_app_test.copy()


shap_values_app = explainer_app.shap_values(
    X_app_sample
)


# ============================================================
# Approval SHAP Summary Plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values_app,
    X_app_sample,
    show=False
)

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "shap_approval_summary.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Approval SHAP Waterfall Plot
# ============================================================

i = 0

approval_base_value = explainer_app.expected_value

if hasattr(approval_base_value, "__len__"):
    approval_base_value = approval_base_value[0]

approval_waterfall = shap.Explanation(
    values=shap_values_app[i],
    base_values=approval_base_value,
    data=X_app_sample.iloc[i],
    feature_names=X_app_sample.columns
)

shap.plots.waterfall(
    approval_waterfall,
    show=False
)

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "shap_approval_waterfall_example.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STAGE 2: DEFAULT MODEL EXPLAINABILITY
# ============================================================

print("\nExplaining default model...")

default_train = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "default_train.csv"
    )
)

default_test = pd.read_csv(
    os.path.join(
        PROCESSED_DIR,
        "default_test.csv"
    )
)

X_def_train = default_train.drop(
    columns="target"
)

X_def_test = default_test.drop(
    columns="target"
)

print(
    "Default model expects features:",
    default_model.n_features_in_
)

print(
    "Default test features:",
    X_def_test.shape[1]
)

print(
    "Default test shape:",
    X_def_test.shape
)


# ============================================================
# Choose correct SHAP explainer for default model
# ============================================================

if isinstance(default_model, LogisticRegression):

    # Logistic Regression → LinearExplainer
    explainer_def = shap.LinearExplainer(
        default_model,
        X_def_train
    )

else:

    # XGBoost / Random Forest → TreeExplainer
    explainer_def = shap.TreeExplainer(
        default_model
    )


# Explain a sample instead of all 269,070 test rows
if len(X_def_test) > SHAP_SAMPLE_SIZE:

    X_def_sample = X_def_test.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=42
    )

else:

    X_def_sample = X_def_test.copy()


shap_values_def = explainer_def.shap_values(
    X_def_sample
)


# ============================================================
# Default SHAP Summary Plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values_def,
    X_def_sample,
    show=False
)

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "shap_default_summary.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Default SHAP Waterfall Plot
# ============================================================

j = 0

default_base_value = explainer_def.expected_value

if hasattr(default_base_value, "__len__"):
    default_base_value = default_base_value[0]

default_waterfall = shap.Explanation(
    values=shap_values_def[j],
    base_values=default_base_value,
    data=X_def_sample.iloc[j],
    feature_names=X_def_sample.columns
)

shap.plots.waterfall(
    default_waterfall,
    show=False
)

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "shap_default_waterfall_example.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LIME CROSS-CHECK FOR DEFAULT MODEL
# ============================================================

print("\nRunning LIME cross-check...")

# Use a smaller training sample for LIME
LIME_SAMPLE_SIZE = 5000

if len(X_def_train) > LIME_SAMPLE_SIZE:

    X_lime_train = X_def_train.sample(
        n=LIME_SAMPLE_SIZE,
        random_state=42
    )

else:

    X_lime_train = X_def_train.copy()


lime_explainer = LimeTabularExplainer(
    X_lime_train.values,
    feature_names=X_lime_train.columns.tolist(),
    class_names=["No Default", "Default"],
    mode="classification",
    random_state=42
)


exp = lime_explainer.explain_instance(
    X_def_sample.iloc[j].values,
    default_model.predict_proba,
    num_features=8
)

exp.save_to_file(
    os.path.join(
        REPORTS_DIR,
        "lime_default_example.html"
    )
)


# ============================================================
# FINISHED
# ============================================================

print(
    f"\nAll explainability outputs saved to: {REPORTS_DIR}"
)