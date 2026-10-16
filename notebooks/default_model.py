import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import joblib
import os


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

train = pd.read_csv(
    os.path.join(PROCESSED_DIR, "default_train.csv")
)

test = pd.read_csv(
    os.path.join(PROCESSED_DIR, "default_test.csv")
)


# ==========================================
# SPLIT FEATURES AND TARGET
# ==========================================

X_train = train.drop(columns="target")
y_train = train["target"]

X_test = test.drop(columns="target")
y_test = test["target"]


# ==========================================
# DIAGNOSTIC INFORMATION
# ==========================================

print("\nNumber of features:", X_train.shape[1])

print("\nFeatures used by model:")
print(X_train.columns.tolist())

print("\nData types:")
print(X_train.dtypes)

print("\nTrain target distribution:")
print(y_train.value_counts())

print("\nTest target distribution:")
print(y_test.value_counts())


# Check for exact duplicate feature rows
train_features = set(
    map(tuple, X_train.astype(str).values)
)

test_features = set(
    map(tuple, X_test.astype(str).values)
)

overlap = train_features.intersection(test_features)

print(
    "\nExact duplicate feature rows between train and test:",
    len(overlap)
)


# Check for suspicious target-related columns
target_related_columns = [
    col for col in X_train.columns
    if "status" in col.lower()
    or "default" in col.lower()
    or "charged" in col.lower()
    or "target" in col.lower()
]

print("\nPossible target-related columns:")
print(target_related_columns)


# ==========================================
# DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}


# ==========================================
# TRAIN MODELS
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Precision": precision_score(
            y_test,
            y_pred
        ),
        "Recall": recall_score(
            y_test,
            y_pred
        ),
        "F1": f1_score(
            y_test,
            y_pred
        ),
        "AUC-ROC": roc_auc_score(
            y_test,
            y_proba
        )
    })


# ==========================================
# MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(
    results
).sort_values(
    "AUC-ROC",
    ascending=False
)

print(
    "\nModel comparison:\n",
    results_df
)


results_df.to_csv(
    os.path.join(
        REPORTS_DIR,
        "default_model_comparison.csv"
    ),
    index=False
)


# ==========================================
# SELECT BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print(
    f"\nBest model: {best_model_name}"
)


# ==========================================
# BEST MODEL EVALUATION
# ==========================================

y_pred_best = best_model.predict(
    X_test
)

print(
    "\n",
    classification_report(
        y_test,
        y_pred_best
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred_best
)

plt.figure()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "No Default",
        "Default"
    ],
    yticklabels=[
        "No Default",
        "Default"
    ]
)

plt.title(
    f"Confusion Matrix — {best_model_name}"
)

plt.savefig(
    os.path.join(
        REPORTS_DIR,
        "default_confusion_matrix.png"
    ),
    bbox_inches="tight"
)

plt.close()


# ==========================================
# SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    os.path.join(
        MODELS_DIR,
        "default_model.pkl"
    )
)

with open(
    os.path.join(
        MODELS_DIR,
        "default_model_name.txt"
    ),
    "w"
) as f:
    f.write(best_model_name)


print(
    f"\nBest model saved to "
    f"{MODELS_DIR}/default_model.pkl"
)