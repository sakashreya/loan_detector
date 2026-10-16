import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import recall_score
import joblib
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

approval_model = joblib.load(os.path.join(MODELS_DIR, "approval_model.pkl"))
test = pd.read_csv(os.path.join(PROCESSED_DIR, "approval_test.csv"))
state_labels = pd.read_csv(os.path.join(PROCESSED_DIR, "approval_test_state.csv"))["addr_state"]

X_test = test.drop(columns="approved")
y_test = test["approved"]
y_pred = approval_model.predict(X_test)

metric_frame = MetricFrame(
    metrics={"selection_rate": selection_rate, "recall": recall_score},
    y_true=y_test.values,
    y_pred=y_pred,
    sensitive_features=state_labels.values,
)

by_group = metric_frame.by_group.sort_values("selection_rate")
print("Approval rate and recall by state:\n", by_group)
by_group.to_csv(os.path.join(REPORTS_DIR, "fairness_by_state.csv"))

gap = by_group["selection_rate"].max() - by_group["selection_rate"].min()
print(f"\nLargest approval-rate gap between states: {gap:.2%}")