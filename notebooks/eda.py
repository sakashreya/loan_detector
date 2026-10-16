import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(PROCESSED_DIR, "default_train.csv"))

plt.figure()
sns.countplot(x="target", data=train)
plt.title("Default vs Non-Default Distribution (after SMOTE)")
plt.savefig(os.path.join(REPORTS_DIR, "eda_class_balance.png"), bbox_inches="tight")
plt.close()

candidate_features = ["annual_inc", "dti", "loan_amnt", "int_rate", "installment"]
present_features = [c for c in candidate_features if c in train.columns]

for col in present_features:
    plt.figure()
    sns.boxplot(x="target", y=col, data=train)
    plt.title(f"{col} by Default Status")
    plt.savefig(os.path.join(REPORTS_DIR, f"eda_{col}_boxplot.png"), bbox_inches="tight")
    plt.close()

plt.figure(figsize=(14, 10))
sns.heatmap(train.corr(numeric_only=True), cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(REPORTS_DIR, "eda_correlation_heatmap.png"), bbox_inches="tight")
plt.close()

print(f"EDA plots saved to {REPORTS_DIR}/")
print("Features plotted:", present_features)