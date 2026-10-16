import os
import itertools
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(ROOT, "models", "approval_model.pkl"))
features = [str(c) for c in model.feature_names_in_]

expected = {"loan_amnt", "dti", "risk_score"}
if set(features) != expected:
    raise SystemExit(f"Model was trained on {features}, not {sorted(expected)}. "
                     "Retrain with approval_model.py first.")

TARGET = 0.70
amounts = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]
dtis = [5, 10, 15, 20, 25, 30, 35, 40, 50]
scores = list(range(600, 851, 10))

grid = pd.DataFrame(list(itertools.product(amounts, dtis, scores)),
                    columns=["loan_amnt", "dti", "risk_score"])
grid["prob"] = model.predict_proba(grid[features])[:, 1]

ok = grid[grid["prob"] >= TARGET]
print(f"{len(ok)} of {len(grid)} combinations reach {TARGET:.0%} or more.\n")

print("Lowest credit score that reaches 70% (rows = loan amount, columns = DTI %):\n")
table = ok.pivot_table(index="loan_amnt", columns="dti",
                       values="risk_score", aggfunc="min")
table = table.reindex(index=amounts, columns=dtis)
print(table.to_string(float_format=lambda x: f"{x:.0f}", na_rep="none"))

print("\nA few safe examples (80%+):")
print(grid[grid["prob"] >= 0.80].sort_values("prob", ascending=False)
      .head(8).round(3).to_string(index=False))