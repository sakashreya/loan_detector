import os
import time
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project root:", ROOT, "\n")

def stamp(path):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))

# 1. Find EVERY copy of these files anywhere in the project
targets = ["approval_train.csv", "approval_test.csv", "approval_model.pkl", "default_model.pkl"]
print("=== All copies found ===")
for dirpath, _, files in os.walk(ROOT):
    for f in files:
        if f in targets:
            full = os.path.join(dirpath, f)
            print(f"{stamp(full)}  {os.path.relpath(full, ROOT)}")

# Also check the folder above the project, in case the notebook wrote there
parent = os.path.dirname(ROOT)
for sub in ["data", "models"]:
    if os.path.exists(os.path.join(parent, sub)):
        print(f"WARNING: stray folder exists outside the project: {os.path.join(parent, sub)}")

# 2. What is in the training data the model would be built from
csv_path = os.path.join(ROOT, "data", "processed", "approval_train.csv")
print("\n=== data/processed/approval_train.csv ===")
if os.path.exists(csv_path):
    print("Columns:", list(pd.read_csv(csv_path, nrows=1).columns))
else:
    print("MISSING")

# 3. What the model the app loads was actually trained on
pkl_path = os.path.join(ROOT, "models", "approval_model.pkl")
print("\n=== models/approval_model.pkl (the file the app loads) ===")
if os.path.exists(pkl_path):
    m = joblib.load(pkl_path)
    print("Trained on:", [str(c) for c in m.feature_names_in_])
else:
    print("MISSING")