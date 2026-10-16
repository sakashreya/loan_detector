import pandas as pd
import os

REPORTS_DIR = "reports"

default_comparison = pd.read_csv(os.path.join(REPORTS_DIR, "default_model_comparison.csv"))
approval_metrics = pd.read_csv(os.path.join(REPORTS_DIR, "approval_metrics.csv"))
fairness = pd.read_csv(os.path.join(REPORTS_DIR, "fairness_by_state.csv"))

best_default_row = default_comparison.sort_values("AUC-ROC", ascending=False).iloc[0]
approval_gap = fairness["selection_rate"].max() - fairness["selection_rate"].min()

summary = f"""# Project Summary — AI Loan Default Predictor with Explainability

## Stage 1 — Approval Model
| Metric | Value |
|---|---|
| Accuracy | {approval_metrics['Accuracy'][0]:.3f} |
| Precision | {approval_metrics['Precision'][0]:.3f} |
| Recall | {approval_metrics['Recall'][0]:.3f} |
| F1 | {approval_metrics['F1'][0]:.3f} |
| AUC-ROC | {approval_metrics['AUC-ROC'][0]:.3f} |

See `shap_approval_summary.png` for the global drivers of approval, and
`shap_approval_waterfall_example.png` for a single-applicant explanation.

## Stage 2 — Default Model
Best model: **{best_default_row['Model']}**

| Metric | Value |
|---|---|
| Accuracy | {best_default_row['Accuracy']:.3f} |
| Precision | {best_default_row['Precision']:.3f} |
| Recall | {best_default_row['Recall']:.3f} |
| F1 | {best_default_row['F1']:.3f} |
| AUC-ROC | {best_default_row['AUC-ROC']:.3f} |

Full comparison across all 3 algorithms is in `default_model_comparison.csv`.
See `shap_default_summary.png` and `shap_default_waterfall_example.png` for
global and per-applicant explanations, and `lime_default_example.html` for
the LIME cross-check.

## Fairness Audit
Largest gap in approval rate between states: **{approval_gap:.1%}**
(full breakdown in `fairness_by_state.csv`)

## Key Takeaway
This project demonstrates a two-stage explainable pipeline: Stage 1 predicts
whether an application would be approved using only pre-decision information,
and Stage 2 — for approved loans — predicts default risk. Both stages are
explained via SHAP (and cross-checked with LIME for Stage 2), and the fairness
audit quantifies whether approval rates vary meaningfully across geography.
"""

with open(os.path.join(REPORTS_DIR, "final_summary.md"), "w") as f:
    f.write(summary)

print(summary)
print(f"\nSaved to {REPORTS_DIR}/final_summary.md")