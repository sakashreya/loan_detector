# Project Summary — AI Loan Default Predictor with Explainability

## Stage 1 — Approval Model
| Metric | Value |
|---|---|
| Accuracy | 0.973 |
| Precision | 0.961 |
| Recall | 0.985 |
| F1 | 0.973 |
| AUC-ROC | 0.995 |

See `shap_approval_summary.png` for the global drivers of approval, and
`shap_approval_waterfall_example.png` for a single-applicant explanation.

## Stage 2 — Default Model
Best model: **Logistic Regression**

| Metric | Value |
|---|---|
| Accuracy | 0.661 |
| Precision | 0.323 |
| Recall | 0.636 |
| F1 | 0.428 |
| AUC-ROC | 0.709 |

Full comparison across all 3 algorithms is in `default_model_comparison.csv`.
See `shap_default_summary.png` and `shap_default_waterfall_example.png` for
global and per-applicant explanations, and `lime_default_example.html` for
the LIME cross-check.

## Fairness Audit
Largest gap in approval rate between states: **40.9%**
(full breakdown in `fairness_by_state.csv`)

## Key Takeaway
This project demonstrates a two-stage explainable pipeline: Stage 1 predicts
whether an application would be approved using only pre-decision information,
and Stage 2 — for approved loans — predicts default risk. Both stages are
explained via SHAP (and cross-checked with LIME for Stage 2), and the fairness
audit quantifies whether approval rates vary meaningfully across geography.
