# PlaNetworks-Attribution-Dashboard
# Palo Alto Networks — Employee Attrition Prediction & Risk Scoring

**Machine Learning-Based Employee Attrition Prediction and Risk Scoring System**

An end-to-end predictive HR analytics project that scores every employee's attrition risk, explains
*why* in plain language, and lets HR teams simulate retention interventions before committing to
them — built as part of the Unified Mentor Data Analyst Internship program.

**Live Dashboard:** https://planetworks-attribution-dashboard-sandeep-reddy-m.streamlit.app/

---

## Overview

Palo Alto Networks' HR leaders lacked a systematic way to answer a simple but critical question:
*which specific employees are most likely to leave in the near future?* This project builds and
deploys a validated machine learning system that answers it — moving HR analytics from descriptive
reporting to predictive decision intelligence.

The project delivers three artifacts:

| Deliverable | Description |
|---|---|
| **Streamlit Dashboard** | Interactive dashboard with 6 modules — Executive Overview, Attrition Risk Dashboard, Employee Risk Profile, Department-Level Risk View, Explainability Panel (with live What-If scenarios), and Model Performance |
| **Jupyter Notebook** | Fully executed data science pipeline: EDA, feature engineering, model training (3 models), held-out test-set evaluation, risk scoring, and SHAP explainability |
| **Research Paper** | Formal write-up with embedded dashboard findings, insights, and recommendations, including an executive summary for HR/government stakeholders |

## Key Results

- **1,470 employees** analyzed across 31 fields; **16.1%** historical attrition rate
- Three models trained and validated on a held-out test set (never touched during training or SMOTE
  resampling): **Logistic Regression, Random Forest, XGBoost**
- Deployed model: **Logistic Regression** — selected for Recall (76.6%) and ROC-AUC (0.807), the two
  metrics that matter most for a retention use case, over higher-accuracy tree-based alternatives that
  missed far more true leavers
- **OverTime** is the single strongest predictor of attrition, confirmed by both model coefficients and
  SHAP analysis — employees working overtime show roughly a 25-percentage-point higher predicted risk
- **347 employees (23.6%)** currently flagged High Risk under the default 60% threshold

## Dashboard Modules

1. **Executive Overview** — 8 animated, interactive KPI cards with live breakdown popovers, automated
   business recommendations generated from current data, and downloadable CSV/summary reports
2. **Attrition Risk Dashboard** — overall risk distribution, high-risk counts segmented by department,
   role, gender, or marital status, and risk vs. key-driver comparisons
3. **Employee Risk Profile** — individual attrition probability lookup by Employee ID, with SHAP-based
   individual reason codes explaining each employee's specific risk drivers
4. **Department-Level Risk View** — aggregated risk by department and role, including a risk heatmap
   and a full department summary table
5. **Explainability Panel** — global feature importance, live SHAP impact distribution, and a **What-If
   Scenario tool** that re-scores a live prediction as you adjust an employee's profile
6. **Model Performance** — full validation metrics (Accuracy, Precision, Recall, F1, ROC-AUC) computed
   on the held-out test set, with plain-language metric definitions and risk-category thresholds

**User capabilities:** Department & role filters, an adjustable risk-threshold slider (recalculates
every chart and KPI live), and an Employee ID selector.

## Repository Structure

```
PlaNetworks-Attribution-Dashboard/
│
├── data/                                 ← shared by both the notebook and the app
│   ├── Palo_Alto_Networks.csv            (raw dataset)
│   ├── employees_scored.csv              (dataset + model predictions)
│   ├── model_comparison.csv              (validation metrics for all 3 models)
│   ├── best_model.pkl                    (trained, deployed model)
│   ├── scaler.pkl                        (fitted StandardScaler)
│   └── label_encoders.pkl                (fitted categorical encoders)
│
├── notebook/
│   └── PaloAlto_Attrition_Prediction.ipynb
│
├── app/
│   ├── app.py
│   ├── predict_utils.py                  (shared preprocessing/prediction pipeline)
│   ├── requirements.txt
│   └── assets/
│       ├── hero_banner.jpg
│       ├── logo.png
│       └── mark.png
│
└── README.md
```

## Local Setup

**Notebook:**
```bash
cd notebook
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn joblib
jupyter notebook PaloAlto_Attrition_Prediction.ipynb
```

**Dashboard:**
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The app loads `best_model.pkl`, `scaler.pkl`, and `label_encoders.pkl` directly rather than retraining
on startup. If you retrain the model in the notebook, re-export those three files to `data/` to keep
the app in sync.

## Dataset

The IBM HR Analytics-style dataset (`Palo_Alto_Networks.csv`) contains 1,470 employee records across
31 fields — demographics, compensation, tenure, satisfaction survey scores, and role information. No
missing values, no duplicates. Full field-by-field documentation is in the research paper (Section 3)
and the notebook's opening cells.

## Methodology

1. **Data Preprocessing** — label encoding, StandardScaler feature scaling, SMOTE for class imbalance
   (training set only), stratified 80/20 train-test split
2. **Feature Engineering** — 4 engineered features: IncomeToExperienceRatio, PromotionDelay,
   EngagementScore, WorkloadStressFlag
3. **Model Development** — Logistic Regression (baseline), Random Forest, XGBoost
4. **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, all computed on a held-out
   test set the models never saw during training
5. **Risk Scoring** — every employee gets an Attrition Probability (0–1) and a Risk Category (Low
   <30%, Medium 30–60%, High >60%), with the High-Risk threshold adjustable live in the dashboard
6. **Explainability** — global feature importance, SHAP values, and individual-level reason codes for
   every prediction

## Known Issues

- The **"Model ROC-AUC" KPI card** on the Executive Overview tab currently displays 0.0% on some
  deployments instead of the correct 0.807. This is caused by `model_comparison.csv` not resolving at
  the expected file path — double-check that your deployed repository's `data/` folder sits exactly
  one directory above `app/app.py`, matching the structure above. The model's true, validated
  performance (unaffected by this display bug) is documented in the research paper, Section 7.

## Links

- **Live Dashboard:** https://planetworks-attribution-dashboard-sandeep-reddy-m.streamlit.app/
- **GitHub Repository:** https://github.com/mSReddy46/PlaNetworks-Attribution-Dashboard

---

* Prepared by M. Sandeep Reddy *
