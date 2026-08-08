"""
Shared prediction utilities for the Palo Alto Networks Attrition Risk app.
Replicates the exact feature engineering + encoding + scaling pipeline used
in the training notebook, so live "What-If" predictions are consistent with
the saved model.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same 4 engineered features used in the training notebook."""
    df = df.copy()
    df["IncomeToExperienceRatio"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
    df["PromotionDelay"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]
    df["EngagementScore"] = (
        df["JobSatisfaction"] + df["EnvironmentSatisfaction"] +
        df["RelationshipSatisfaction"] + df["WorkLifeBalance"]
    ) / 4
    df["WorkloadStressFlag"] = (
        (df["OverTime"] == "Yes") & (df["WorkLifeBalance"] <= 2)
    ).astype(int)
    return df


def encode_and_scale(df: pd.DataFrame, label_encoders: dict, scaler) -> pd.DataFrame:
    """Apply saved label encoders + scaler to a raw (already feature-engineered) dataframe."""
    df = df.copy()
    for col, le in label_encoders.items():
        if col in df.columns:
            # handle unseen categories gracefully by mapping to the first known class
            known = set(le.classes_)
            df[col] = df[col].apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    feature_order = list(scaler.feature_names_in_)
    df = df[feature_order]
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_order, index=df.index)


def predict_risk(raw_row: pd.DataFrame, model, label_encoders: dict, scaler):
    """Full pipeline: raw employee row(s) -> attrition probability + risk category."""
    engineered = engineer_features(raw_row)
    scaled = encode_and_scale(engineered, label_encoders, scaler)
    proba = model.predict_proba(scaled)[:, 1]
    categories = pd.cut(
        proba, bins=[0, 0.30, 0.60, 1.0], labels=["Low Risk", "Medium Risk", "High Risk"], include_lowest=True
    )
    return proba, categories
