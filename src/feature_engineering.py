"""
src/feature_engineering.py

Transforms the cleaned dataset into a model-ready feature matrix.
Saves the fitted preprocessing pipeline (encoders/scaler) so the exact
same transformation can be reused at inference time in the FastAPI app.
"""

import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_PATH = r"D:\churn_prediction\data\processed\churn_clean.csv"
PIPELINE_PATH = r"D:\churn_prediction\models\preprocessor.joblib"

TARGET_COL = "Churn"
ID_COLS = ["customerID"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # avg monthly spend so far - captures pricing pressure independent of tenure
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    # tenure buckets often carry signal that raw tenure (linear) misses
    df["TenureBucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"]
    ).astype(str)
    # flag customers with no add-on services at all (low engagement signal)
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["NumAddonServices"] = (df[addon_cols] == "Yes").sum(axis=1)
    return df


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def get_feature_columns():
    numeric = NUMERIC_FEATURES + ["AvgMonthlySpend", "NumAddonServices"]
    categorical = CATEGORICAL_FEATURES + ["TenureBucket"]
    return numeric, categorical


def main():
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(PROCESSED_PATH)
    df = add_derived_features(df)

    numeric_cols, categorical_cols = get_feature_columns()
    X = df[numeric_cols + categorical_cols]
    y = df[TARGET_COL]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X)

    joblib.dump(preprocessor, PIPELINE_PATH)
    print(f"Saved fitted preprocessor to {PIPELINE_PATH}")
    print(f"Feature matrix shape after transform: {preprocessor.transform(X).shape}")


if __name__ == "__main__":
    main()
