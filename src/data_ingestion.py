"""
src/data_ingestion.py

Loads the raw customer churn dataset, cleans it, and writes a processed
CSV to data/processed/.

Expected raw file: data/raw/telco_churn.csv
(Telco Customer Churn dataset - download from Kaggle:
 https://www.kaggle.com/datasets/blastchar/telco-customer-churn
 and place the CSV at data/raw/telco_churn.csv)

If no raw file is found, this script generates a synthetic dataset with
the same schema so the rest of the pipeline (EDA, training, API) can be
run and demoed end-to-end without needing to download anything first.
Swap in the real Kaggle CSV whenever you're ready - no other code changes
needed since the schema matches.
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = r"D:\churn_prediction\data\processed\raw\telco_churn.csv"
PROCESSED_PATH = r"D:\churn_prediction\data\processed\churn_clean.csv"

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
TARGET_COL = "Churn"

def load_raw():
    if os.path.exists(RAW_PATH):
        print(f"Loading raw data from {RAW_PATH}")
        return pd.read_csv(RAW_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalCharges is sometimes a string with blanks for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["MonthlyCharges"] * df["tenure"]
    )

    df = df.dropna(subset=[TARGET_COL])

    if "customerID" in df.columns:
        df = df.drop_duplicates(subset=["customerID"])
    else:
        df = df.drop_duplicates()

    print("Unique values in Churn before mapping:")
    print(df[TARGET_COL].unique())

    mapping = {
        "Yes": 1,
        "No": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    }

    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .map(mapping)
    )

    if df[TARGET_COL].isna().any():
        print("Unexpected values found:")
        print(df.loc[df[TARGET_COL].isna(), TARGET_COL].unique())
        raise ValueError("Unexpected values in Churn column.")

    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df.reset_index(drop=True)




def main():
    os.makedirs("data/processed", exist_ok=True)
    df = load_raw()
    df = clean(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH} - shape={df.shape}")
    print(f"Churn rate: {df[TARGET_COL].mean():.2%}")


if __name__ == "__main__":
    main()
