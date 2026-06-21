"""
app/main.py

FastAPI service that loads the trained model + preprocessing pipeline
and exposes a /predict endpoint for churn inference.

Run with:
    uvicorn app.main:app --reload --port 8000

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_engineering import add_derived_features, get_feature_columns  # noqa: E402

from app.schemas import CustomerFeatures, PredictionResponse

MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(
            "Model or preprocessor not found. Run `python src/train.py` first."
        )
    ml_models["model"] = joblib.load(MODEL_PATH)
    ml_models["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    yield
    ml_models.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_tier(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


@app.get("/")
def root():
    return {"status": "ok", "message": "Churn prediction API is running. See /docs for usage."}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_models,
        "preprocessor_loaded": "preprocessor" in ml_models,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        raw_df = pd.DataFrame([customer.model_dump()])
        df = add_derived_features(raw_df)

        numeric_cols, categorical_cols = get_feature_columns()
        X = df[numeric_cols + categorical_cols]

        X_t = ml_models["preprocessor"].transform(X)
        proba = float(ml_models["model"].predict_proba(X_t)[:, 1][0])
        pred = "Yes" if proba >= 0.5 else "No"

        return PredictionResponse(
            churn_prediction=pred,
            churn_probability=round(proba, 4),
            risk_tier=_risk_tier(proba),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
