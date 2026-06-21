# Customer Churn Prediction

End-to-end churn prediction project: data cleaning, EDA, feature engineering,
model comparison (Logistic Regression vs XGBoost) with cross-validation,
SHAP explainability, and a FastAPI service for real-time inference.

## Project structure

```
churn_prediction/
├── data/
│   ├── raw/              # place telco_churn.csv here (see Dataset section)
│   └── processed/        # cleaned output from data_ingestion.py
├── notebooks/
│   └── eda.ipynb          # exploratory data analysis
├── src/
│   ├── data_ingestion.py  # load + clean raw data
│   ├── feature_engineering.py  # derived features + preprocessing pipeline
│   └── train.py           # CV, model comparison, SHAP plots, model saving
├── app/
│   ├── main.py             # FastAPI app with /predict endpoint
│   └── schemas.py          # Pydantic request/response models
├── tests/
│   └── test_api.py         # smoke test script for the running API
├── models/                 # saved model + preprocessor (generated)
├── reports/                # ROC curve, confusion matrix, SHAP plots (generated)
└── requirements.txt
```

## Dataset

This project is built around the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Download it and place the CSV at `data/raw/telco_churn.csv`.



## Run the pipeline

```bash
# 1. Clean the raw data
python src/data_ingestion.py

# 2. Fit and save the preprocessing pipeline
python src/feature_engineering.py

# 3. Train, compare models, generate SHAP/ROC/confusion matrix plots
python src/train.py
```

This produces:
- `models/best_model.joblib` — the better-performing model (by test ROC-AUC)
- `models/preprocessor.joblib` — fitted preprocessing pipeline
- `reports/roc_curve_comparison.png`
- `reports/confusion_matrix_best_model.png`
- `reports/shap_summary_<model>.png`

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI, or test it
with the included script:

```bash
python tests/test_api.py
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.5, "TotalCharges": 420.0
  }'
```

Response:

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.78,
  "risk_tier": "High"
}
```

