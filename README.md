# Customer Churn Prediction

End-to-end churn prediction project: data cleaning, EDA, feature engineering,
model comparison (Logistic Regression vs XGBoost) with cross-validation,
SHAP explainability, and a FastAPI service for real-time inference.

## Dataset

This project is built around the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Download it and place the CSV at `data/raw/telco_churn.csv`.


## 🛠️ Tech Stack

**Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, FastAPI, Pydantic, Matplotlib**

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



## 🚀 Highlights

* Built a complete ML pipeline covering data cleaning, feature engineering, model training, explainability, and deployment.
* Compared Logistic Regression and XGBoost using 5-fold cross-validation and achieved **84.51% ROC-AUC**.
* Used **SHAP** to identify key churn drivers such as contract type and tenure.
* Deployed the model using **FastAPI** with **Pydantic** validation for real-time predictions.




## ▶️ Run the Project

```bash
python src/data_ingestion.py
python src/feature_engineering.py
python src/train.py
```

Start API:

```bash
uvicorn app.main:app --reload
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

## 📈 Results

| Metric         | Value                   |
| -------------- | ----------------------- |
| Best Model     | Logistic Regression     |
| ROC-AUC        | 84.51%                  |
| Validation     | 5-Fold Cross Validation |
| Explainability | SHAP                    |

## 🔮 Future Improvements

* MLflow Experiment Tracking
* Docker Deployment
* Hyperparameter Tuning
* Model Monitoring & Drift Detection
