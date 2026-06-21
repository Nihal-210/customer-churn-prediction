"""
src/train.py

Trains and compares Logistic Regression vs XGBoost on the churn dataset
using stratified k-fold cross-validation, picks the better model by
ROC-AUC, generates SHAP explainability plots, and saves the final
trained model + preprocessor for the FastAPI app.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, RocCurveDisplay, confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

from feature_engineering import (
    PROCESSED_PATH, PIPELINE_PATH, TARGET_COL, add_derived_features, get_feature_columns,
)

MODEL_DIR = "models"
REPORTS_DIR = "reports"
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    df = add_derived_features(df)
    numeric_cols, categorical_cols = get_feature_columns()
    X = df[numeric_cols + categorical_cols]
    y = df[TARGET_COL]
    return X, y, numeric_cols, categorical_cols


def cross_validate_models(X_train_t, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    results = {}
    for name, model in [("LogisticRegression", log_reg), ("XGBoost", xgb)]:
        scores = cross_val_score(model, X_train_t, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = scores
        print(f"{name}: mean ROC-AUC = {scores.mean():.4f} (+/- {scores.std():.4f})")
    return results, log_reg, xgb


def evaluate_on_test(model, X_test_t, y_test, name):
    proba = model.predict_proba(X_test_t)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    print(f"\n--- {name} | Test ROC-AUC: {auc:.4f} ---")
    print(classification_report(y_test, preds))
    return auc, proba, preds


def save_plots(y_test, results_dict, cm_model, cm_name, preprocessor, model, X_test_raw, X_test_t):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in results_dict.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name, ax=ax)
    ax.set_title("ROC Curve - Logistic Regression vs XGBoost")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/roc_curve_comparison.png", dpi=150)
    plt.close(fig)

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, cm_model)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {cm_name} (best model)")
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/confusion_matrix_best_model.png", dpi=150)
    plt.close(fig)

    # SHAP summary plot (best model)
    feature_names = preprocessor.get_feature_names_out()
    X_test_t_dense = X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t

    if isinstance(model, XGBClassifier):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_t_dense)
    else:
        background = shap.sample(X_test_t_dense, min(100, X_test_t_dense.shape[0]), random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_test_t_dense)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test_t_dense, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary_{cm_name.lower()}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plots to {REPORTS_DIR}/")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y, numeric_cols, categorical_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = joblib.load(PIPELINE_PATH) if os.path.exists(PIPELINE_PATH) else None
    if preprocessor is None:
        from feature_engineering import build_preprocessor
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(X_train)

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    print("Running 5-fold cross-validation on training set...\n")
    cv_results, log_reg, xgb = cross_validate_models(X_train_t, y_train)

    # fit both on full training set for final test evaluation
    log_reg.fit(X_train_t, y_train)
    xgb.fit(X_train_t, y_train)

    auc_lr, proba_lr, preds_lr = evaluate_on_test(log_reg, X_test_t, y_test, "LogisticRegression")
    auc_xgb, proba_xgb, preds_xgb = evaluate_on_test(xgb, X_test_t, y_test, "XGBoost")

    # pick best model by test ROC-AUC
    if auc_xgb >= auc_lr:
        best_model, best_name, best_preds = xgb, "XGBoost", preds_xgb
    else:
        best_model, best_name, best_preds = log_reg, "LogisticRegression", preds_lr

    print(f"\nBest model: {best_name} (Test ROC-AUC: {max(auc_lr, auc_xgb):.4f})")

    save_plots(
        y_test,
        {"LogisticRegression": proba_lr, "XGBoost": proba_xgb},
        best_preds, best_name, preprocessor, best_model, X_test, X_test_t,
    )

    joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
    joblib.dump(preprocessor, PIPELINE_PATH)
    with open(f"{MODEL_DIR}/model_info.txt", "w") as f:
        f.write(f"best_model={best_name}\n")
        f.write(f"test_roc_auc={max(auc_lr, auc_xgb):.4f}\n")

    print(f"\nSaved best model ({best_name}) to {MODEL_DIR}/best_model.joblib")


if __name__ == "__main__":
    main()
