"""
tests/test_api.py

Simple smoke test that sends sample requests to the running FastAPI
service and prints the responses. Not a unit test framework - just a
quick way to confirm the API works end-to-end before a demo/interview.

Usage:
    1. In one terminal: uvicorn app.main:app --reload --port 8000
    2. In another terminal: python tests/test_api.py
"""

import requests

BASE_URL = "http://127.0.0.1:8000"

HIGH_RISK_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 190.0,
}

LOW_RISK_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0,
}


def check_health():
    resp = requests.get(f"{BASE_URL}/health")
    print("GET /health ->", resp.status_code, resp.json())
    assert resp.status_code == 200


def check_predict(name, payload):
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"\nPOST /predict ({name}) -> {resp.status_code}")
    print(resp.json())
    assert resp.status_code == 200


def check_invalid_payload():
    bad_payload = dict(HIGH_RISK_CUSTOMER)
    bad_payload["Contract"] = "Lifetime"  # not a valid Literal value
    resp = requests.post(f"{BASE_URL}/predict", json=bad_payload)
    print(f"\nPOST /predict (invalid Contract value) -> {resp.status_code}")
    print(resp.json())
    assert resp.status_code == 422  # FastAPI/Pydantic validation error


if __name__ == "__main__":
    check_health()
    check_predict("likely high risk", HIGH_RISK_CUSTOMER)
    check_predict("likely low risk", LOW_RISK_CUSTOMER)
    check_invalid_payload()
    print("\nAll smoke tests passed.")
