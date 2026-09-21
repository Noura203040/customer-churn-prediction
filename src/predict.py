from pathlib import Path

import joblib
import pandas as pd


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# Load model and preprocessor
model = joblib.load(
    MODEL_DIR / "logistic_regression.pkl"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)


# Example customer
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
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
    "MonthlyCharges": 85.5,
    "TotalCharges": 427.5
}])


# Preprocess customer data
customer_processed = preprocessor.transform(customer)


# Make prediction
prediction = model.predict(customer_processed)[0]


# Get probability
probability = model.predict_proba(customer_processed)[0]


# Display result
print("\n--- Customer Churn Prediction ---")

print(f"Prediction: {prediction}")

print(f"Probability of No Churn: {probability[0]:.2%}")
print(f"Probability of Churn: {probability[1]:.2%}")
