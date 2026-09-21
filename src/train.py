from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"

# Create models directory if it doesn't exist
MODEL_DIR.mkdir(exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)


# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Remove customer ID
df = df.drop(columns=["customerID"])


# Separate features and target
X = df.drop(columns=["Churn"])
y = df["Churn"]


# Identify feature types
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train the model
model.fit(X_train_processed, y_train)


# Check training and testing accuracy
train_accuracy = model.score(
    X_train_processed,
    y_train
)

test_accuracy = model.score(
    X_test_processed,
    y_test
)


print("Logistic Regression trained successfully!")

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")


# Save model and preprocessor
joblib.dump(
    model,
    MODEL_DIR / "logistic_regression.pkl"
)

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)


print("\nModel saved to:")
print(MODEL_DIR / "logistic_regression.pkl")

print("\nPreprocessor saved to:")
print(MODEL_DIR / "preprocessor.pkl")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# Make predictions on test data
y_pred = model.predict(X_test_processed)


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="Yes"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="Yes"
)


print("\n--- Logistic Regression Evaluation ---")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")


# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["No", "Yes"]
)

print("\nConfusion Matrix:")
print(cm)

from sklearn.ensemble import RandomForestClassifier
# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Train Random Forest
rf_model.fit(
    X_train_processed,
    y_train
)


# Make predictions
rf_pred = rf_model.predict(X_test_processed)


# Evaluate Random Forest
rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    pos_label="Yes"
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    pos_label="Yes"
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    pos_label="Yes"
)


print("\n--- Random Forest Evaluation ---")

print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)


# Random Forest Confusion Matrix
rf_cm = confusion_matrix(
    y_test,
    rf_pred,
    labels=["No", "Yes"]
)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)

# Save Random Forest model
joblib.dump(
    rf_model,
    MODEL_DIR / "random_forest.pkl"
)

print("\nRandom Forest model saved to:")
print(MODEL_DIR / "random_forest.pkl")