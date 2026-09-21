from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)


# Display first 5 rows
print("First 5 rows:")
print(df.head())


# Display dataset shape
print("\nDataset Shape:")
print(df.shape)


# Display column names
print("\nColumns:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Sammary
print("\nStatistical Summary:")
print(df.describe())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values created by conversion
print("\nMissing values after converting TotalCharges:")
print(df["TotalCharges"].isnull().sum())

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Check dataset after cleaning
print("\nDataset Shape after cleaning:")
print(df.shape)
print("\nMissing Values after cleaning:")
print(df.isnull().sum())

# Remove customer ID
df = df.drop(columns=["customerID"])

# Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# Visualize churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Churn"
)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# Churn by contract type
print("\nChurn by Contract Type:")
print(
    pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100
)

# Visualize churn by contract type
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)
plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# Average tenure by churn status
print("\nAverage Tenure by Churn:")
print(
    df.groupby("Churn")["tenure"].mean()
)

# Visualize tenure distribution by churn
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="tenure"
)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()
plt.show()

# Average monthly charges by churn status
print("\nAverage Monthly Charges by Churn:")
print(
    df.groupby("Churn")["MonthlyCharges"].mean()
)

# Visualize monthly charges distribution by churn
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges"
)
plt.title("Monthly Charges Distribution by Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()
plt.show()