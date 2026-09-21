# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to **churn (leave the service)** based on demographic, service, contract, and billing information.

The project covers the complete machine learning workflow, from data preprocessing and exploratory data analysis to model training, evaluation, prediction, and deployment through a **Gradio web interface**.

---

## 📌 Project Overview

Customer churn prediction is an important machine learning task for identifying customers who may leave a service.

In this project, customer information such as:

* Demographics
* Contract type
* Internet and phone services
* Tenure
* Monthly charges
* Total charges
* Payment method

is used to train classification models that predict the customer's churn status.

---

## 🎯 Objectives

* Clean and preprocess the customer dataset.
* Perform Exploratory Data Analysis (EDA).
* Handle numerical and categorical features.
* Train multiple classification models.
* Evaluate model performance using classification metrics.
* Save trained models for future predictions.
* Build an interactive prediction interface using Gradio.

---

## 📊 Dataset

The project uses the **Telco Customer Churn** dataset.

### Dataset Information

* **Rows after cleaning:** 7,032
* **Features:** 19
* **Target:** `Churn`
* **Classes:** `Yes` / `No`

The dataset contains customer demographic, service, contract, and billing information.

### Target Distribution

| Churn Status | Customers | Percentage |
| ------------ | --------: | ---------: |
| No           |     5,163 |     73.42% |
| Yes          |     1,869 |     26.58% |

---

## 🔎 Exploratory Data Analysis

The EDA included:

* Missing value analysis
* Duplicate detection
* Churn distribution
* Contract type vs. churn
* Tenure vs. churn
* Monthly charges vs. churn
* Statistical analysis of numerical features

### Example Findings

Customers with different contract types showed different observed churn rates in this dataset.

For example:

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     42.71% |
| One year       |     11.28% |
| Two year       |      2.85% |

These are descriptive statistics from the dataset and do not establish causation.

---

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

1. Converting `TotalCharges` to numeric format.
2. Removing rows with invalid/missing `TotalCharges`.
3. Removing the `customerID` column.
4. Separating features and target.
5. Splitting the data into training and testing sets.
6. Standardizing numerical features using `StandardScaler`.
7. Encoding categorical features using `OneHotEncoder`.
8. Using `ColumnTransformer` to combine preprocessing steps.

### Train/Test Split

```text
Training Set: 80%
Testing Set: 20%
```

Stratified splitting was used to preserve the churn class distribution.

---

## 🤖 Machine Learning Models

Two classification models were trained and evaluated:

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model.

### 2. Random Forest

Random Forest was trained using:

* 200 estimators
* `random_state=42`
* Parallel processing with `n_jobs=-1`

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

### Logistic Regression

| Metric                  |  Score |
| ----------------------- | -----: |
| Accuracy                | 80.53% |
| Precision (Churn = Yes) | 65.15% |
| Recall (Churn = Yes)    | 57.49% |
| F1-Score (Churn = Yes)  | 61.08% |

### Random Forest

| Metric                  |  Score |
| ----------------------- | -----: |
| Accuracy                | 78.68% |
| Precision (Churn = Yes) | 62.25% |
| Recall (Churn = Yes)    | 50.27% |
| F1-Score (Churn = Yes)  | 55.62% |

On this test set, Logistic Regression produced higher scores across these reported metrics.

---

## 🧮 Confusion Matrix

### Logistic Regression

```text
                 Predicted
                 No     Yes
Actual No        918    115
Actual Yes       159    215
```

This allows us to analyze:

* True Negatives
* False Positives
* False Negatives
* True Positives

---

## 🖥️ Gradio Prediction App

The project includes an interactive **Gradio web application** that allows users to enter customer information and receive:

* Churn prediction
* Probability of staying
* Probability of churning

Example output:

```text
⚠️ Customer is likely to churn

Probability of No Churn: 23.29%
Probability of Churn: 76.71%
```

---

## 📁 Project Structure

```text
Customer_Churn_Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── preprocessor.pkl
│
├── src/
│   ├── data_analysis.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

> The dataset and virtual environment are excluded from the Git repository using `.gitignore`.

---

## 🛠️ Technologies & Libraries

### Programming Language

* Python

### Data Science & Machine Learning

* Pandas
* NumPy
* Scikit-learn

### Data Visualization

* Matplotlib
* Seaborn

### Model Persistence

* Joblib

### Deployment / Interface

* Gradio

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Noura203040/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Environment

#### Windows

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Prediction Script

```bash
python src/predict.py
```

### 6. Run the Gradio Application

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering & Preprocessing
   ↓
Train / Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Saving
   ↓
Prediction
   ↓
Gradio Web Interface
```

---

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning
* Feature importance analysis
* Handling class imbalance
* Trying additional classification algorithms
* Cross-validation
* Model comparison visualization
* Deploying the Gradio application online

---

## 👩‍💻 Author

**Noura Eslam**

Computer Science Engineering Student
E-JUST

GitHub: [Noura203040](https://github.com/Noura203040)

---

## ⭐ Project Highlights

* End-to-end Machine Learning project
* Data cleaning and EDA
* Numerical feature scaling
* Categorical feature encoding
* Multiple classification models
* Model evaluation with multiple metrics
* Saved ML models and preprocessing pipeline
* Interactive Gradio prediction interface
* GitHub-ready project structure
