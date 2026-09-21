from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


# Project directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"


# Load model and preprocessor
model = joblib.load(
    MODEL_DIR / "logistic_regression.pkl"
)

preprocessor = joblib.load(
    MODEL_DIR / "preprocessor.pkl"
)


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges
):

    # Create customer dataframe
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])


    # Preprocess input
    customer_processed = preprocessor.transform(customer)


    # Prediction
    prediction = model.predict(customer_processed)[0]


    # Probability
    probability = model.predict_proba(
        customer_processed
    )[0]


    no_churn_probability = probability[0]
    churn_probability = probability[1]


    if prediction == "Yes":
        result = "⚠️ Customer is likely to churn"
    else:
        result = "✅ Customer is likely to stay"


    return (
        result,
        f"{no_churn_probability:.2%}",
        f"{churn_probability:.2%}"
    )


# Gradio interface
with gr.Blocks(title="Customer Churn Prediction") as demo:

    gr.Markdown(
        """
        # 📊 Customer Churn Prediction

        Predict whether a telecom customer is likely to churn
        using a Machine Learning model.
        """
    )


    with gr.Row():

        with gr.Column():

            gender = gr.Dropdown(
                ["Male", "Female"],
                label="Gender",
                value="Female"
            )

            senior_citizen = gr.Dropdown(
                [0, 1],
                label="Senior Citizen",
                value=0
            )

            partner = gr.Dropdown(
                ["Yes", "No"],
                label="Partner",
                value="Yes"
            )

            dependents = gr.Dropdown(
                ["Yes", "No"],
                label="Dependents",
                value="No"
            )

            tenure = gr.Number(
                label="Tenure (Months)",
                value=5
            )

            phone_service = gr.Dropdown(
                ["Yes", "No"],
                label="Phone Service",
                value="Yes"
            )

            multiple_lines = gr.Dropdown(
                ["Yes", "No", "No phone service"],
                label="Multiple Lines",
                value="No"
            )

            internet_service = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                label="Internet Service",
                value="Fiber optic"
            )

            online_security = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Security",
                value="No"
            )

            online_backup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Online Backup",
                value="No"
            )


        with gr.Column():

            device_protection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Device Protection",
                value="No"
            )

            tech_support = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Tech Support",
                value="No"
            )

            streaming_tv = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming TV",
                value="Yes"
            )

            streaming_movies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                label="Streaming Movies",
                value="Yes"
            )

            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                label="Contract",
                value="Month-to-month"
            )

            paperless_billing = gr.Dropdown(
                ["Yes", "No"],
                label="Paperless Billing",
                value="Yes"
            )

            payment_method = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                label="Payment Method",
                value="Electronic check"
            )

            monthly_charges = gr.Number(
                label="Monthly Charges",
                value=85.5
            )

            total_charges = gr.Number(
                label="Total Charges",
                value=427.5
            )


    predict_button = gr.Button(
        "🔍 Predict Churn",
        variant="primary"
    )


    result = gr.Textbox(
        label="Prediction"
    )

    no_churn_probability = gr.Textbox(
        label="Probability of No Churn"
    )

    churn_probability = gr.Textbox(
        label="Probability of Churn"
    )


    predict_button.click(
        fn=predict_churn,
        inputs=[
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges
        ],
        outputs=[
            result,
            no_churn_probability,
            churn_probability
        ]
    )


# Launch application
demo.launch()