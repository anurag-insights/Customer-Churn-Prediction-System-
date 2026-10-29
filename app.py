
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

import shap
import matplotlib.pyplot as plt
import numpy as np
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load saved model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_churn_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer():
    loaded_model = load_model()
    classifier = loaded_model.named_steps["classifier"]
    return shap.TreeExplainer(classifier)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# -----------------------------
# Application header
# -----------------------------
st.title("📊 Customer Churn Prediction System")
st.write(
    "Predict whether a telecom customer is likely to leave "
    "using a trained XGBoost machine learning model."
)

st.info(
    "Enter the customer's information below and click "
    "'Predict Churn' to generate a prediction."
)

# -----------------------------
# Customer input form
# -----------------------------
with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox(
            "Has Partner?", ["Yes", "No"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["Yes", "No"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=100000.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )


# -----------------------------
# Prediction + SHAP Explanation
# -----------------------------
if submitted:

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

    if tenure == 0:
        customer["TotalCharges"] = 0.0

    probability = model.predict_proba(customer)[0, 1]
    prediction = int(model.predict(customer)[0])

    st.divider()
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Likely to Churn")
        else:
            st.success("✅ Likely to Stay")

    with col2:
        st.metric(
            "Estimated Churn Probability",
            f"{probability:.1%}"
        )

    st.progress(float(probability))

    st.caption(
        "This probability is a model estimate, not a guarantee "
        "of future customer behavior."
    )

    st.divider()
    st.header("Why did the model make this prediction?")
    st.write(
        "SHAP estimates how each feature contributes to the "
        "model's prediction for this customer."
    )

    try:
        explainer = load_explainer()
        preprocessor = model.named_steps["preprocessor"]

        transformed_customer = preprocessor.transform(customer)

        if hasattr(transformed_customer, "toarray"):
            transformed_customer = transformed_customer.toarray()

        feature_names = preprocessor.get_feature_names_out()

        transformed_df = pd.DataFrame(
            transformed_customer,
            columns=feature_names
        )

        customer_shap = explainer(transformed_df)
        shap_scores = np.asarray(
            customer_shap.values[0]
        ).reshape(-1)

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_scores,
            "Absolute Impact": np.abs(shap_scores)
        })

        importance_df["Feature"] = (
            importance_df["Feature"]
            .str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False)
        )

        importance_df = importance_df.sort_values(
            "Absolute Impact",
            ascending=False
        ).head(10).sort_values("SHAP Value")

        st.subheader("Top 10 Influencing Features")
        st.caption(
            "Positive values push the model output toward churn; "
            "negative values push it away from churn. These values "
            "are on the model's raw output scale, not percentage points."
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        bar_colors = [
            "tomato" if value > 0 else "steelblue"
            for value in importance_df["SHAP Value"]
        ]

        ax.barh(
            importance_df["Feature"],
            importance_df["SHAP Value"],
            color=bar_colors
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (raw model output)")
        ax.set_title("Features Influencing This Customer's Prediction")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Individual Prediction Explanation")
        st.write(
            "The waterfall plot shows how feature contributions "
            "move the model output from its baseline to this customer's "
            "prediction."
        )

        shap.plots.waterfall(
            customer_shap[0],
            max_display=12,
            show=False
        )

        waterfall_fig = plt.gcf()
        st.pyplot(waterfall_fig)
        plt.close(waterfall_fig)

        st.subheader("Feature Contribution Details")

        st.dataframe(
            importance_df[["Feature", "SHAP Value"]].sort_values(
                "SHAP Value",
                key=lambda series: series.abs(),
                ascending=False
            ).round(4),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(
            "The prediction succeeded, but the SHAP explanation "
            "could not be generated."
        )
        st.exception(e)

