"""Streamlit app for Telco customer churn prediction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import predict_churn  # noqa: E402
from preprocess import load_pipeline  # noqa: E402


MODEL_PATH = ROOT_DIR / "models/xgboost_churn_model.pkl"
TELCO_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@st.cache_resource(show_spinner=False)
def get_model() -> Any:
    """Load the saved churn model from disk."""

    return load_pipeline(MODEL_PATH)


def customer_form() -> pd.DataFrame:
    """Render the single-customer input form and return one-row dataframe."""

    with st.form("customer_details"):
        left, middle, right = st.columns(3)

        with left:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure months", min_value=0, max_value=100, value=12)
            monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=250.0, value=75.0)

        with middle:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])

        with right:
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            max_value=20000.0,
            value=float(monthly_charges * max(tenure, 1)),
        )
        submitted = st.form_submit_button("Predict churn")

    data = pd.DataFrame(
        [
            {
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
                "TotalCharges": total_charges,
            }
        ]
    )
    data.attrs["submitted"] = submitted
    return data


def shap_contribution_chart(pipeline: Any, frame: pd.DataFrame, top_n: int = 12) -> None:
    """Render a Plotly bar chart with the largest SHAP contributions.

    Args:
        pipeline: Fitted model pipeline.
        frame: One-row customer dataframe.
        top_n: Number of feature contributions to display.
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(frame)
    feature_names = preprocessor.get_feature_names_out().tolist()
    explainer = shap.Explainer(classifier, transformed, feature_names=feature_names)
    explanation = explainer(transformed)[0]

    values = np.asarray(explanation.values, dtype=float)
    contribution_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": values,
            "abs_contribution": np.abs(values),
        }
    )
    top_features = (
        contribution_frame.sort_values("abs_contribution", ascending=False)
        .head(top_n)
        .sort_values("contribution")
    )
    fig = px.bar(
        top_features,
        x="contribution",
        y="feature",
        orientation="h",
        color="contribution",
        color_continuous_scale=["#2166ac", "#f7f7f7", "#b2182b"],
        title="Top SHAP feature contributions",
    )
    fig.update_layout(coloraxis_showscale=False, height=460, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_results(pipeline: Any, frame: pd.DataFrame) -> None:
    """Render churn prediction metrics and SHAP chart for one customer."""

    prediction = predict_churn(pipeline, frame)
    probability = float(prediction.loc[0, "churn_probability"])
    label = str(prediction.loc[0, "predicted_label"])

    col_a, col_b = st.columns(2)
    col_a.metric("Churn probability", f"{probability:.1%}")
    col_b.metric("Prediction", label)
    shap_contribution_chart(pipeline, frame)


def main() -> None:
    """Run the Streamlit app."""

    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    st.title("Customer Churn Prediction")

    if not MODEL_PATH.exists():
        st.error(
            "Model file not found. Train the model with "
            "`python src/train.py --data-path data/WA_Fn-UseC_-Telco-Customer-Churn.csv`."
        )
        st.stop()

    pipeline = get_model()
    tab_upload, tab_single = st.tabs(["CSV upload", "Single customer"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload a customer CSV", type=["csv"])
        if uploaded_file is not None:
            batch = pd.read_csv(uploaded_file)
            predictions = predict_churn(pipeline, batch)
            st.dataframe(predictions, use_container_width=True)
            st.download_button(
                "Download predictions",
                predictions.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

    with tab_single:
        frame = customer_form()
        if bool(frame.attrs.get("submitted")):
            render_prediction_results(pipeline, frame[TELCO_COLUMNS])


if __name__ == "__main__":
    main()
