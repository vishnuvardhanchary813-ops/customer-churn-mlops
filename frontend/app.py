"""Streamlit form for calling the Customer Churn Prediction API."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT_SECONDS = 15


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="centered")
st.title("Customer Churn Prediction")
st.write("Enter a customer's account and service details to estimate churn risk.")
st.caption(f"Prediction service: {API_URL}")


with st.form("customer_form"):
    st.subheader("Customer information")
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
        partner = st.selectbox("Has a partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    with right:
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
        monthly_charges = st.number_input(
            "Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0
        )
        has_total_charges = st.checkbox("Enter total charges", value=True)
        total_charges = (
            st.number_input(
                "Total charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(tenure * monthly_charges),
                step=10.0,
            )
            if has_total_charges
            else None
        )

    st.subheader("Services")
    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = (
        st.selectbox("Multiple lines", ["No", "Yes"])
        if phone_service == "Yes"
        else "No phone service"
    )
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])

    service_columns = st.columns(2)
    if internet_service == "No":
        online_security = online_backup = device_protection = "No internet service"
        tech_support = streaming_tv = streaming_movies = "No internet service"
        with service_columns[0]:
            st.text_input("Online security", value="No internet service", disabled=True)
            st.text_input("Device protection", value="No internet service", disabled=True)
            st.text_input("Streaming TV", value="No internet service", disabled=True)
        with service_columns[1]:
            st.text_input("Online backup", value="No internet service", disabled=True)
            st.text_input("Tech support", value="No internet service", disabled=True)
            st.text_input("Streaming movies", value="No internet service", disabled=True)
    else:
        with service_columns[0]:
            online_security = st.selectbox("Online security", ["No", "Yes"])
            device_protection = st.selectbox("Device protection", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        with service_columns[1]:
            online_backup = st.selectbox("Online backup", ["No", "Yes"])
            tech_support = st.selectbox("Tech support", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes"])

    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)


if submitted:
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
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
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges) if total_charges is not None else None,
    }
    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f"{API_URL}/predict", json=payload, timeout=REQUEST_TIMEOUT_SECONDS
            )
        if response.status_code == 422:
            st.error(f"The API rejected one or more input fields: {response.text}")
        else:
            response.raise_for_status()
            result = response.json()
            probability = float(result["churn_probability"])
            if result["churn"]:
                st.warning("This customer is predicted to churn.")
            else:
                st.success("This customer is predicted to stay.")
            metric_col, detail_col = st.columns(2)
            metric_col.metric("Churn probability", f"{probability:.1%}")
            detail_col.metric("Prediction", result["label"])
            st.progress(max(0.0, min(probability, 1.0)), text="Estimated churn probability")
    except requests.exceptions.Timeout:
        st.error("The prediction API timed out. Try again in a moment.")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to the prediction API at {API_URL}. "
            "Start FastAPI and confirm API_URL points to it."
        )
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"The prediction API returned an error: {detail}")
    except (ValueError, KeyError) as exc:
        st.error(f"Could not read the API response: {exc}")
