import streamlit as st
import joblib
import os
import pandas as pd
import shap
import plotly.graph_objects as go
import requests


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# LOAD MODEL AND PREPROCESSOR
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR, "models", "best_xgb_model.pkl"
)

preprocessor_path = os.path.join(
    BASE_DIR, "models", "preprocessor.pkl"
)

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)


# --------------------------------------------------
# FASTAPI CONFIGURATION
# --------------------------------------------------

API_URL = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
) + "/predict"


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("📊 Customer Churn Prediction")

st.markdown(
    """
    ### Predict customer churn risk using Machine Learning

    This application uses a trained **XGBoost classification model**
    to estimate the probability that a customer will churn based on
    their demographic, service, contract, tenure, and billing information.
    """
)

st.divider()

st.info(
    "Enter the customer's details below to generate a churn-risk prediction."
)


# --------------------------------------------------
# CUSTOMER PROFILE
# --------------------------------------------------

st.subheader("👤 Customer Profile")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

with col3:
    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )


col4, col5 = st.columns(2)

with col4:
    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

with col5:
    tenure_months = st.number_input(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12,
        step=1
    )


# --------------------------------------------------
# SERVICES & INTERNET
# --------------------------------------------------

st.subheader("🌐 Services & Internet")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


col4, col5, col6 = st.columns(3)

with col4:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

with col5:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

with col6:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )


col7, col8, col9 = st.columns(3)

with col7:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

with col8:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

with col9:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )


# --------------------------------------------------
# CONTRACT & BILLING
# --------------------------------------------------

st.subheader("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col4, col5, col6 = st.columns(3)

with col4:
    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

with col5:
    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=840.0,
        step=10.0
    )

with col6:
    cltv = st.number_input(
        "Customer Lifetime Value (CLTV)",
        min_value=0.0,
        max_value=10000.0,
        value=5000.0,
        step=100.0
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

st.subheader("🔮 Churn Prediction")

predict_button = st.button(
    "🚀 Predict Churn Risk",
    type="primary",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------
    # PREPARE DATA FOR FASTAPI
    # --------------------------------------------------

    api_data = {
        "Gender": gender,
        "Senior_Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure_Months": tenure_months,
        "Phone_Service": phone_service,
        "Multiple_Lines": multiple_lines,
        "Internet_Service": internet_service,
        "Online_Security": online_security,
        "Online_Backup": online_backup,
        "Device_Protection": device_protection,
        "Tech_Support": tech_support,
        "Streaming_TV": streaming_tv,
        "Streaming_Movies": streaming_movies,
        "Contract": contract,
        "Paperless_Billing": paperless_billing,
        "Payment_Method": payment_method,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "CLTV": cltv
    }


    # --------------------------------------------------
    # CALL FASTAPI
    # --------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            json=api_data,
            timeout=10
        )


        # --------------------------------------------------
        # SUCCESSFUL RESPONSE
        # --------------------------------------------------

        if response.status_code == 200:

            result = response.json()

            churn_probability = float(
                result["churn_probability"]
            )

            churn_percentage = float(
                result["churn_probability_percentage"]
            )

            prediction = result["prediction"]

            risk_category = result["risk_category"]


            # --------------------------------------------------
            # DISPLAY RESULT
            # --------------------------------------------------

            st.success("Prediction generated successfully!")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{churn_percentage:.2f}%"
                )

            with col2:
                st.metric(
                    "Prediction",
                    prediction
                )

            with col3:
                st.metric(
                    "Risk Category",
                    risk_category
                )


            # --------------------------------------------------
            # RISK VISUALIZATION
            # --------------------------------------------------

            st.markdown("### 📈 Churn Risk Assessment")

            st.progress(
                churn_probability,
                text=f"Churn Probability: {churn_percentage:.2f}%"
            )


            # --------------------------------------------------
            # BUSINESS RECOMMENDATION
            # --------------------------------------------------

            st.markdown("### 💡 Business Recommendation")

            if risk_category == "High Risk":

                st.error(
                    "⚠️ High churn risk detected. "
                    "Consider immediate customer retention actions such as "
                    "personalized offers, contract incentives, or proactive support."
                )

            elif risk_category == "Medium Risk":

                st.warning(
                    "⚠️ Medium churn risk detected. "
                    "Consider targeted engagement, service improvements, "
                    "and personalized retention offers."
                )

            else:

                st.success(
                    "✅ Low churn risk detected. "
                    "Continue regular engagement and customer satisfaction initiatives."
                )


            # --------------------------------------------------
            # PREDICTION EXPLANATION
            # --------------------------------------------------

            st.markdown("### 🔍 Why this prediction?")

            st.caption(
                "Key customer attributes associated with the predicted churn risk."
            )

            explanation_col1, explanation_col2 = st.columns(2)

            with explanation_col1:

                st.markdown("#### 📌 Customer Profile")

                st.write(f"**Tenure:** {tenure_months} months")
                st.write(f"**Contract:** {contract}")
                st.write(f"**Dependents:** {dependents}")
                st.write(f"**Partner:** {partner}")

            with explanation_col2:

                st.markdown("#### 📌 Service & Billing")

                st.write(f"**Internet Service:** {internet_service}")
                st.write(f"**Payment Method:** {payment_method}")
                st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")
                st.write(f"**Tech Support:** {tech_support}")


            # --------------------------------------------------
            # SHAP MODEL EXPLANATION
            # --------------------------------------------------

            st.markdown("### 🧠 Model Explanation")

            st.caption(
                "SHAP identifies which customer attributes pushed the prediction "
                "toward higher or lower churn risk."
            )


            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # --------------------------------------------------
            # PREPARE INPUT FOR SHAP
            # --------------------------------------------------

            input_data = pd.DataFrame([{
                "Gender": gender,
                "Senior Citizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "Tenure Months": tenure_months,
                "Phone Service": phone_service,
                "Multiple Lines": multiple_lines,
                "Internet Service": internet_service,
                "Online Security": online_security,
                "Online Backup": online_backup,
                "Device Protection": device_protection,
                "Tech Support": tech_support,
                "Streaming TV": streaming_tv,
                "Streaming Movies": streaming_movies,
                "Contract": contract,
                "Paperless Billing": paperless_billing,
                "Payment Method": payment_method,
                "Monthly Charges": monthly_charges,
                "Total Charges": total_charges,
                "CLTV": cltv
            }])


            # Apply preprocessing for SHAP
            input_processed = preprocessor.transform(input_data)


            # Calculate SHAP values
            shap_values = explainer.shap_values(
                input_processed
            )

            # Get SHAP values for this customer
            customer_shap = shap_values[0]

            # Get processed feature names
            feature_names = preprocessor.get_feature_names_out()


            # --------------------------------------------------
            # KEEP ONLY RELEVANT FEATURES
            # --------------------------------------------------

            feature_values = input_processed[0]

            relevant_features = []

            for feature, value, shap_value in zip(
                feature_names,
                feature_values,
                customer_shap
            ):

                # Numeric features are always relevant
                if feature.startswith("num__"):

                    relevant_features.append(
                        (feature, shap_value)
                    )

                # For one-hot categorical features,
                # keep only the active category
                elif feature.startswith("cat__") and value == 1:

                    relevant_features.append(
                        (feature, shap_value)
                    )


            shap_df = pd.DataFrame(
                relevant_features,
                columns=["Feature", "SHAP Value"]
            )


            # --------------------------------------------------
            # MAKE FEATURE NAMES READABLE
            # --------------------------------------------------

            def clean_feature_name(feature):

                feature = feature.replace("num__", "")
                feature = feature.replace("cat__", "")

                if "_" in feature:

                    parts = feature.split("_", 1)

                    feature = f"{parts[0]}: {parts[1]}"

                return feature


            shap_df["Feature"] = shap_df["Feature"].apply(
                clean_feature_name
            )


            # --------------------------------------------------
            # POSITIVE / NEGATIVE IMPACT
            # --------------------------------------------------

            positive_features = (
                shap_df[shap_df["SHAP Value"] > 0]
                .sort_values("SHAP Value", ascending=False)
                .head(5)
            )

            negative_features = (
                shap_df[shap_df["SHAP Value"] < 0]
                .sort_values("SHAP Value", ascending=True)
                .head(5)
            )


            # --------------------------------------------------
            # VISUAL SHAP EXPLANATION
            # --------------------------------------------------

            display_df = pd.concat(
                [positive_features, negative_features]
            ).copy()

            display_df = display_df.sort_values(
                "SHAP Value"
            )


            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=display_df["SHAP Value"],
                    y=display_df["Feature"],
                    orientation="h",
                    marker=dict(
                        color=[
                            "#d62728" if value > 0 else "#2ca02c"
                            for value in display_df["SHAP Value"]
                        ]
                    ),
                    text=[
                        f"{value:+.3f}"
                        for value in display_df["SHAP Value"]
                    ],
                    textposition="outside",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "SHAP Impact: %{x:.3f}"
                        "<extra></extra>"
                    )
                )
            )


            fig.add_vline(
                x=0,
                line_width=1
            )


            fig.update_layout(
                title="Top Factors Influencing Churn Prediction",
                xaxis_title="SHAP Impact",
                yaxis_title="",
                height=420,
                showlegend=False,
                margin=dict(
                    l=20,
                    r=80,
                    t=60,
                    b=40
                )
            )


            st.plotly_chart(
                fig,
                use_container_width=True
            )


            st.info(
                "Positive SHAP values push the prediction toward churn, "
                "while negative values push it toward customer retention."
            )


        # --------------------------------------------------
        # API ERROR
        # --------------------------------------------------

        else:

            st.error(
                f"API request failed with status code "
                f"{response.status_code}"
            )

            try:

                error_detail = response.json()

                st.json(error_detail)

            except Exception:

                st.text(response.text)


    # --------------------------------------------------
    # CONNECTION ERROR
    # --------------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI backend. "
            "Please make sure the FastAPI server is running."
        )


    # --------------------------------------------------
    # TIMEOUT ERROR
    # --------------------------------------------------

    except requests.exceptions.Timeout:

        st.error(
            "❌ The API request timed out. "
            "Please try again."
        )


    # --------------------------------------------------
    # OTHER ERRORS
    # --------------------------------------------------

    except Exception as e:

        st.error(
            f"❌ Unexpected error: {e}"
        )