from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "Streamlit App" / "models"

MODEL_PATH = MODEL_DIR / "best_xgb_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"


# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using Machine Learning",
    version="1.0.0"
)

# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class CustomerData(BaseModel):
    Gender: Literal["Female", "Male"]
    
    Senior_Citizen: Literal["No", "Yes"]
    
    Partner: Literal["No", "Yes"]
    
    Dependents: Literal["No", "Yes"]

    Tenure_Months: float = Field(
        ...,
        ge=0,
        le=72,
        description="Customer tenure in months"
    )

    Phone_Service: Literal["Yes", "No"]

    Multiple_Lines: Literal[
        "No",
        "Yes",
        "No phone service"
    ]

    Internet_Service: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    Online_Security: Literal[
        "No",
        "Yes",
        "No internet service"
    ]

    Online_Backup: Literal[
        "No",
        "Yes",
        "No internet service"
    ]

    Device_Protection: Literal[
        "No",
        "Yes",
        "No internet service"
    ]

    Tech_Support: Literal[
        "No",
        "Yes",
        "No internet service"
    ]

    Streaming_TV: Literal[
        "No",
        "Yes",
        "No internet service"
    ]

    Streaming_Movies: Literal[
        "No",
        "Yes",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    Paperless_Billing: Literal["Yes", "No"]

    Payment_Method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    Monthly_Charges: float = Field(
        ...,
        ge=0,
        le=200,
        description="Monthly charges"
    )

    Total_Charges: float = Field(
        ...,
        ge=0,
        le=10000,
        description="Total charges"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Gender": "Female",
                "Senior_Citizen": "No",
                "Partner": "Yes",
                "Dependents": "No",
                "Tenure_Months": 12,
                "Phone_Service": "Yes",
                "Multiple_Lines": "No",
                "Internet_Service": "Fiber optic",
                "Online_Security": "No",
                "Online_Backup": "No",
                "Device_Protection": "No",
                "Tech_Support": "No",
                "Streaming_TV": "Yes",
                "Streaming_Movies": "Yes",
                "Contract": "Month-to-month",
                "Paperless_Billing": "Yes",
                "Payment_Method": "Electronic check",
                "Monthly_Charges": 85.0,
                "Total_Charges": 1020.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction: Literal["Churn", "No Churn"]

    churn_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of customer churn"
    )

    churn_probability_percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Churn probability as a percentage"
    )

    risk_category: Literal[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]

# --------------------------------------------------
# Home Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "status": "success"
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict_churn(customer: CustomerData):
    # Convert API input into DataFrame
    input_data = pd.DataFrame([{
        "Gender": customer.Gender,
        "Senior Citizen": customer.Senior_Citizen,
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "Tenure Months": customer.Tenure_Months,
        "Phone Service": customer.Phone_Service,
        "Multiple Lines": customer.Multiple_Lines,
        "Internet Service": customer.Internet_Service,
        "Online Security": customer.Online_Security,
        "Online Backup": customer.Online_Backup,
        "Device Protection": customer.Device_Protection,
        "Tech Support": customer.Tech_Support,
        "Streaming TV": customer.Streaming_TV,
        "Streaming Movies": customer.Streaming_Movies,
        "Contract": customer.Contract,
        "Paperless Billing": customer.Paperless_Billing,
        "Payment Method": customer.Payment_Method,
        "Monthly Charges": customer.Monthly_Charges,
        "Total Charges": customer.Total_Charges
    }])


    # Apply the same preprocessing used during training
    input_processed = preprocessor.transform(input_data)


    # Generate prediction
    prediction = model.predict(input_processed)[0]

    # Generate probability
    churn_probability = float(
        model.predict_proba(input_processed)[0, 1]
    )

    churn_percentage = round(churn_probability * 100, 2)


    # Determine risk category
    if churn_probability < 0.30:
        risk_category = "Low Risk"

    elif churn_probability < 0.60:
        risk_category = "Medium Risk"

    else:
        risk_category = "High Risk"


    # Final response
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": churn_probability,
        "churn_probability_percentage": churn_percentage,
        "risk_category": risk_category
    }