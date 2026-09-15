# 📊 Customer Churn Prediction Platform

An end-to-end Machine Learning platform that predicts customer churn probability and risk level using customer demographics, services, contract, tenure, and billing information.

The project combines **Machine Learning, Feature Selection, Explainable AI, Business Analytics, REST API development, and Cloud Deployment**.

## 🚀 Live Demo

**Streamlit App:**  
https://customer-churn-predict-model02.streamlit.app/

**FastAPI Backend:**  
https://customer-churn-prediction-project-tf60.onrender.com/

**Swagger API Docs:**  
https://customer-churn-prediction-project-tf60.onrender.com/docs

---

## 🎯 Project Objective

Customer churn is a major business problem for subscription-based companies. The objective of this project is to identify customers who are likely to churn and provide actionable risk-based retention recommendations.

The system provides:

- Churn prediction
- Churn probability
- Low / Medium / High risk classification
- SHAP-based explanations
- Business retention recommendations
- Interactive Tableau analytics
- REST API for predictions
- Cloud-hosted application

---

## 🧠 End-to-End Architecture

```text
Customer Data
     │
     ▼
EDA & Data Cleaning
     │
     ▼
Preprocessing
     │
     ▼
Feature Selection
     │
     ├──────────────┐
     ▼              ▼
Correlation        RFE
     │              │
     └──────┬───────┘
            ▼
     Model Benchmarking
            │
     ├── Decision Tree
     ├── Random Forest
     ├── SVM
     ├── XGBoost
     └── ANN / MLP
            │
            ▼
    Hyperparameter Tuning
            │
            ▼
       Tuned XGBoost
            │
       ┌────┴─────┐
       ▼          ▼
    SHAP       Predictions
       │          │
       ▼          ▼
 Explainability  Risk
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
   Streamlit             Tableau
        │
        ▼
     FastAPI
        │
        ▼
     Render
```

---

## 📂 Dataset

**Dataset:** Telco Customer Churn Dataset

- Records: **7,043 customers**
- Original features: **33**
- Target: `Churn Label`
- Churn: **26.54%**
- Non-Churn: **73.46%**

### Main Predictors

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure Months
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- CLTV

---

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

- Data type correction
- Missing-value handling
- Duplicate checking
- Target encoding
- Categorical One-Hot Encoding
- Numerical feature processing
- Stratified 80/20 train-test split
- Training-only preprocessing to prevent leakage

The following identifier, geographic and outcome-related fields were excluded:

```text
CustomerID
Count
Country
State
City
Zip Code
Lat Long
Latitude
Longitude
Churn Label
Churn Score
Churn Reason
```

> `CLTV` is retained in the current trained model because it was part of the modeling pipeline. For production use, it should only be used if it is available before prediction and does not contain future information.

---

## 🔍 Exploratory Data Analysis

EDA examined:

- Missing values
- Duplicate records
- Customer demographics
- Churn distribution
- Tenure patterns
- Contract behavior
- Payment methods
- Internet services
- Monthly and total charges

### Important Findings

- Month-to-month customers have substantially higher churn.
- Customers with shorter tenure are more likely to churn.
- Electronic-check customers show higher churn.
- Fiber-optic customers show higher observed churn.
- Long-term contracts are associated with much lower churn.

---

## 🧪 Feature Selection

Two feature-selection techniques were implemented:

### Pearson Correlation

Selected features with:

```text
|correlation| >= 0.05
```

**26 features selected**

### Recursive Feature Elimination

RFE was performed using a Decision Tree estimator.

**15 features selected**

Both approaches were performed using training data to reduce information leakage.

---

## 🤖 Machine Learning Models

The project evaluates:

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- Artificial Neural Network / MLP

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

### Cross-Validation

```text
5-Fold Stratified Cross-Validation
```

---

## 📊 Model Comparison

Performance on the full processed feature set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 73.62% | 50.38% | 52.24% | 51.25% | 66.82% |
| Random Forest | 80.18% | 65.97% | 52.51% | 58.46% | 84.22% |
| SVM | 80.84% | 68.18% | 52.51% | 59.30% | 82.61% |

XGBoost was subsequently tuned and selected as the final advanced model.

---

## 🏆 Final XGBoost Model

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**, optimizing ROC-AUC.

### Best Parameters

```python
{
    "colsample_bytree": 0.8,
    "learning_rate": 0.03,
    "max_depth": 3,
    "n_estimators": 300,
    "subsample": 0.8
}
```

**Best CV ROC-AUC:** `0.8642`

### Test Performance

| Metric | Score |
|---|---:|
| Accuracy | **80.24%** |
| Precision | **64.91%** |
| Recall | **55.88%** |
| F1 Score | **60.06%** |
| ROC-AUC | **84.81%** |
| PR-AUC | **66.27%** |

### Confusion Matrix

```text
                 Predicted
                 No     Yes

Actual No        920    113
Actual Yes       165    209
```

---

## 🔎 Explainable AI — SHAP

SHAP is integrated into the application to explain individual predictions.

### Important Global Features

- Tenure Months
- Two-year Contract
- Dependents
- Fiber Optic Internet
- One-year Contract
- Electronic Check
- Monthly Charges
- Total Charges
- Internet Service
- Online Security

The application displays the customer's top factors contributing **toward churn** and **toward retention**.

```text
Positive SHAP → pushes prediction toward Churn
Negative SHAP → pushes prediction toward Retention
```

---

## 📈 Tableau Analytics

A Tableau dashboard, **Customer Churn Intelligence Dashboard**, was created for business-level analysis.

### Dashboard Includes

- Total Customers
- Actual Churn Rate
- High-Risk Customers
- Average Churn Probability
- Churn Rate by Contract
- Churn Rate by Payment Method
- Customer Risk Distribution
- Churn Rate by Internet Service
- Churn Rate by Tenure

### Test Dataset Dashboard

```text
Customers              : 1,407
Actual Churn Rate      : 26.6%
High-Risk Customers    : 225
Average Churn Probability: 27.4%
```

### Churn by Contract

```text
Month-to-month : 41.3%
One year       : 13.8%
Two year       :  2.4%
```

### Churn by Payment Method

```text
Electronic check          : 44.8%
Bank transfer (automatic) : 19.5%
Mailed check              : 19.0%
Credit card (automatic)   : 14.7%
```

---

## 🖥️ Streamlit Application

The Streamlit frontend allows users to enter customer information and receive a real-time churn assessment.

### Features

- Customer profile inputs
- Service and internet details
- Contract and billing details
- Churn probability
- Churn prediction
- Risk classification
- Risk visualization
- Business recommendation
- Customer context
- SHAP explanation
- Interactive Plotly visualization

Example:

```text
Churn Probability : 59.97%
Prediction        : Churn
Risk Category     : Medium Risk
```

---

## ⚡ FastAPI REST API

The Machine Learning model is exposed through a FastAPI backend.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Churn prediction |
| GET | `/docs` | Swagger documentation |

### Example Request

```json
{
  "Gender": "Female",
  "Senior_Citizen": "No",
  "Partner": "Yes",
  "Dependents": "No",
  "Tenure_Months": 5,
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
  "Monthly_Charges": 90.0,
  "Total_Charges": 450.0,
  "CLTV": 5000
}
```

### Example Response

```json
{
  "prediction": "Churn",
  "churn_probability": 0.9241200113635338,
  "churn_probability_percentage": 92.42,
  "risk_category": "High Risk"
}
```

---

## ☁️ Deployment

### Frontend

**Streamlit Community Cloud**

```text
https://customer-churn-predict-model02.streamlit.app/
```

### Backend

**Render**

```text
https://customer-churn-prediction-project-tf60.onrender.com/
```

### Deployment Flow

```text
User
 │
 ▼
Streamlit Cloud
 │
 │ POST /predict
 ▼
FastAPI on Render
 │
 ▼
Preprocessor
 │
 ▼
XGBoost
 │
 ▼
Prediction + Probability + Risk
 │
 ▼
Streamlit
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| ML | Scikit-learn, XGBoost |
| Deep Learning | MLP / ANN |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| BI | Tableau |
| Frontend | Streamlit |
| Backend | FastAPI, Pydantic |
| Deployment | Render, Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

```text
Customer-Churn-Prediction-Project/
│
├── Data/
│   └── customer_churn_tableau.csv
│
├── Notebook/
│   └── ChurnPredictProject.ipynb
│
├── Streamlit App/
│   ├── assets/
│   ├── models/
│   │   ├── best_xgb_model.pkl
│   │   └── preprocessor.pkl
│   └── app.py
│
├── FastAPI/
│   └── api.py
│
├── .streamlit/
│   └── config.toml
│
├── Customer_Churn_Intelligence_Platform.twb
├── Customer_Churn_Intelligence_Platform.twbx
├── Telco_customer_churn.csv
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

---

## 💻 Run Locally

### 1. Clone

```bash
git clone https://github.com/priyanshuranjan02/Customer-Churn-Prediction-Project.git
cd Customer-Churn-Prediction-Project
```

### 2. Create Environment

```bash
python3.11 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start FastAPI

```bash
python -m uvicorn FastAPI.api:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### 5. Start Streamlit

Open another terminal:

```bash
python -m streamlit run "Streamlit App/app.py"
```

Application:

```text
http://localhost:8501
```

For local Streamlit → FastAPI communication:

```text
FASTAPI_URL=http://127.0.0.1:8000
```

For deployment:

```text
FASTAPI_URL=https://customer-churn-prediction-project-tf60.onrender.com
```

---

## 💼 Business Recommendations

The prediction system supports risk-based customer retention.

### 🔴 High Risk

- Immediate retention outreach
- Personalized offers
- Contract incentives
- Service-quality review
- Customer support intervention

### 🟠 Medium Risk

- Targeted engagement
- Personalized retention offers
- Service improvements
- Satisfaction monitoring

### 🟢 Low Risk

- Regular engagement
- Loyalty programs
- Standard satisfaction monitoring
- Cross-selling opportunities

---

## 🔮 Future Improvements

- Automated model retraining
- Model monitoring and drift detection
- Customer database integration
- Authentication and authorization
- API rate limiting
- CI/CD pipeline
- Docker deployment
- Real-time prediction pipeline
- Cost-sensitive threshold optimization
- Automated retention campaigns

---

## 👨‍💻 Author

**Priyanshu Ranjan**

B.Tech Computer Science Engineering  
Artificial Intelligence & Machine Learning

GitHub:  
https://github.com/priyanshuranjan02

---

## ⭐ Project Summary

This project demonstrates a complete Machine Learning lifecycle:

```text
EDA
 → Data Cleaning
 → Preprocessing
 → Feature Selection
 → Model Benchmarking
 → Cross-Validation
 → Hyperparameter Tuning
 → XGBoost
 → Model Evaluation
 → SHAP Explainability
 → Tableau Analytics
 → Streamlit Application
 → FastAPI REST API
 → Cloud Deployment
```

**Built as an end-to-end Customer Churn Intelligence Platform combining ML, XAI, analytics, software engineering, and deployment.**
