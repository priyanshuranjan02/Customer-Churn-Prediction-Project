# 📊 Customer Churn Prediction Platform

An end-to-end machine learning application that predicts customer churn probability, classifies risk, and provides explanations and retention recommendations. It combines model benchmarking, feature selection, XGBoost, SHAP, Tableau, FastAPI, Streamlit, and cloud deployment.

## 🚀 Live Demo

- **Streamlit App:** https://customer-churn-predict-model02.streamlit.app/
- **FastAPI Backend:** https://customer-churn-prediction-project-tf60.onrender.com/
- **Swagger API Docs:** https://customer-churn-prediction-project-tf60.onrender.com/docs

## 🎯 Project Overview

The application helps identify customers who may churn using demographic, service, contract, tenure, and billing information. It returns a churn prediction, probability, risk category, customer-specific SHAP explanation, and a suggested retention action.

## 🧠 Architecture

```text
Telco Customer Data
        │
        ▼
Cleaning, EDA & Feature Selection
        │
        ▼
Model Benchmarking + 5-Fold CV
        │
        ▼
XGBoost Hyperparameter Tuning
        │
        ▼
No-CLTV Model + Fitted Preprocessor
        │
        ├── SHAP explanations
        ├── Tableau analytics
        └── Model artifacts
                 │
                 ▼
        Streamlit Frontend
                 │ HTTPS POST /predict
                 ▼
         FastAPI on Render
                 │
                 ▼
     Prediction + Probability + Risk
```

## 📂 Dataset

**Dataset:** Telco Customer Churn (7,043 original customers and 33 columns). After converting `Total Charges` to numeric and removing rows with invalid values, **7,032 customers** remained for modeling. The target is `Churn Value`; churn prevalence in the cleaned data is approximately **26.58%**.

### Final model predictors (19)

`Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Tenure Months`, `Phone Service`, `Multiple Lines`, `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, and `Total Charges`.

### Preprocessing

- Convert `Total Charges` to numeric and remove rows with invalid values.
- Encode the target as `Churn Value` (0 = No Churn, 1 = Churn).
- Use a stratified 80/20 train-test split with `random_state=42` (5,625 train rows; 1,407 test rows).
- Standardize numeric features and one-hot encode categorical features.
- Fit preprocessing within cross-validation folds for the final model-selection experiment.
- Exclude identifiers, geographic fields, `Churn Label`, `Churn Score`, `Churn Reason`, and `CLTV` from the predictors. **CLTV is not used by the deployed model or API.**

## 🔍 EDA and Feature Selection

EDA examined missing values, duplicates, customer demographics, tenure, contract, internet services, payment method, and charges. Key observed patterns included higher churn among month-to-month customers, shorter-tenure customers, electronic-check payers, and fiber-optic customers; longer contracts were associated with lower churn.

Two feature-selection techniques were implemented on training data:

- **Pearson correlation:** absolute target correlation threshold of 0.05.
- **Recursive Feature Elimination (RFE):** Decision Tree estimator, 15 selected processed features.

## 🤖 Models and Evaluation

The project benchmarked Decision Tree, Random Forest, SVM, XGBoost, and an ANN/MLP. Metrics included accuracy, precision, recall, F1, ROC-AUC, and PR-AUC, with stratified cross-validation.

### Initial classical-model benchmark

These are the project's earlier benchmark results on the full processed feature set, before the CLTV ablation and final no-CLTV model selection.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 73.62% | 50.38% | 52.24% | 51.25% | 66.82% |
| Random Forest | 80.18% | 65.97% | 52.51% | 58.46% | 84.22% |
| SVM | 80.84% | 68.18% | 52.51% | 59.30% | 82.61% |

### Final selected model: XGBoost without CLTV

A controlled ablation compared XGBoost with and without CLTV on the **same held-out test customers**, using a stratified 80/20 split and 5-fold cross-validation. The no-CLTV variant was selected as the production candidate because it performed slightly better on this test split and removes a feature whose availability at prediction time was uncertain.

**Best parameters**

```python
{
    "colsample_bytree": 1.0,
    "learning_rate": 0.03,
    "max_depth": 4,
    "n_estimators": 200,
    "subsample": 0.8
}
```

**Best 5-fold CV ROC-AUC:** `0.8658`

### Held-out test performance

| Metric | Without-CLTV XGBoost |
|---|---:|
| Accuracy | **80.17%** |
| Precision | **64.53%** |
| Recall | **56.42%** |
| F1 Score | **60.20%** |
| ROC-AUC | **84.96%** |
| PR-AUC | **66.70%** |

**Confusion matrix** (rows = actual, columns = predicted):

```text
                 Predicted No   Predicted Yes
Actual No              917             116
Actual Yes             163             211
```

| Count | Value |
|---|---:|
| True negatives | 917 |
| False positives | 116 |
| False negatives | 163 |
| True positives | 211 |

These metrics are for the held-out split in this experiment. The with-CLTV CV-safe variant scored 84.83% ROC-AUC and 59.23% F1 on the same test rows; the without-CLTV variant scored 84.96% ROC-AUC and 60.20% F1. The differences are modest and should not be interpreted as proof of a statistically significant improvement.

## 🔎 Explainable AI — SHAP

SHAP explains individual predictions by showing which customer features push the model toward churn or toward retention.

- **Positive SHAP value:** moves the prediction toward churn.
- **Negative SHAP value:** moves the prediction toward no churn/retention.

The deployed model's explanations are calculated using the same 19 raw predictors and fitted preprocessor as the prediction API. CLTV is not included.

## 📈 Tableau Analytics

The **Customer Churn Intelligence Dashboard** includes:

- Total customers and actual churn rate
- Churn rate by contract, payment method, internet service, and tenure
- Customer risk-category distribution
- Average predicted churn probability
- High-risk customer counts

The Tableau source was built from the 1,407-customer held-out test split. Prediction-dependent risk counts and probabilities should be regenerated from the selected no-CLTV model before being interpreted as current model outputs.

## 🖥️ Streamlit Application

The Streamlit frontend collects the 19 model inputs, calls the FastAPI `/predict` endpoint, displays churn probability and risk, provides a business recommendation, and renders a customer-specific SHAP chart. The CLTV input has been removed.

Risk categories currently use these thresholds:

- **Low Risk:** probability below 30%
- **Medium Risk:** probability from 30% to below 60%
- **High Risk:** probability of 60% or higher

## ⚡ FastAPI REST API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Generate a churn prediction |
| GET | `/docs` | Interactive Swagger documentation |

### Example request (19 predictors; no CLTV)

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
  "Total_Charges": 450.0
}
```

### Example response

```json
{
  "prediction": "Churn",
  "churn_probability": 0.800042450428009,
  "churn_probability_percentage": 80.0,
  "risk_category": "High Risk"
}
```

## 🛠️ Technology Stack

| Area | Technologies |
|---|---|
| Language / data | Python, Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization / BI | Plotly, Matplotlib, Seaborn, Tableau |
| Application | Streamlit, FastAPI, Pydantic |
| Deployment | Streamlit Community Cloud, Render |
| Version control | Git, GitHub |

## 📁 Project Structure

```text
Customer-Churn-Prediction-Project/
├── Data/
│   └── customer_churn_tableau.csv
├── Notebook/
│   └── ChurnPredictProject.ipynb
├── Streamlit App/
│   ├── assets/
│   ├── models/
│   │   ├── best_xgb_model.pkl
│   │   └── preprocessor.pkl
│   └── app.py
├── FastAPI/
│   └── api.py
├── validation_results/
│   ├── cltv_ablation_metrics.csv
│   └── run_metadata.json
├── validate_cltv.py
├── .streamlit/config.toml
├── Customer_Churn_Intelligence_Platform.twb
├── Customer_Churn_Intelligence_Platform.twbx
├── Telco_customer_churn.csv
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

## 💻 Run Locally

```bash
git clone https://github.com/priyanshuranjan02/Customer-Churn-Prediction-Project.git
cd Customer-Churn-Prediction-Project
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the backend in one terminal:

```bash
python -m uvicorn FastAPI.api:app --reload
```

Start the frontend in a second terminal:

```bash
python -m streamlit run "Streamlit App/app.py"
```

Local Streamlit-to-FastAPI configuration:

```text
FASTAPI_URL=http://127.0.0.1:8000
```

For the deployed frontend, `FASTAPI_URL` is set to:

```text
https://customer-churn-prediction-project-tf60.onrender.com
```

The frontend appends `/predict` to this base URL.

## 💼 Business Use

The predictions can help prioritize outreach, personalized offers, contract incentives, and service reviews for customers with higher estimated churn risk. Model scores support prioritization; they do not guarantee that a customer will churn or that a retention action will prevent churn.

## 🔮 Potential Improvements

- Refresh Tableau predictions and risk KPIs from the no-CLTV model
- Add automated tests, logging, and API monitoring
- Add authentication and rate limiting
- Monitor performance and data drift
- Optimize probability thresholds using retention costs
- Add automated retraining and CI/CD

## 👨‍💻 Author

**Priyanshu Ranjan** — B.Tech Computer Science Engineering, AI & Machine Learning

GitHub: https://github.com/priyanshuranjan02

---

**Project summary:** An end-to-end customer churn platform combining supervised ML, a validated no-CLTV XGBoost model, SHAP explanations, Tableau analytics, a Streamlit interface, and a FastAPI prediction service.