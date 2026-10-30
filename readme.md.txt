
# Customer Churn Prediction System

## Project Overview
This project predicts whether a telecom customer is likely to leave the company (churn) using machine learning. It includes data preprocessing, exploratory data analysis, model evaluation, SHAP-based explanations, and an interactive Streamlit web application.

## Objectives
- Analyze customer behavior and churn patterns.
- Clean and preprocess customer data.
- Train and evaluate machine learning models.
- Predict customer churn probability.
- Explain individual predictions using SHAP.
- Provide an interactive interface for testing customer profiles.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Customer Churn Prediction System/
├── data/
│   └── cleaned_customer_churn.csv
├── models/
│   └── xgboost_churn_pipeline.pkl
├── notebooks/
├── app.py
├── README.md
└── requirements.txt
```

## Features
- Customer churn prediction
- Estimated churn probability
- Machine learning model comparison
- SHAP feature importance
- Individual prediction explanations
- Interactive Streamlit interface

## How to Run the Project

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL: https://github.com/anurag-insights/Customer-Churn-Prediction-System-
cd "Customer Churn Prediction System"
```

### 2. Install dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the application
```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:
http://localhost:8501

## Dataset
The project uses the IBM Telco Customer Churn dataset. It contains customer demographics, subscribed services, account information, and churn labels.

## Machine Learning
The project includes Logistic Regression as a baseline and XGBoost as a more advanced model. SHAP is used to explain feature contributions to individual predictions.

## Disclaimer
Predictions are estimates based on historical data. They do not guarantee future customer behavior.

## Author
Anurag Sharma || aanuraag007@gmail.com

## Project Status
Developed as a machine learning and data science portfolio project.
