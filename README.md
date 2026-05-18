# UMBC-DATA606 Capstone

# Evaluating the Impact of Feature Engineering on Machine Learning Model Performance

This repository contains the capstone project for DATA 606 – Capstone in Data Science at the University of Maryland, Baltimore County (UMBC).

The project focuses on understanding how feature engineering improves machine learning performance in credit default risk prediction using behavioral financial indicators and repayment patterns.

An interactive Streamlit dashboard was also developed to demonstrate real-time credit risk analysis and prediction.

---

# Project Overview

Credit card default prediction is an important problem in financial risk management.

Traditional machine learning models often achieve high accuracy while still failing to identify actual defaulters due to class imbalance and weak behavioral representation.

This project focuses on:

- Transforming raw financial data into meaningful behavioral features
- Improving recall for high-risk customer detection
- Comparing machine learning model performance
- Building an interactive real-time prediction dashboard

The project demonstrates how feature engineering can significantly improve the practical usefulness of machine learning systems.

---

# Repository Structure

```text
UMBC-DATA606-Capstone/
│
├── App/
│   ├── app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── requirements.txt
│   └── README.md
│
├── Data/
│   ├── default of credit card clients.xls
│   ├── credit_clean.csv
│   ├── credit_feature_engineered.csv
│   └── README.md
│
├── Docs/
│   ├── project_report.md
│   ├── Resume.md
│   ├── Headshot.jpg
│   └── README.md
│
├── Notebooks/
│   ├── 01_data_cleaning_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering_modeling.ipynb
│   ├── 04_streamlit_dashboard.ipynb
│   └── README.md
│
├── requirements.txt
└── README.md
```

---

# Project Objectives

The primary objectives of this project are to:

- Analyze customer credit default behavior
- Perform data preprocessing and cleaning
- Engineer behavioral financial features
- Improve machine learning model performance
- Increase recall for high-risk customer detection
- Compare multiple machine learning models
- Develop a real-time Streamlit dashboard

---

# Dataset

The project uses the:

## UCI Credit Card Default Dataset

Dataset Characteristics:
- ~30,000 customer records
- Financial and demographic information
- Billing and repayment history
- Credit default labels

Target Variable:
- `default.payment.next.month`
  - 0 → No Default
  - 1 → Default

---

# Methodology

The project follows a structured end-to-end data science workflow.

---

# 1. Data Cleaning & Preprocessing

📁 `Notebooks/01_data_cleaning_preprocessing.ipynb`

Tasks performed:
- Loaded and cleaned raw financial data
- Checked missing values
- Standardized variables
- Processed repayment history features
- Prepared datasets for analysis

Output:
- `credit_clean.csv`

---

# 2. Exploratory Data Analysis (EDA)

📁 `Notebooks/02_exploratory_data_analysis.ipynb`

Analysis included:
- Distribution analysis
- Correlation analysis
- Default behavior trends
- Payment delay patterns
- Class imbalance investigation
- Financial behavior visualization

---

# 3. Feature Engineering & Modeling

📁 `Notebooks/03_feature_engineering_modeling.ipynb`

The core contribution of this project is feature engineering.

Behavioral features created:
- `avg_bill`
- `avg_payment`
- `avg_delay`
- `delay_count`

These features summarize customer repayment behavior over time and improve the model’s ability to identify risky customers.

Models evaluated:
- Logistic Regression
- Random Forest
- Balanced Logistic Regression

Evaluation Metrics:
- Accuracy
- Recall
- F1 Score

Special emphasis was placed on recall because detecting actual defaulters is critical in financial systems.

---

# 4. Dashboard Development

📁 `Notebooks/04_streamlit_dashboard.ipynb`
📁 `App/app.py`

An interactive Streamlit dashboard was developed for real-time credit risk prediction.

Dashboard Features:
- Customer profile input
- Financial behavior analysis
- Real-time risk prediction
- Default probability scoring
- Model comparison visualization
- Risk insights and recommendations

Output:
- Interactive Credit Risk Dashboard

---

# Streamlit Dashboard

The Streamlit application allows users to:

- Input customer financial information
- Analyze repayment behavior
- Predict default risk
- View probability-based risk scores
- Compare machine learning model performance
- Generate business-oriented recommendations

The dashboard demonstrates how machine learning and feature engineering can support financial decision-making systems.

---

# Technologies Used

| Category | Tools & Libraries |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Development | Jupyter Notebook |

---

# Key Findings

The project demonstrated that:

- Feature engineering significantly improved model performance
- Behavioral features were stronger predictors than raw variables
- Balanced Logistic Regression achieved the highest recall
- Accuracy alone was insufficient for evaluating credit risk models
- Better feature representation improved real-world risk detection

The results show that engineered behavioral features can outperform simply using more complex machine learning models.

---

# How to Run the Project

## 1. Clone the Repository

```bash
git clone <repository-link>
cd <repository-name>
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Run the Streamlit Application

```bash
cd App
streamlit run app.py
```

---

# Future Improvements

Potential future enhancements include:

- SHAP explainability integration
- XGBoost and LightGBM experimentation
- Batch prediction support
- Real-time API deployment
- Cloud deployment (AWS/GCP)
- Advanced dashboard analytics

---

# Author

Lakshmi Manya Eleti  
Graduate Student – Data Science

University of Maryland, Baltimore County (UMBC)
