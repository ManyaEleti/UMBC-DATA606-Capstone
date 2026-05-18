# Data Directory

This folder contains the datasets and supporting files used in the Credit Risk Intelligence Dashboard project.

## Contents

```bash
Data/
├── credit_clean.csv
├── credit_feature_engineered.csv
├── default of credit card clients.xls
└── README.md
```

---

# Dataset Description

The datasets in this folder are used for:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model training
- Model evaluation
- Dashboard prediction workflows

The project is based on the UCI Credit Card Default Dataset, which contains real-world credit card customer information and repayment behavior.

---

# File Descriptions

| File | Description |
|---|---|
| `default of credit card clients.xls` | Original raw dataset from the UCI Machine Learning Repository |
| `credit_clean.csv` | Cleaned and preprocessed dataset after handling transformations |
| `credit_feature_engineered.csv` | Final dataset containing engineered behavioral features used for modeling |
| `README.md` | Documentation overview for this folder |

---

# Dataset Information

The dataset contains:
- Customer demographic information
- Credit limit details
- Billing statement history
- Payment history
- Repayment status records
- Default labels

---

# Feature Engineering

The main focus of this project is feature engineering.

Behavioral features created include:
- `avg_bill`
- `avg_payment`
- `avg_delay`
- `delay_count`

These engineered features summarize customer financial behavior and improve the model’s ability to identify high-risk customers.

---

# Usage

The datasets are used throughout the project pipeline for:
- Cleaning and preprocessing
- Exploratory analysis
- Behavioral feature construction
- Model training
- Prediction and evaluation
- Streamlit dashboard integration

---

# Data Source

Dataset Source:

UCI Machine Learning Repository — Default of Credit Card Clients Dataset

The dataset represents real-world credit card customer behavior and default outcomes.

---

# Purpose

This folder serves as the central storage location for all datasets used in the project workflow.

The data supports:
- Academic research
- Machine learning experimentation
- Feature engineering analysis
- Credit risk prediction modeling
- Interactive dashboard deployment
