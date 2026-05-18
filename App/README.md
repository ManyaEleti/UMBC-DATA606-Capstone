# Application Module

This folder contains the main application source code for the Credit Risk Intelligence Dashboard.

## Contents

```bash
App/
├── app.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

---

# Description

The application provides an interactive interface for real-time credit risk prediction using machine learning.

The dashboard allows users to:
- Input customer demographic information
- Analyze repayment behavior
- Generate credit risk predictions
- Visualize model performance
- View probability-based risk insights
- Receive actionable recommendations

The application demonstrates how feature engineering can improve practical machine learning systems for financial risk analysis.

---

# Main Files

## `app.py`

The primary Streamlit application responsible for:
- User interface rendering
- Real-time prediction workflow
- Feature input collection
- Model performance visualization
- Risk analysis and recommendations

---

## `model.pkl`

Serialized machine learning model used for:
- Credit risk classification
- Default probability prediction

The deployed model is the Balanced Logistic Regression model selected based on recall performance.

---

## `scaler.pkl`

Saved preprocessing scaler used to:
- Normalize feature inputs
- Maintain consistency between training and prediction data

---

## `requirements.txt`

Contains all required Python dependencies for running the application.

---

# Dashboard Features

## Interactive Customer Profiling

Users can enter:
- Credit limit
- Age
- Gender
- Education
- Marital status

---

## Financial Behavior Analysis

Behavioral inputs include:
- Repayment status history
- Average bill amount
- Average payment amount
- Average payment delay
- Delay frequency

These engineered behavioral features are the core contribution of the project.

---

## Model Performance Visualization

The dashboard compares:
- Accuracy
- Recall
- F1 Score

across multiple machine learning models.

---

## Risk Prediction

The system generates:
- High-risk or low-risk classification
- Default probability score
- Behavioral insight summary
- Recommended business action

---

# Running the Application

From the project root:

```bash
cd App
streamlit run app.py
```

---

# Dependencies

Install required packages using:

```bash
pip install -r requirements.txt
```

---

# Technologies Used

- Streamlit
- Python
- Scikit-learn
- Plotly
- NumPy
- Joblib

---

# Purpose

The application serves as a lightweight prototype for real-world credit risk analysis systems.

It demonstrates how feature engineering and machine learning can be combined to support:
- Financial decision-making
- Risk monitoring
- Customer risk assessment
- Early default detection

---

# Future Enhancements

- SHAP explainability integration
- Batch prediction using CSV upload
- Real-time API deployment
- Cloud deployment (AWS/GCP)
- Advanced visual analytics
- User authentication system
