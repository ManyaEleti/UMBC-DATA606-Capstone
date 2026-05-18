# Notebooks

This folder contains Jupyter notebooks used for data analysis, feature engineering, machine learning experimentation, and model evaluation for the Credit Risk Intelligence project.

## Purpose

The notebooks support:
- Exploratory Data Analysis (EDA)
- Data preprocessing and cleaning
- Behavioral feature engineering
- Machine learning model development
- Model evaluation and comparison
- Visualization and reporting

---

# Main Focus

The primary objective of the notebook workflow is to evaluate how feature engineering impacts machine learning model performance in credit default prediction.

Special emphasis is placed on:
- Behavioral pattern extraction
- Repayment trend analysis
- Recall improvement for high-risk customer detection

---

# Workflow

1. Load datasets from the `Data/` directory
2. Clean and preprocess financial data
3. Perform Exploratory Data Analysis (EDA)
4. Create engineered behavioral features
5. Train and compare machine learning models
6. Evaluate model performance using recall, accuracy, and F1 score
7. Export trained models and processed datasets

---

# Key Notebook Activities

## Exploratory Data Analysis (EDA)

Analysis includes:
- Distribution analysis
- Correlation analysis
- Default rate visualization
- Payment behavior trends
- Class imbalance investigation

---

## Feature Engineering

Behavioral features created include:
- `avg_bill`
- `avg_payment`
- `avg_delay`
- `delay_count`

These features summarize customer repayment behavior over time and improve predictive performance.

---

## Model Development

Models explored include:
- Logistic Regression
- Random Forest
- Balanced Logistic Regression

Model comparison focuses heavily on:
- Recall
- Accuracy
- F1 Score

---

# Recommended Tools

- Jupyter Notebook
- JupyterLab
- VS Code Notebook Extension
- Google Colab

---

# Purpose of This Folder

This folder serves as the experimental and analytical workspace for the project.

The notebooks document:
- The complete analytical workflow
- Feature engineering methodology
- Machine learning experimentation
- Model evaluation process
- Research findings and insights
