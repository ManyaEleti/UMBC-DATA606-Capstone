# 1. Title and Author

## Project Title

**Evaluating the Impact of Feature Engineering on Machine Learning Model Performance**

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaojie (Jay) Wang

---

## Author Information

- **Author**: Lakshmi Manya Eleti
- **Semester**: Spring 2026
- **Program**: Master’s in Data Science
- **University**: University of Maryland, Baltimore County (UMBC)

---

## Project Links

- **GitHub Repository**  
  https://github.com/ManyaEleti/UMBC-DATA606-Capstone

- **LinkedIn Profile**  
  www.linkedin.com/in/lakshmi-manya-eleti-116142241

- **PowerPoint Presentation**  
 [Presentation file](Presentation.pdf)

- **YouTube Presentation Video**  
  Add YouTube link here

---

# 2. Background

## Project Overview

Credit card default prediction is an important problem in financial risk management.

Banks and lending institutions rely on predictive systems to identify customers who are likely to default on payments. However, many traditional machine learning models focus heavily on accuracy while failing to identify actual defaulters effectively.

This project investigates how feature engineering improves machine learning model performance in credit default prediction.

The project combines:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Behavioral feature engineering
- Machine learning modeling
- Model evaluation
- Interactive dashboard development

to improve real-world risk detection.

---

## Motivation

Credit default creates major financial challenges for banks and financial institutions.

Failure to identify high-risk customers can lead to:
- Financial losses
- Increased credit risk exposure
- Poor lending decisions
- Reduced profitability

Many classification models achieve high accuracy simply because most customers do not default. However, accuracy alone is not sufficient in imbalanced financial datasets.

This project focuses on improving:
- Risk detection
- Recall performance
- Behavioral understanding of customer repayment patterns

through engineered financial features.

---

## Research Questions

1. Can feature engineering improve machine learning model performance?
2. Which engineered behavioral features best capture repayment risk?
3. How does recall change after feature engineering?
4. Which model performs best for detecting high-risk customers?
5. Are behavioral features stronger predictors than raw financial variables?

---

# 3. Data

## Data Source

This project uses the:

**UCI Machine Learning Repository — Default of Credit Card Clients Dataset**

The dataset contains financial and demographic information for credit card customers and includes repayment behavior over multiple months.

---

## Dataset Summary

| Attribute | Value |
|---|---|
| Dataset Size | ~30,000 records |
| Features | 24 original variables |
| Time Period | 6 months repayment history |
| Problem Type | Binary Classification |

---

## Unit of Analysis

Each row in the dataset represents an individual credit card customer and includes demographic information, billing history, repayment behavior, and default outcome.

---

## Data Dictionary

| Column Name | Description |
|---|---|
| LIMIT_BAL | Credit limit amount |
| SEX | Gender |
| EDUCATION | Education level |
| MARRIAGE | Marital status |
| AGE | Customer age |
| PAY_0 to PAY_6 | Repayment status history |
| BILL_AMT1 to BILL_AMT6 | Monthly bill amounts |
| PAY_AMT1 to PAY_AMT6 | Monthly payment amounts |
| default.payment.next.month | Default label |

---

## Target Variable

The target variable for the project is:

```text
default.payment.next.month
```

Categories:
- 0 → No Default
- 1 → Default

---

## Features / Predictors

The project uses:
- Demographic information
- Billing behavior
- Payment history
- Engineered behavioral indicators

The primary focus is feature engineering.

---

# 4. Exploratory Data Analysis (EDA)

## Overview

Exploratory Data Analysis was conducted to understand:
- Data quality
- Class imbalance
- Repayment behavior
- Financial trends
- Relationships between variables

The analysis focused on identifying behavioral patterns associated with default risk.

---

## Summary Statistics & Observations

### Credit Limit
- Significant variation across customers
- Higher limits associated with different repayment behavior patterns

### Repayment Status
- Delayed payments strongly associated with default risk
- Repeated delays increased probability of default

### Billing Amounts
- Large bill amounts alone were not sufficient predictors
- Behavioral repayment trends were more informative

### Payment Amounts
- Lower payments relative to bills indicated financial stress

### Class Imbalance
- Majority of customers were non-defaulters
- Accuracy alone became misleading

---

## Correlation Analysis

The analysis revealed:
- Strong relationships between repayment delays and default risk
- Payment behavior more predictive than demographic variables
- Behavioral trends stronger than isolated monthly values

---

## Data Quality Assessment

### Missing Values
- No major missing value issues identified

### Duplicate Records
- No duplicate customer records detected

---

## Data Transformations

Preprocessing steps included:
- Standardization
- Feature scaling
- Behavioral feature engineering
- Creation of repayment trend metrics

---

## Key Findings

Customers with:
- Frequent delays
- Lower repayment consistency
- Higher accumulated unpaid balances

were more likely to default.

These findings motivated the behavioral feature engineering process.

---

# 5. Model Training

## Modeling Approach

The project focuses on understanding the impact of feature engineering on model performance.

Instead of relying only on complex algorithms, the project emphasizes transforming raw data into meaningful behavioral indicators.

---

## Feature Engineering

The following engineered features were created:

| Feature | Description |
|---|---|
| avg_bill | Average billing amount |
| avg_payment | Average payment amount |
| avg_delay | Average repayment delay |
| delay_count | Number of delayed payments |

These features summarize customer repayment behavior over time and improve model interpretability.

---

## Models Evaluated

The following machine learning models were compared:

- Logistic Regression
- Random Forest
- Balanced Logistic Regression

---

## Evaluation Metrics

The models were evaluated using:
- Accuracy
- Recall
- F1 Score

Special emphasis was placed on recall because detecting actual defaulters is critical in financial systems.

---

## Results

### Logistic Regression
- High accuracy
- Low recall

### Balanced Logistic Regression
- Significant recall improvement
- Better high-risk customer detection

### Random Forest
- Strong overall performance
- Good balance between metrics

---

## Key Insight

The results demonstrated that:
- Feature engineering improved model performance significantly
- Behavioral features were stronger predictors than raw variables
- Better features improved risk detection more than increasing model complexity

---

# 6. Application of the Trained Models

## Streamlit Dashboard

An interactive Streamlit dashboard was developed for real-time credit risk prediction.

The dashboard allows users to:
- Input customer information
- Analyze repayment behavior
- Predict default risk
- View probability-based insights
- Compare model performance

---

## Dashboard Features

Users can:
- Enter customer demographic details
- Input financial behavior indicators
- Generate real-time predictions
- View risk probabilities
- Analyze model performance charts
- Receive business-oriented recommendations

---

## Technologies Used

- Streamlit
- Plotly
- Scikit-learn
- Pandas
- NumPy

---

## Application Purpose

The dashboard transforms machine learning outputs into actionable business insights that support:
- Credit risk analysis
- Lending decisions
- Customer monitoring
- Financial risk management

---

# 7. Conclusion

## Summary

This project investigated how feature engineering impacts machine learning model performance in credit default prediction.

The analysis demonstrated that transforming raw repayment data into behavioral features significantly improves the ability to detect high-risk customers.

The project successfully combined:
- Data preprocessing
- Behavioral feature engineering
- Machine learning modeling
- Interactive dashboard deployment

into a practical financial risk prediction system.

---

## Key Outcomes

The project successfully:
- Improved recall using engineered features
- Identified important repayment behavior patterns
- Compared multiple machine learning models
- Built an interactive prediction dashboard
- Demonstrated the importance of feature engineering

---

## Limitations

Several limitations exist:
- Dataset limited to historical repayment behavior
- No real-time transaction data
- Limited external financial indicators
- Simplified deployment environment

---

## Lessons Learned

This project provided experience in:
- Feature engineering
- Data preprocessing
- Machine learning model evaluation
- Financial risk analysis
- Dashboard development
- Real-world ML deployment concepts

---

## Future Work

Potential future improvements include:
- SHAP explainability integration
- XGBoost and LightGBM experimentation
- Batch prediction support
- Real-time API deployment
- Cloud deployment
- Advanced analytics and monitoring

---

# 8. References

1. UCI Machine Learning Repository  
   https://archive.ics.uci.edu/ml/index.php

2. Streamlit Documentation  
   https://docs.streamlit.io/

3. Plotly Documentation  
   https://plotly.com/python/

4. Scikit-learn Documentation  
   https://scikit-learn.org/

5. Pandas Documentation  
   https://pandas.pydata.org/

6. NumPy Documentation  
   https://numpy.org/

7. Jupyter Notebook Documentation  
   https://jupyter.org/
