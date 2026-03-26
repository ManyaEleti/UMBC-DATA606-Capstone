
## 1. Title and Author

- Project Title -  **Evaluating the Impact of Feature Engineering on Machine Learning Model Performance**
- Author - Lakshmi Manya Eleti
- Semester - Spring'26
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Link to the author's GitHub repo of the project: https://github.com/ManyaEleti/UMBC-DATA606-Capstone
    
---

## 2. Background

### What is it about?

The proposed project aims at learning about the potential impact of data preparation and transformation on the performance of machine learning models. The project does not focus on complicated algorithms, but instead focuses on how different types of features created from raw data affect the prediction ability of models.

The project uses real credit card data that includes billing details, payments, and repayment behavior over time.

Different versions of features are created:

- raw features  
- aggregated features  
- behavioral features  

These are compared to understand how they affect:

- model accuracy  
- model stability  
- model interpretability  

The main idea is to show that **good feature engineering improves model performance**, while poor or excessive feature creation can reduce model reliability.

---

### Machine Learning Models

The models used in this project are:

- **Logistic Regression**  
- **Random Forest**  

These models are simple, widely used, and easy to understand. The focus of this project is not comparing algorithms, but understanding the role of features.

Model performance is measured using:

- Accuracy  
- ROC-AUC  

---

### Why does it matter?

Feature engineering is one of the most important steps in machine learning, but it is often ignored.

This project shows clearly how different types of features affect:

- prediction performance  
- model stability  
- interpretability  

Understanding this helps in building better real-world machine learning models.

---

### Research Questions

1. How does model performance differ when using raw features versus engineered features?  
2. Which feature engineering strategies improve performance the most?  
3. Do behavioral features provide more predictive power than simple features?  
4. When does feature engineering stop improving performance?  
5. How does feature engineering affect model stability and interpretability?  

---

## 3. Data 

### Data sources

- The dataset used in this project is the Default of Credit Card Clients Dataset from the UCI Machine Learning Repository.

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

---

### Data size

- ~5 MB (Excel format)

---

### Data shape

- Number of rows: 30,000  
- Number of columns: 24  

---

### Time period

- Historical data covering six months of billing and payment behavior per customer

---

### What does each row represent?

Each row represents one individual credit card customer, including:

- credit limit  
- monthly bill amounts  
- monthly payment amounts  
- repayment behavior  
- default outcome  

---

### Data dictionary

| Column Name | Data Type | Definition | Potential Values |
|------------|----------|------------|------------------|
| ID | Numeric | Unique identifier | Integer |
| LIMIT_BAL | Numeric | Credit limit | Numeric |
| SEX | Categorical | Gender | 1 = Male, 2 = Female |
| EDUCATION | Categorical | Education level | 1–4 |
| MARRIAGE | Categorical | Marital status | 1–3 |
| AGE | Numeric | Age | Integer |
| PAY_0 to PAY_6 | Categorical | Repayment status | -1, 0, 1–9 |
| BILL_AMT1–6 | Numeric | Bill amounts | Numeric |
| PAY_AMT1–6 | Numeric | Payment amounts | Numeric |
| default.payment.next.month | Binary | Default status | 0 = No, 1 = Yes |

---

### Target Variable

- **default.payment.next.month**
  - 0 → No default  
  - 1 → Default  

---

### Features / Predictors

- credit limit  
- bill amounts  
- payment amounts  
- repayment history  

These are later transformed into engineered features.

---

## 4. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset clearly before building machine learning models. This step helps in identifying patterns, relationships, and data issues.

### Overview

- The dataset was explored using Jupyter Notebook.  
- Plotly visualizations were used to understand distributions and relationships.  
- The focus was mainly on:
  - target variable (default)
  - important features such as payment behavior and bill amounts  

---

### Summary Statistics

- Total number of customers: 30,000  
- Non-defaulters: around 78%  
- Defaulters: around 22%  

- This shows the dataset is **imbalanced**, meaning one class is much larger than the other.  
- This imbalance affects model performance, especially recall.  

---

### Distribution Analysis

- Credit limit is **right-skewed**:
  - most customers have low or medium limits  
  - few customers have very high limits  

- Age distribution:
  - most customers are between 25 and 40  
  - very few older customers  

- Bill and payment amounts:
  - show large variation  
  - indicate different spending behaviors  

---

### Behavioral Insights

- Customers with **higher repayment delays** are more likely to default  
- Customers who **pay smaller amounts** compared to their bills are more risky  
- Repayment history (PAY_X variables) is the **strongest predictor**  

- This shows:
  - behavior is more important than income or spending  
  - financial habits matter more than financial size  

---

### Feature Engineering

New features were created to better represent customer behavior:

- **avg_bill**
  - average bill amount over 6 months  
  - reduces noise from monthly fluctuations  

- **avg_payment**
  - average payment amount  
  - shows repayment consistency  

- **avg_delay**
  - average repayment delay  
  - captures severity of delay  

- **delay_count**
  - number of delayed months  
  - captures frequency of delay  

- These features combine multiple columns into simple and meaningful values.  

---

### Correlation Insights

- delay_count has a **strong positive relationship** with default  
- avg_delay is also strongly related  
- avg_bill has weak or no relationship  
- age has almost no impact  

- This shows:
  - behavioral features are the most useful  
  - financial features alone are not enough  

---

### Data Cleaning

- Checked for missing values: none found  

- Checked for duplicate rows: none found  

- Removed unnecessary column: ID (not useful for prediction)  

---

### Data Transformation

- Removed raw bill and payment columns: replaced by aggregated features  

- Removed redundant features: to avoid multicollinearity  

- Applied one-hot encoding: converted categorical variables into numeric form  

---

### Final Dataset

- Clean and structured  
- No missing or duplicate values  
- Reduced noise and redundancy  
- Ready for machine learning  

---

## 5. Model Training

### Models Used

- Logistic Regression
  - simple and interpretable  
  - works well with scaled data  

- Random Forest
  - handles non-linear relationships  
  - captures complex patterns  

- Logistic Regression (Balanced)
  - handles class imbalance  
  - improves recall for defaulters  

---

### Training Process

- Dataset split into:
  - 80% training data  
  - 20% testing data  

- Stratified sampling used:
  - keeps same class distribution  

- Feature scaling applied:
  - StandardScaler used for Logistic Regression  

---

### Tools Used

- Python  
- pandas and numpy for data handling  
- scikit-learn for modeling  
- Plotly for visualization  
- Jupyter Notebook for development  

---

### Evaluation Metrics

- **Accuracy**
  - overall correctness of model  

- **ROC-AUC**
  - ability to distinguish between classes  

- **Precision**
  - how many predicted defaults are correct  

- **Recall**
  - how many actual defaults are detected  

- **F1-score**
  - balance between precision and recall  

---

### Results

**Logistic Regression**

- Accuracy: ~0.81  
- ROC-AUC: ~0.74  
- Recall: 0.29  

- Good overall performance  
- Misses many defaulters  

---

**Random Forest**

- Accuracy: ~0.81  
- ROC-AUC: ~0.75  
- Recall: 0.36  

- Better than Logistic Regression  
- Captures complex patterns  

---

**Balanced Logistic Regression**

- Accuracy: ~0.76  
- ROC-AUC: ~0.74  
- Recall: 0.56  

- Lower accuracy  
- Much better at detecting defaulters  

---

### Key Insights

- Accuracy alone is misleading  
- Recall is more important in credit risk  
- Balanced model performs best for real-world use  
