

## 1. Title and Author

- Project Title -  **Evaluating the Impact of Feature Engineering on Machine Learning Model Performance**
- Author - Lakshmi Manya Eleti
- Semester - Spring'26
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Link to the author's GitHub repo of the project: https://github.com/ManyaEleti/UMBC-DATA606-Capstone
    
## 2. Background
- **What is it about?**

The proposed project aims at learning about the potential impact of data preparation and transformation on the performance of machine learning models. The project does not focus on the complicated algorithms, but rather examines the impact of various methods to generate features with raw data on the predictive ability of a model. The research utilizes actual credit card data that entails details of billing, payments and repayment pattern with time.

The project performs a comparison of the effects of each set of features on model accuracy, stability and interpretability by first creating several versions of features starting with basic raw values and then moving up to more specific behavioral patterns. It aims to demonstrate that careful feature engineering can help a lot in enhancing the performance of models, whereas too much or badly engineered features can decrease reliability. This aids in bringing out the significance of data interpretation and feature development in the real-world machine learning practice.

**Machine Learning Models**

The models that will be employed in this project are **Logistic Regression** and **Random Forest model**. The models are understandable, consistent, and prevalent in the academic literature. The fact that the analysis was based on these models only means that the focus was directed towards these models and no comparison was made between the algorithms. **Accuracy** and **ROC-AUC** will be used to measure model performance.

- **Why does it matter?**

The improvement of machine learning with feature engineering has significant contribution, but it is usually ignored in favor of more complicated algorithms. The importance of this project is that it demonstrates the impact of varying engineering of features in a clear and organised manner on the performance, stability and interpretability of a model.

The project facilitates the process of showing what actually matters in a machine learning process by comparing several feature representations on the same dataset systematically. The lessons learnt can be used in designing a better feature in the real world, resulting in more trusted models and better informed interpretation of machine learning outputs.

**What are your research questions?**

1. How does model performance differ when using raw features versus engineered features?

2. Which feature engineering strategies contribute the most to performance improvements?

3. Do behavioral and temporal features provide more predictive power than simple aggregations?

4. At what point does feature engineering lead to diminishing or negative returns?

5. How does feature engineering affect model stability and interpretability?

## 3. Data 

**Data sources**

- The dataset used in this project is the Default of Credit Card Clients Dataset from the UCI Machine Learning Repository.

 https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**Data size**
- ~5 MB (Excel format)

**Data shape**
- Number of rows: 30,000
- Number of columns: 24

**Time period:** 
- Historical data covering six months of billing and payment behavior per customer
 
**What does each row represent?**

Each row represents one individual credit card customer, including their:

* credit limit

* monthly bill amounts

* monthly payment amounts

* repayment behavior

* default outcome
 
**Data dictionary**

| Column Name | Data Type | Definition | Potential Values |
|------------|----------|------------|------------------|
| ID | Numeric | Unique identifier assigned to each credit card customer | Integer |
| LIMIT_BAL | Numeric | Total credit limit assigned to the customer | Positive numeric value |
| SEX | Categorical | Gender of the customer | 1 = Male, 2 = Female |
| EDUCATION | Categorical | Education level of the customer | 1 = Graduate school, 2 = University, 3 = High school, 4 = Others |
| MARRIAGE | Categorical | Marital status of the customer | 1 = Married, 2 = Single, 3 = Others |
| AGE | Numeric | Age of the customer in years | Integer |
| PAY_0 | Categorical | Repayment status in the most recent month | -1 = Paid in full, 0 = Revolving credit, 1–9 = Delay in months |
| PAY_2 | Categorical | Repayment status two months ago | -1, 0, 1–9 |
| PAY_3 | Categorical | Repayment status three months ago | -1, 0, 1–9 |
| PAY_4 | Categorical | Repayment status four months ago | -1, 0, 1–9 |
| PAY_5 | Categorical | Repayment status five months ago | -1, 0, 1–9 |
| PAY_6 | Categorical | Repayment status six months ago | -1, 0, 1–9 |
| BILL_AMT1 | Numeric | Bill statement amount in the most recent month | Numeric value |
| BILL_AMT2 | Numeric | Bill statement amount two months ago | Numeric value |
| BILL_AMT3 | Numeric | Bill statement amount three months ago | Numeric value |
| BILL_AMT4 | Numeric | Bill statement amount four months ago | Numeric value |
| BILL_AMT5 | Numeric | Bill statement amount five months ago | Numeric value |
| BILL_AMT6 | Numeric | Bill statement amount six months ago | Numeric value |
| PAY_AMT1 | Numeric | Amount paid by the customer in the most recent month | Numeric value |
| PAY_AMT2 | Numeric | Amount paid by the customer two months ago | Numeric value |
| PAY_AMT3 | Numeric | Amount paid by the customer three months ago | Numeric value |
| PAY_AMT4 | Numeric | Amount paid by the customer four months ago | Numeric value |
| PAY_AMT5 | Numeric | Amount paid by the customer five months ago | Numeric value |
| PAY_AMT6 | Numeric | Amount paid by the customer six months ago | Numeric value |
| default.payment.next.month | Binary | Indicates whether the customer defaulted on payment in the following month | 0 = No, 1 = Yes |


**Which variable/column will be your target/label in your ML model?**
- The target variable for this project is **default.payment.next.month**, which indicates whether a customer defaulted on their credit card payment in the following month.
- This variable is used only for training and evaluating the models and is not included in the feature engineering process.
  
**Which variables/columns may be selected as features/predictors for your ML models?**
- The model will use information such as a **customer’s credit limit**, **monthly bill amounts**, **payment amounts** and **repayment history**.
- These variables will be transformed into meaningful features that reflect how customers use and repay credit over time.
- The target is excluded from predictor construction
