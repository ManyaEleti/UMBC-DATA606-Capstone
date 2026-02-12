

## 1. Title and Author

- Project Title -  **Evaluating the Impact of Feature Engineering on Machine Learning Model Performance**
- Author - Lakshmi Manya Eleti
- Semester - Spring'26
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Link to the author's GitHub repo of the project:
- Link to the author's LinkedIn profile:
    
## 2. Background
- **What is it about?**



- **Why does it matter?**



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
| Date received | Date | Date the complaint was received by the CFPB | YYYY-MM-DD |
| Product | Categorical | Financial product involved in the complaint | Credit reporting, Mortgage, Student loan, Bank account, etc. |
| Sub-product | Categorical | More specific category of the financial product | Varies by product |
| Issue | Categorical | Primary issue reported by the consumer | Billing disputes, Incorrect information, Payment issues |
| Sub-issue | Categorical | More detailed description of the issue | Varies |
| Consumer complaint narrative | Text | Free-text description of the complaint submitted by the consumer | Unstructured text |
| Company | Categorical | Name of the company named in the complaint | Financial institution names |
| State | Categorical | State where the consumer resides | WA |
| ZIP code | Categorical | Partial ZIP code of the consumer | 981XX, 983XX |
| Submitted via | Categorical | Channel through which the complaint was submitted | Web, Phone |
| Company response to consumer | Categorical | Outcome of the company’s response | Closed, In progress |
| Timely response? | Categorical | Indicates whether the company responded on time | Yes, No |
| Consumer disputed? | Categorical | Indicates whether the consumer disputed the company’s response | Yes, No |
| Complaint ID | Numeric | Unique identifier for each complaint | Integer |

**Which variable/column will be your target/label in your ML model?**
- This project does not have any target variable provided in the dataset, the target variable is derived during the analysis.
- **Complaint root cause** will be the target variable for the modeling.
  
**Which variables/columns may be selected as features/predictors for your ML models?**
- The primary column used will be **Consumer complaint narrative**.
- **Supporting features:** Product, Sun-Product, Issue, Company and Date received.
