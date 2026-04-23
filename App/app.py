import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("../model.pkl")
scaler = joblib.load("../scaler.pkl")

st.set_page_config(page_title="Credit Default Predictor", layout="centered")

st.title("💳 Credit Card Default Prediction")
st.write("Enter customer details to predict default risk.")

# -------------------------------
# 🔹 USER INPUTS
# -------------------------------

limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", value=20000)
age = st.number_input("Age", value=30)

pay_0 = st.slider("Repayment Status (PAY_0)", -2, 8, 0)
pay_2 = st.slider("Repayment Status (PAY_2)", -2, 8, 0)
pay_3 = st.slider("Repayment Status (PAY_3)", -2, 8, 0)
pay_4 = st.slider("Repayment Status (PAY_4)", -2, 8, 0)
pay_5 = st.slider("Repayment Status (PAY_5)", -2, 8, 0)
pay_6 = st.slider("Repayment Status (PAY_6)", -2, 8, 0)

avg_bill = st.number_input("Average Bill Amount", value=5000.0)
avg_payment = st.number_input("Average Payment Amount", value=2000.0)

avg_delay = st.number_input("Average Delay", value=0.0)
delay_count = st.number_input("Number of Delayed Months", value=0)

# -------------------------------
# 🔹 CATEGORICAL INPUTS
# -------------------------------

sex = st.selectbox("Sex", [1, 2])
education = st.selectbox("Education", [1, 2, 3, 4])
marriage = st.selectbox("Marriage", [1, 2, 3])

# -------------------------------
# 🔹 ENCODING (VERY IMPORTANT)
# -------------------------------

sex_2 = 1 if sex == 2 else 0

education_2 = 1 if education == 2 else 0
education_3 = 1 if education == 3 else 0
education_4 = 1 if education == 4 else 0

marriage_2 = 1 if marriage == 2 else 0
marriage_3 = 1 if marriage == 3 else 0

# -------------------------------
# 🔹 FEATURE ARRAY (EXACT ORDER)
# -------------------------------

features = np.array([[
    limit_bal, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
]])

# -------------------------------
# 🔹 PREDICTION
# -------------------------------

if st.button("Predict Default Risk"):

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Default ({probability:.2%})")
