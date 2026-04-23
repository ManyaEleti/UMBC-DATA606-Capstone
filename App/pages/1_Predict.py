import streamlit as st
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "../model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "../scaler.pkl"))

st.title("🔍 Predict Customer Risk")

col1, col2 = st.columns(2)

# -------------------------------
# INPUTS
# -------------------------------
with col1:
    st.subheader("Customer Profile")

    limit_bal = st.number_input("Credit Limit", 1000, 100000, 20000)
    age = st.number_input("Age", 18, 80, 30)

with col2:
    st.subheader("Behavior")

    payment_map = {
        "On Time": 0,
        "1 Month Late": 1,
        "2+ Months Late": 2
    }

    pay_0 = payment_map[st.selectbox("Last Month Payment", payment_map.keys())]
    pay_2 = payment_map[st.selectbox("2 Months Ago", payment_map.keys())]

avg_bill = st.number_input("Avg Bill", 0.0, 100000.0, 5000.0)
avg_payment = st.number_input("Avg Payment", 0.0, 100000.0, 2000.0)

# -------------------------------
# PREDICT
# -------------------------------
if st.button("Predict"):

    features = np.array([[limit_bal, age, pay_0, pay_2,
                          avg_bill, avg_payment]])

    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    st.subheader("Result")

    st.metric("Default Probability", f"{prob:.2%}")
    st.progress(float(prob))

    if prob > 0.7:
        st.error("High Risk")
    elif prob > 0.4:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")
