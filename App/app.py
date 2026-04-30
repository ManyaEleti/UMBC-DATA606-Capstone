import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd

# --------------------------
# CONFIG
# --------------------------
st.set_page_config(page_title="Credit Risk Platform", layout="wide")

# --------------------------
# LOAD MODEL
# --------------------------
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# --------------------------
# HEADER
# --------------------------
st.markdown("""
    <div style='background: linear-gradient(90deg,#0f2027,#203a43,#2c5364);
                padding:25px;border-radius:15px;text-align:center'>
        <h1 style='color:white;'>💳 Credit Risk Decision Platform</h1>
        <p style='color:white;'>AI-powered system for real-time credit risk assessment</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# EXECUTIVE DASHBOARD
# --------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "81%")
col2.metric("Recall (Defaulters)", "56%")
col3.metric("F1 Score", "0.51")
col4.metric("Customers", "30K+")

st.info("👉 Model optimized for **high recall** to detect risky customers early.")

st.markdown("---")

# --------------------------
# MODEL JUSTIFICATION
# --------------------------
st.subheader("🏆 Model Selection")

st.success("""
Balanced Logistic Regression selected:
- Highest recall (better risk detection)
- Acceptable accuracy tradeoff
- Suitable for financial risk systems
""")

st.markdown("---")

# --------------------------
# INPUT SECTION
# --------------------------
left, right = st.columns(2)

with left:
    st.subheader("👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", min_value=0, value=0, step=1000)

    # AGE FIXED
    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    sex = st.selectbox("Sex", ["Select", "Male", "Female"])
    education = st.selectbox("Education", ["Select", "Graduate School", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Select", "Married", "Single", "Other"])

with right:
    st.subheader("📊 Financial Behavior")

    pay_0 = st.selectbox("Last Month", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_2 = st.selectbox("2 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_3 = st.selectbox("3 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_4 = st.selectbox("4 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_5 = st.selectbox("5 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_6 = st.selectbox("6 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])

    avg_bill = st.number_input("Average Bill", min_value=0.0, value=0.0)
    avg_payment = st.number_input("Average Payment", min_value=0.0, value=0.0)
    avg_delay = st.number_input("Average Delay", min_value=0.0, value=0.0)
    delay_count = st.number_input("Delay Count", min_value=0, value=0)

st.markdown("---")

# --------------------------
# ENCODING
# --------------------------
def safe(x):
    return 0 if x == "Select" else x

sex_2 = 1 if sex == "Female" else 0
education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0
marriage_2 = 1 if marriage == "Single" else 0
marriage_3 = 1 if marriage == "Other" else 0

features = np.array([[
    limit_bal, age,
    safe(pay_0), safe(pay_2), safe(pay_3), safe(pay_4), safe(pay_5), safe(pay_6),
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
]])

# --------------------------
# PREDICTION
# --------------------------
if st.button("🚀 Analyze Customer Risk"):

    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Risk Assessment")

    # --------------------------
    # RISK SEGMENT
    # --------------------------
    if prob > 0.75:
        st.error("🔴 Critical Risk")
        decision = "Reject / Reduce Credit"
    elif prob > 0.5:
        st.warning("🟠 High Risk")
        decision = "Monitor Closely"
    elif prob > 0.3:
        st.info("🟡 Medium Risk")
        decision = "Watch Behavior"
    else:
        st.success("🟢 Low Risk")
        decision = "Approve"

    st.metric("Default Probability", f"{prob:.2%}")

    # --------------------------
    # EXPLANATION
    # --------------------------
    st.subheader("🧠 Explanation")

    if delay_count > 2:
        st.write("• High delay count increases default probability")

    if avg_delay > 1:
        st.write("• Frequent delays signal financial instability")

    if avg_payment < avg_bill * 0.5:
        st.write("• Low payments relative to bill indicate financial stress")

    if safe(pay_0) > 1:
        st.write("• Recent missed payments strongly affect risk")

    if delay_count == 0 and avg_payment >= avg_bill:
        st.write("• Stable financial behavior observed")

    # --------------------------
    # RECOMMENDATION
    # --------------------------
    st.subheader("💼 Recommended Action")
    st.info(f"👉 {decision}")

    # --------------------------
    # SCENARIO SIMULATION (FIXED)
    # --------------------------
    st.subheader("🔄 Scenario Simulation")

    new_delay = st.number_input(
        "Adjust delay count:",
        min_value=0,
        max_value=10,
        value=delay_count
    )

    temp = features.copy()
    temp[0][11] = new_delay

    new_prob = model.predict_proba(scaler.transform(temp))[0][1]

    st.write(f"New Risk: **{new_prob:.2%}**")

    # Correct interpretation
    if new_prob < 0.3:
        st.success("✅ Low risk — customer becomes safe")

    elif new_prob < 0.5:
        st.info("🟡 Medium risk — improvement seen but still needs monitoring")

    else:
        st.warning("🟠 High risk — still risky despite changes")

    # Compare change
    if new_prob < prob:
        st.write("⬇️ Risk decreased due to improved behavior")
    elif new_prob > prob:
        st.write("⬆️ Risk increased")
    else:
        st.write("➡️ No significant change")
