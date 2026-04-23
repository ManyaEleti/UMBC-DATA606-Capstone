import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Credit Default Risk Predictor")
st.markdown("### AI-powered financial risk assessment")

st.info("🎯 Focus: Detect high-risk customers (recall over accuracy)")

st.divider()

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# CUSTOMER INFO
# -------------------------------
with col1:
    st.subheader("👤 Customer Profile")

    # NO +/- buttons
    limit_bal = st.text_input("Credit Limit ($)", value="0")
    age = st.slider("Age", 18, 80, 18)

    sex = st.selectbox("Sex", [1, 2])
    education = st.selectbox("Education Level", [1, 2, 3, 4])
    marriage = st.selectbox("Marital Status", [1, 2, 3])

# -------------------------------
# FINANCIAL BEHAVIOR
# -------------------------------
with col2:
    st.subheader("📊 Payment Behavior")

    payment_options = {
        "On Time": 0,
        "1 Month Late": 1,
        "2 Months Late": 2,
        "3+ Months Late": 3
    }

    pay_0 = payment_options[st.selectbox("Recent Payment (Last Month)", list(payment_options.keys()))]
    pay_2 = payment_options[st.selectbox("2 Months Ago", list(payment_options.keys()))]
    pay_3 = payment_options[st.selectbox("3 Months Ago", list(payment_options.keys()))]
    pay_4 = payment_options[st.selectbox("4 Months Ago", list(payment_options.keys()))]
    pay_5 = payment_options[st.selectbox("5 Months Ago", list(payment_options.keys()))]
    pay_6 = payment_options[st.selectbox("6 Months Ago", list(payment_options.keys()))]

    st.markdown("#### 💰 Financial Summary")

    avg_bill = st.text_input("Average Monthly Bill ($)", value="0")
    avg_payment = st.text_input("Average Monthly Payment ($)", value="0")
    avg_delay = st.text_input("Average Delay (Months)", value="0")
    delay_count = st.text_input("Number of Delayed Payments", value="0")

st.divider()

# -------------------------------
# SAFE CONVERSION
# -------------------------------
try:
    limit_bal = float(limit_bal)
    avg_bill = float(avg_bill)
    avg_payment = float(avg_payment)
    avg_delay = float(avg_delay)
    delay_count = int(delay_count)
except:
    st.error("⚠️ Please enter valid numeric values")
    st.stop()

# -------------------------------
# ENCODING (UNCHANGED)
# -------------------------------
sex_2 = 1 if sex == 2 else 0

education_2 = 1 if education == 2 else 0
education_3 = 1 if education == 3 else 0
education_4 = 1 if education == 4 else 0

marriage_2 = 1 if marriage == 2 else 0
marriage_3 = 1 if marriage == 3 else 0

# -------------------------------
# FEATURE VECTOR
# -------------------------------
features = np.array([[
    limit_bal, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
]])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Risk"):

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.divider()
    st.subheader("📊 Risk Analysis")

    # -------------------------------
    # RISK CATEGORY
    # -------------------------------
    if probability > 0.7:
        st.error("🔴 High Risk Customer")
        risk_label = "High"
    elif probability > 0.4:
        st.warning("🟠 Moderate Risk Customer")
        risk_label = "Moderate"
    else:
        st.success("🟢 Low Risk Customer")
        risk_label = "Low"

    # -------------------------------
    # PROBABILITY
    # -------------------------------
    st.write("### Default Probability")
    st.progress(float(probability))
    st.metric("Risk Score", f"{probability:.2%}")

    # -------------------------------
    # INSIGHTS
    # -------------------------------
    st.subheader("🧠 AI Insights")

    insights = []

    if delay_count > 2:
        insights.append("Frequent payment delays detected")
    if avg_delay > 1:
        insights.append("Consistent payment delays")
    if avg_payment < avg_bill:
        insights.append("Payments lower than billed amount")
    if pay_0 >= 2:
        insights.append("Recent payment behavior is risky")

    if len(insights) == 0:
        insights.append("Customer shows stable behavior")

    for i in insights:
        st.write(f"• {i}")

    # -------------------------------
    # DECISION
    # -------------------------------
    st.subheader("🏦 Recommendation")

    if risk_label == "High":
        st.error("❌ Do NOT approve credit")
    elif risk_label == "Moderate":
        st.warning("⚠️ Approve with caution")
    else:
        st.success("✅ Safe to approve")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.markdown("Capstone Project • Machine Learning + Feature Engineering + Streamlit")
