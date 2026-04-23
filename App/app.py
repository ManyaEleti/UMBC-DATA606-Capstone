import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# CONFIG (LIGHT MODE + CLEAN UI)
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# LOAD MODEL + SCALER
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# CUSTOM CSS (PREMIUM LOOK)
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #1e3a8a;
}
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
}
.stButton>button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Credit Default Risk Predictor")
st.markdown("### Smart risk assessment for financial decision-making")

st.info("🎯 **Goal:** Identify high-risk customers early (focus on recall, not just accuracy)")

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

    limit_bal = st.number_input("Credit Limit ($)", value=20000)
    age = st.number_input("Age", value=30)

    sex = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["Graduate", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Single", "Married", "Other"])

# -------------------------------
# FINANCIAL BEHAVIOR
# -------------------------------
with col2:
    st.subheader("📊 Payment Behavior")

    payment_options = {
        "Paid Early": -1,
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
    avg_bill = st.number_input("Average Monthly Bill ($)", value=5000.0)
    avg_payment = st.number_input("Average Monthly Payment ($)", value=2000.0)
    avg_delay = st.number_input("Average Delay (Months)", value=0.0)
    delay_count = st.number_input("Number of Delayed Payments", value=0)

st.divider()

# -------------------------------
# ENCODING
# -------------------------------
sex_2 = 1 if sex == "Female" else 0

education_map = {
    "University": 2,
    "Graduate": 3,
    "High School": 4
}
education_2 = 1 if education_map.get(education) == 2 else 0
education_3 = 1 if education_map.get(education) == 3 else 0
education_4 = 1 if education_map.get(education) == 4 else 0

marriage_2 = 1 if marriage == "Married" else 0
marriage_3 = 1 if marriage == "Other" else 0

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
# PREDICT BUTTON
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
    # PROGRESS BAR
    # -------------------------------
    st.write("### Default Probability")
    st.progress(float(probability))
    st.metric("Risk Score", f"{probability:.2%}")

    # -------------------------------
    # SMART INSIGHTS
    # -------------------------------
    st.subheader("🧠 AI Insights")

    insights = []

    if delay_count > 2:
        insights.append("Frequent payment delays detected")
    if avg_delay > 1:
        insights.append("Customer consistently delays payments")
    if avg_payment < avg_bill:
        insights.append("Payments are lower than billed amount")
    if pay_0 >= 2:
        insights.append("Recent payment behavior indicates high risk")

    if len(insights) == 0:
        insights.append("Customer shows stable financial behavior")

    for i in insights:
        st.write(f"• {i}")

    # -------------------------------
    # FINAL DECISION BLOCK
    # -------------------------------
    st.subheader("🏦 Recommendation")

    if risk_label == "High":
        st.error("❌ Do NOT approve credit increase")
    elif risk_label == "Moderate":
        st.warning("⚠️ Approve with caution")
    else:
        st.success("✅ Safe to approve")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()
st.markdown(
    "Built for Capstone Project • Machine Learning + Feature Engineering + Streamlit Deployment"
)
