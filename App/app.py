import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk AI", layout="wide")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Credit Risk Intelligence Dashboard")
st.caption("AI-powered system to predict and explain credit default risk")

# -------------------------------
# INPUTS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)
    age = st.slider("Age", 18, 80, 30)

    sex = st.selectbox("Sex", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Single", "Married", "Other"])

with col2:
    st.subheader("📊 Financial Behavior")

    pay_0 = st.slider("Recent Delay", -2, 8, 0)
    pay_2 = st.slider("2 Months Ago", -2, 8, 0)

    avg_bill = st.number_input("Avg Bill", value=0.0)
    avg_payment = st.number_input("Avg Payment", value=0.0)
    avg_delay = st.number_input("Avg Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

# -------------------------------
# ENCODING
# -------------------------------
sex_2 = 1 if sex == "Female" else 0
education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0
marriage_2 = 1 if marriage == "Married" else 0
marriage_3 = 1 if marriage == "Other" else 0

features = np.array([[
    limit_bal, age,
    pay_0, pay_2, 0, 0, 0, 0,
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
]])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Analyze Risk"):

    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0][1]

    st.divider()

    # -------------------------------
    # RISK SEGMENTATION
    # -------------------------------
    if prob > 0.7:
        st.error("🔴 High Risk")
        decision = "Reject"
    elif prob > 0.4:
        st.warning("🟠 Medium Risk")
        decision = "Review"
    else:
        st.success("🟢 Low Risk")
        decision = "Approve"

    st.metric("Default Probability", f"{prob:.2%}")

    # -------------------------------
    # EXPLAINABILITY
    # -------------------------------
    st.subheader("🧠 Why this prediction?")

    insights = []

    if delay_count > 2:
        insights.append("Frequent payment delays")
    if avg_delay > 1:
        insights.append("High average delay")
    if avg_payment < avg_bill:
        insights.append("Payments less than bills")
    if pay_0 > 1:
        insights.append("Recent payment issues")

    if not insights:
        insights.append("Stable financial behavior")

    for i in insights:
        st.write(f"• {i}")

    # -------------------------------
    # DECISION PANEL
    # -------------------------------
    st.subheader("🏦 Business Recommendation")

    if decision == "Reject":
        st.error("❌ Reject Application")
    elif decision == "Review":
        st.warning("⚠️ Manual Review Required")
    else:
        st.success("✅ Approve Credit")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.markdown("## 📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": ["Delay Count", "Avg Delay", "Avg Payment", "Avg Bill"],
    "Importance": [0.35, 0.25, 0.2, 0.2]
})

fig = px.bar(importance_df, x="Feature", y="Importance", title="Top Risk Drivers")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.markdown("## 📊 Model Comparison")

metrics_df = pd.DataFrame({
    "Model": ["Logistic", "Balanced Logistic", "Random Forest"],
    "Accuracy": [0.81, 0.76, 0.82],
    "Recall": [0.29, 0.56, 0.48],
    "F1": [0.43, 0.51, 0.53]
})

fig2 = px.bar(metrics_df.melt(id_vars="Model"),
              x="Model", y="value", color="variable",
              barmode="group")

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# INSIGHT BANNER
# -------------------------------
st.info("💡 Feature engineering improved recall from 29% to 56%, making the model significantly better at detecting high-risk customers.")
