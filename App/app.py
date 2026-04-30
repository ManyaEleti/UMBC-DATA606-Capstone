import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# CSS (CLEAN + PREMIUM)
# -------------------------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef2f7, #e6ecf5);
}

/* Header */
.header {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="header">
<h2>💳 Credit Default Risk Predictor</h2>
<p>Predict and analyze customer default risk using machine learning</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# 📊 TOP SECTION (ONLY WHAT YOU WANTED)
# =========================================================

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.markdown("## 📊 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": ["Delay Count", "Avg Delay", "Avg Payment", "Avg Bill"],
    "Importance": [0.35, 0.25, 0.2, 0.2]
})

fig1 = px.bar(
    importance_df,
    x="Feature",
    y="Importance",
    title="Top Risk Drivers",
    color="Importance",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# MODEL COMPARISON
# -------------------------------
st.markdown("## 📊 Model Comparison")

metrics_df = pd.DataFrame({
    "Model": ["Logistic", "Balanced Logistic", "Random Forest"],
    "Accuracy": [0.81, 0.76, 0.82],
    "Recall": [0.29, 0.56, 0.48],
    "F1": [0.43, 0.51, 0.53]
})

fig2 = px.bar(
    metrics_df.melt(id_vars="Model"),
    x="Model",
    y="value",
    color="variable",
    barmode="group",
    title="Model Performance"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# =========================================================
# 🎯 MAIN APP (UNCHANGED CLEAN UI)
# =========================================================

col1, col2 = st.columns(2)

# -------------------------------
# CUSTOMER PROFILE
# -------------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)

    age = st.selectbox(
        "Age Group",
        ["Select...", "18-25", "26-35", "36-50", "50+"]
    )

    sex = st.selectbox("Sex", ["Select...", "Male", "Female"])
    education = st.selectbox(
        "Education",
        ["Select...", "Graduate School", "University", "High School", "Other"]
    )
    marriage = st.selectbox(
        "Marital Status",
        ["Select...", "Single", "Married", "Other"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FINANCIAL BEHAVIOR
# -------------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Financial Behavior")

    options = ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"]

    def map_pay(x):
        return {"Select...": 0, "Paid on Time": 0, "1 Month Delay": 1, "2+ Months Delay": 2}[x]

    pay_0 = st.selectbox("Last Month", options)
    pay_2 = st.selectbox("2 Months Ago", options)
    pay_3 = st.selectbox("3 Months Ago", options)
    pay_4 = st.selectbox("4 Months Ago", options)
    pay_5 = st.selectbox("5 Months Ago", options)
    pay_6 = st.selectbox("6 Months Ago", options)

    avg_bill = st.number_input("Average Bill", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# ENCODING
# -------------------------------
sex_2 = 1 if sex == "Female" else 0
education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0
marriage_2 = 1 if marriage == "Married" else 0
marriage_3 = 1 if marriage == "Other" else 0

age_val = {
    "18-25": 22,
    "26-35": 30,
    "36-50": 40,
    "50+": 55
}.get(age, 0)

# -------------------------------
# FEATURES
# -------------------------------
features = np.array([[
    limit_bal, age_val,
    map_pay(pay_0), map_pay(pay_2), map_pay(pay_3),
    map_pay(pay_4), map_pay(pay_5), map_pay(pay_6),
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
]])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Risk"):

    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]

    st.divider()

    if prob > 0.7:
        st.error("🔴 High Risk")
    elif prob > 0.4:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    st.metric("Default Probability", f"{prob:.2%}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<p style='text-align:center;'>Capstone Project • Machine Learning + Streamlit</p>
""", unsafe_allow_html=True)
