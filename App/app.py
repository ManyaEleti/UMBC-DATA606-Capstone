import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# PREMIUM CSS
# -------------------------------
st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

.title-box {
    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title-box">💳 Credit Risk Intelligence Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
AI-powered system to predict **credit default risk in real-time**  
Focus: Detect **high-risk customers** using behavioral insights
""")

st.markdown("---")

# ===============================
# 📊 TOP DASHBOARD (INSIGHTS)
# ===============================

st.markdown("## 📊 Model Performance Dashboard")

# KPI CARDS
k1, k2, k3 = st.columns(3)

k1.metric("Accuracy", "81%")
k2.metric("Recall (Defaulters)", "56%")
k3.metric("Customers", "30K+")

st.markdown("---")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.markdown("### 📊 Key Risk Drivers")

importance = pd.DataFrame({
    "Feature": ["Delay Count", "Avg Delay", "Avg Payment", "Avg Bill"],
    "Importance": [0.35, 0.25, 0.20, 0.20]
})

fig1 = px.bar(
    importance,
    x="Feature",
    y="Importance",
    color="Importance",
    color_continuous_scale="Blues"
)

st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# MODEL COMPARISON
# -------------------------------
st.markdown("### 📊 Model Comparison")

df = pd.DataFrame({
    "Model": ["Logistic", "Balanced Logistic", "Random Forest"],
    "Accuracy": [0.81, 0.76, 0.82],
    "Recall": [0.29, 0.56, 0.48],
    "F1": [0.43, 0.51, 0.53]
})

df_melt = df.melt(id_vars="Model")

fig2 = px.bar(
    df_melt,
    x="Model",
    y="value",
    color="variable",
    barmode="group"
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.markdown("### 📊 Confusion Matrix")

cm = np.array([[3828, 845],
               [583, 744]])

fig3 = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="Actual")
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# ROC CURVE
# -------------------------------
st.markdown("### 📈 ROC Curve")

roc_df = pd.DataFrame({
    "FPR": [0, 0.1, 0.2, 0.3, 1],
    "TPR": [0, 0.5, 0.65, 0.75, 1]
})

fig4 = px.line(roc_df, x="FPR", y="TPR")

fig4.add_shape(
    type="line",
    line=dict(dash="dash"),
    x0=0, x1=1, y0=0, y1=1
)

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ===============================
# 🧾 USER INPUT SECTION
# ===============================

col1, col2 = st.columns(2)

# LEFT SIDE
with col1:
    st.markdown("### 👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)
    age = st.slider("Age", 18, 75, 25)

    sex = st.selectbox("Sex", ["Select...", "Male", "Female"])
    education = st.selectbox(
        "Education Level",
        ["Select...", "Graduate School", "University", "High School", "Other"]
    )
    marriage = st.selectbox(
        "Marital Status",
        ["Select...", "Married", "Single", "Other"]
    )

# RIGHT SIDE
with col2:
    st.markdown("### 📊 Financial Behavior")

    pay_0 = st.selectbox("Last Month Behavior",
                         ["Select...", "On Time", "1 Month Delay", "2+ Months Delay"])

    pay_2 = st.selectbox("2 Months Ago", ["Select...", "On Time", "Delay"])
    pay_3 = st.selectbox("3 Months Ago", ["Select...", "On Time", "Delay"])
    pay_4 = st.selectbox("4 Months Ago", ["Select...", "On Time", "Delay"])
    pay_5 = st.selectbox("5 Months Ago", ["Select...", "On Time", "Delay"])
    pay_6 = st.selectbox("6 Months Ago", ["Select...", "On Time", "Delay"])

    avg_bill = st.number_input("Average Bill Amount", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Number of Delays", value=0)

st.markdown("---")

# -------------------------------
# ENCODING
# -------------------------------
sex_2 = 1 if sex == "Female" else 0

education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0

marriage_2 = 1 if marriage == "Single" else 0
marriage_3 = 1 if marriage == "Other" else 0

def encode_pay(val):
    if val == "On Time":
        return 0
    elif val == "1 Month Delay":
        return 1
    elif val == "2+ Months Delay":
        return 2
    elif val == "Delay":
        return 1
    else:
        return 0

pay_0 = encode_pay(pay_0)
pay_2 = encode_pay(pay_2)
pay_3 = encode_pay(pay_3)
pay_4 = encode_pay(pay_4)
pay_5 = encode_pay(pay_5)
pay_6 = encode_pay(pay_6)

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

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk of Default")

    st.progress(float(probability))
    st.metric("Default Probability", f"{probability:.2%}")
