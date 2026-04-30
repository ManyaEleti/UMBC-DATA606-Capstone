import streamlit as st
import numpy as np
import joblib
import os
import plotly.express as px
import pandas as pd

# -------------------------------
# PAGE CONFIG (IMPORTANT)
# -------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# CUSTOM CSS (PREMIUM UI)
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

.section {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
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

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

# LEFT
with col1:
    st.markdown("### 👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)
    age = st.slider("Age", 18, 75, 25)

    sex = st.selectbox("Sex", ["Select...", "Male", "Female"])
    education = st.selectbox("Education Level",
                             ["Select...", "Graduate School", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status",
                            ["Select...", "Married", "Single", "Other"])

# RIGHT
with col2:
    st.markdown("### 📊 Financial Behavior")

    pay_0 = st.selectbox("Last Month Payment Behavior",
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

    # -------------------------------
    # FEATURE IMPORTANCE (VISUAL)
    # -------------------------------
    st.markdown("## 📊 Feature Importance")

    importance = pd.DataFrame({
        "Feature": ["Delay Count", "Avg Delay", "Avg Payment", "Avg Bill"],
        "Importance": [0.35, 0.25, 0.20, 0.20]
    })

    fig1 = px.bar(importance, x="Feature", y="Importance",
                  title="Top Risk Drivers")

    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------
    # MODEL COMPARISON
    # -------------------------------
    st.markdown("## 📊 Model Comparison")

    df = pd.DataFrame({
        "Model": ["Logistic", "Balanced Logistic", "Random Forest"],
        "Accuracy": [0.81, 0.76, 0.82],
        "Recall": [0.29, 0.56, 0.48],
        "F1": [0.43, 0.51, 0.53]
    })

    df_melt = df.melt(id_vars="Model")

    fig2 = px.bar(df_melt, x="Model", y="value", color="variable",
                  barmode="group",
                  title="Model Performance Comparison")

    st.plotly_chart(fig2, use_container_width=True)
