import streamlit as st
import numpy as np
import joblib
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Intelligence",
    layout="wide"
)

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

body {
    background: linear-gradient(135deg, #f5f7fb, #eef2f7);
}

/* Header Block */
.header-box {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 25px;
    border-radius: 12px;
    color: white;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
}

/* Cards */
.glass-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Inputs */
input, select {
    border-radius: 8px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
}

/* KPI cards */
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="header-box">
    <h1>💳 Credit Risk Intelligence Dashboard</h1>
    <p>AI-powered system to predict credit default risk in real-time.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# -------------------------------
# KPI SECTION
# -------------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.markdown('<div class="metric-card"><h2>~81%</h2><p>Model Accuracy</p></div>', unsafe_allow_html=True)

with colB:
    st.markdown('<div class="metric-card"><h2>56%</h2><p>Recall (Defaulters)</p></div>', unsafe_allow_html=True)

with colC:
    st.markdown('<div class="metric-card"><h2>30K+</h2><p>Customers Analyzed</p></div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# MAIN LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# LEFT: CUSTOMER PROFILE
# -------------------------------
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### 👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)

    age = st.selectbox(
        "Age Group",
        ["Select...", "18-25", "26-35", "36-50", "50+"]
    )

    sex = st.selectbox(
        "Sex",
        ["Select...", "Male", "Female"]
    )

    education = st.selectbox(
        "Education Level",
        ["Select...", "Graduate School", "University", "High School", "Other"]
    )

    marriage = st.selectbox(
        "Marital Status",
        ["Select...", "Single", "Married", "Other"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# RIGHT: FINANCIAL BEHAVIOR
# -------------------------------
with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown("### 📊 Financial Behavior")

    def pay_map(val):
        mapping = {
            "Select...": 0,
            "Paid on Time": 0,
            "1 Month Delay": 1,
            "2+ Months Delay": 2
        }
        return mapping[val]

    pay_0 = st.selectbox("Last Month", ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"])
    pay_2 = st.selectbox("2 Months Ago", ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"])
    pay_3 = st.selectbox("3 Months Ago", ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"])
    pay_4 = st.selectbox("4 Months Ago", ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"])
    pay_5 = st.selectbox("5 Months Ago", ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"])
    pay_6 = st.selectbox("6 Months Ago", ["Select...", "Paid on Time", "1 Month Delay", "2+ Months Delay"])

    avg_bill = st.number_input("Average Bill Amount", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# ENCODING
# -------------------------------
sex_2 = 1 if sex == "Female" else 0

education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0

marriage_2 = 1 if marriage == "Married" else 0
marriage_3 = 1 if marriage == "Other" else 0

age_val = 0
if age == "18-25":
    age_val = 22
elif age == "26-35":
    age_val = 30
elif age == "36-50":
    age_val = 40
elif age == "50+":
    age_val = 55

# -------------------------------
# FEATURE VECTOR
# -------------------------------
features = np.array([[
    limit_bal,
    age_val,
    pay_map(pay_0),
    pay_map(pay_2),
    pay_map(pay_3),
    pay_map(pay_4),
    pay_map(pay_5),
    pay_map(pay_6),
    avg_bill,
    avg_payment,
    avg_delay,
    delay_count,
    sex_2,
    education_2,
    education_3,
    education_4,
    marriage_2,
    marriage_3
]])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Risk"):

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.divider()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk of Default")

    st.progress(float(probability))

    st.metric("Default Probability", f"{probability:.2%}")

    if probability > 0.7:
        st.warning("Very high risk — strong chance of default.")
    elif probability > 0.4:
        st.info("Moderate risk — monitor customer behavior.")
    else:
        st.success("Low risk — customer is likely safe.")

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<hr>
<p style='text-align:center;'>Built for Capstone Project • Machine Learning + Streamlit</p>
""", unsafe_allow_html=True)
