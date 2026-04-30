import os
import numpy as np
import joblib
import streamlit as st

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Risk AI",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# ADVANCED CSS (🔥 MODERN FINTECH)
# -------------------------------
st.markdown("""
<style>

/* Background Gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

/* Header */
.title {
    font-size: 40px;
    font-weight: 800;
    color: #1e3a8a;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* Gradient Button */
.stButton>button {
    background: linear-gradient(90deg, #4f46e5, #3b82f6);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0px 6px 20px rgba(59,130,246,0.4);
}

/* Input Styling */
.stTextInput, .stNumberInput, .stSelectbox {
    border-radius: 12px !important;
}

/* Metric Cards */
.metric-box {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(135deg, #ffffff, #f1f5f9);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.05);
    text-align: center;
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #1f2937;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL
# -------------------------------
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="title">💳 Credit Risk Intelligence Dashboard</div>', unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
AI-powered system to predict credit default risk in real-time.<br><br>
<b>Focus:</b> Detect high-risk customers using behavioral patterns and feature engineering.
</div>
""", unsafe_allow_html=True)

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# LEFT PANEL
# -------------------------------
with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Customer Profile</div>', unsafe_allow_html=True)

    limit_bal = st.text_input("Credit Limit ($)", value="0")
    try:
        limit_bal = float(limit_bal)
    except:
        limit_bal = 0

    age = st.number_input("Age", 18, 100, 25)

    sex = st.selectbox("Sex", ["Select...", "Male", "Female"])

    education = st.selectbox(
        "Education Level",
        ["Select...", "Graduate School", "University", "High School", "Others"]
    )

    marriage = st.selectbox(
        "Marital Status",
        ["Select...", "Married", "Single", "Others"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# PAYMENT OPTIONS
# -------------------------------
payment_options = {
    "No delay / Paid early": -2,
    "Paid on time": 0,
    "1 month delay": 1,
    "2 months delay": 2,
    "3 months delay": 3,
    "4 months delay": 4,
    "5 months delay": 5,
    "6+ months delay": 6
}

# -------------------------------
# RIGHT PANEL
# -------------------------------
with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Financial Behavior</div>', unsafe_allow_html=True)

    pay_0 = payment_options.get(st.selectbox("Last Month", ["Select..."] + list(payment_options.keys())))
    pay_2 = payment_options.get(st.selectbox("2 Months Ago", ["Select..."] + list(payment_options.keys())))
    pay_3 = payment_options.get(st.selectbox("3 Months Ago", ["Select..."] + list(payment_options.keys())))
    pay_4 = payment_options.get(st.selectbox("4 Months Ago", ["Select..."] + list(payment_options.keys())))
    pay_5 = payment_options.get(st.selectbox("5 Months Ago", ["Select..."] + list(payment_options.keys())))
    pay_6 = payment_options.get(st.selectbox("6 Months Ago", ["Select..."] + list(payment_options.keys())))

    avg_bill = st.number_input("Average Bill ($)", value=0.0)
    avg_payment = st.number_input("Average Payment ($)", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# ENCODING
# -------------------------------
sex_val = 1 if sex == "Male" else 2 if sex == "Female" else None

edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
education_val = edu_map.get(education)

mar_map = {"Married": 1, "Single": 2, "Others": 3}
marriage_val = mar_map.get(marriage)

sex_2 = 1 if sex_val == 2 else 0
education_2 = 1 if education_val == 2 else 0
education_3 = 1 if education_val == 3 else 0
education_4 = 1 if education_val == 4 else 0
marriage_2 = 1 if marriage_val == 2 else 0
marriage_3 = 1 if marriage_val == 3 else 0

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Analyze Risk"):

    if None in [sex_val, education_val, marriage_val, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]:
        st.error("⚠️ Please complete all fields")
    else:
        features = np.array([[limit_bal, age, pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
                              avg_bill, avg_payment, avg_delay, delay_count,
                              sex_2, education_2, education_3, education_4,
                              marriage_2, marriage_3]])

        features_scaled = scaler.transform(features)

        probability = model.predict_proba(features_scaled)[0][1]

        st.divider()

        # -------------------------------
        # RESULT DASHBOARD
        # -------------------------------
        colA, colB = st.columns(2)

        with colA:
            st.markdown(f"""
            <div class="metric-box">
                <h2>{probability:.2%}</h2>
                <p>Default Probability</p>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            st.progress(float(probability))

        # Risk Color Logic
        if probability > 0.7:
            st.error("🔴 High Risk Customer")
        elif probability > 0.4:
            st.warning("🟠 Moderate Risk")
        else:
            st.success("🟢 Low Risk Customer")
