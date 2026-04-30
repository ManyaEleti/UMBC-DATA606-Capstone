import streamlit as st
import numpy as np
import joblib
import os

# --------------------------
# LOAD MODEL
# --------------------------
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# --------------------------
# HEADER
# --------------------------
st.markdown("""
    <div style='background: linear-gradient(90deg,#1e3c72,#2a5298);
                padding:20px;border-radius:12px;text-align:center'>
        <h1 style='color:white;'>💳 Credit Risk Intelligence Dashboard</h1>
        <p style='color:white;'>AI system to detect high-risk customers</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --------------------------
# TOP INSIGHTS (STATIC)
# --------------------------
colA, colB, colC = st.columns(3)
colA.metric("Accuracy", "81%")
colB.metric("Recall (Defaulters)", "56%")
colC.metric("Customers", "30K+")

st.markdown("---")

# --------------------------
# INPUT SECTION
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)
    age = st.slider("Age", 18, 75, 25)

    sex = st.selectbox("Sex", ["Select", "Male", "Female"])
    education = st.selectbox("Education", ["Select", "Graduate School", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Select", "Married", "Single", "Other"])

with col2:
    st.subheader("📊 Financial Behavior")

    pay_0 = st.selectbox("Last Month Payment Status", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_2 = st.selectbox("2 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_3 = st.selectbox("3 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_4 = st.selectbox("4 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_5 = st.selectbox("5 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])
    pay_6 = st.selectbox("6 Months Ago", ["Select", -2, -1, 0, 1, 2, 3, 4])

    avg_bill = st.number_input("Average Bill", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

st.markdown("---")

# --------------------------
# ENCODING
# --------------------------
sex_2 = 1 if sex == "Female" else 0

education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0

marriage_2 = 1 if marriage == "Single" else 0
marriage_3 = 1 if marriage == "Other" else 0

def safe(x):
    return 0 if x == "Select" else x

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
if st.button("🚀 Analyze Risk"):

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    # --------------------------
    # RISK CATEGORY
    # --------------------------
    if probability > 0.75:
        risk_level = "🔴 Critical Risk"
        st.error(risk_level)
    elif probability > 0.5:
        risk_level = "🟠 High Risk"
        st.warning(risk_level)
    elif probability > 0.3:
        risk_level = "🟡 Medium Risk"
        st.info(risk_level)
    else:
        risk_level = "🟢 Low Risk"
        st.success(risk_level)

    st.metric("Default Probability", f"{probability:.2%}")

    # --------------------------
    # EXPLAINABILITY
    # --------------------------
    st.subheader("🧠 Why this prediction?")

    explanations = []

    if delay_count > 2:
        explanations.append("High number of delays increases default risk")

    if avg_delay > 1:
        explanations.append("Frequent late payments indicate instability")

    if avg_payment < avg_bill * 0.5:
        explanations.append("Low payment compared to bill suggests financial stress")

    if safe(pay_0) > 1:
        explanations.append("Recent missed payments strongly increase risk")

    if len(explanations) == 0:
        explanations.append("Customer shows stable financial behavior")

    for e in explanations:
        st.write("•", e)

    # --------------------------
    # RECOMMENDATIONS
    # --------------------------
    st.subheader("💡 Recommended Action")

    if probability > 0.7:
        st.error("Reduce credit limit and flag for monitoring")
    elif probability > 0.4:
        st.warning("Monitor customer behavior closely")
    else:
        st.success("Customer is safe to continue credit")

    # --------------------------
    # WHAT-IF SIMULATION
    # --------------------------
    st.subheader("🔄 What-if Scenario")

    new_delay = st.slider("If delay count changes to:", 0, 10, delay_count)

    temp_features = features.copy()
    temp_features[0][11] = new_delay

    new_scaled = scaler.transform(temp_features)
    new_prob = model.predict_proba(new_scaled)[0][1]

    st.write(f"👉 New Risk: **{new_prob:.2%}**")

    if new_prob < probability:
        st.success("Risk decreases with improved payment behavior")
    else:
        st.error("Risk increases further")
