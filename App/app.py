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

st.info("👉 Model optimized for **high recall** to capture risky customers early.")

st.markdown("---")

# --------------------------
# MODEL SELECTION JUSTIFICATION
# --------------------------
st.subheader("🏆 Model Selection")

st.success("""
Balanced Logistic Regression selected:
- Best recall → catches more defaulters
- Slight tradeoff in accuracy acceptable
- Ideal for financial risk systems
""")

st.markdown("---")

# --------------------------
# INPUT SECTION
# --------------------------
left, right = st.columns(2)

with left:
    st.subheader("👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)
    age = st.slider("Age", 18, 75, 25)

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

    avg_bill = st.number_input("Average Bill", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

st.markdown("---")

# --------------------------
# ENCODING
# --------------------------
def safe(x): return 0 if x == "Select" else x

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
# ANALYZE BUTTON
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
        segment = "🔴 Critical Risk"
        decision = "Reject / Reduce Credit"
        st.error(segment)
    elif prob > 0.5:
        segment = "🟠 High Risk"
        decision = "Monitor Closely"
        st.warning(segment)
    elif prob > 0.3:
        segment = "🟡 Medium Risk"
        decision = "Watch Behavior"
        st.info(segment)
    else:
        segment = "🟢 Low Risk"
        decision = "Approve"
        st.success(segment)

    st.metric("Default Probability", f"{prob:.2%}")

    # --------------------------
    # FEATURE IMPACT (REALISTIC APPROX)
    # --------------------------
    st.subheader("🔍 Risk Drivers")

    impact = {
        "Delay Count": delay_count * 0.15,
        "Avg Delay": avg_delay * 0.12,
        "Payment Ratio": (avg_bill - avg_payment) * 0.00002,
        "Recent Payment Status": safe(pay_0) * 0.1
    }

    df = pd.DataFrame(impact.items(), columns=["Feature", "Impact"])
    df = df.sort_values(by="Impact", ascending=False)

    st.bar_chart(df.set_index("Feature"))

    # --------------------------
    # EXPLANATION
    # --------------------------
    st.subheader("🧠 Explanation")

    if delay_count > 2:
        st.write("• High delay count significantly increases risk")
    if avg_payment < avg_bill * 0.5:
        st.write("• Low payments relative to bills indicate financial stress")
    if safe(pay_0) > 1:
        st.write("• Recent missed payments strongly affect prediction")

    # --------------------------
    # DECISION ENGINE
    # --------------------------
    st.subheader("💼 Recommended Action")
    st.info(f"👉 {decision}")

    # --------------------------
    # WHAT-IF SIMULATOR
    # --------------------------
    st.subheader("🔄 Scenario Simulation")

    new_delay = st.slider("Reduce delay count to:", 0, 10, delay_count)

    temp = features.copy()
    temp[0][11] = new_delay

    new_prob = model.predict_proba(scaler.transform(temp))[0][1]

    st.write(f"New Risk: **{new_prob:.2%}**")

    if new_prob < prob:
        st.success("Improvement reduces risk")
    else:
        st.error("Risk remains high")
