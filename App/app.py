import streamlit as st
import numpy as np
import os
import joblib

# -------------------------------
# PAGE CONFIG (LIGHT MODE)
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    layout="wide",
)

# -------------------------------
# CUSTOM LIGHT THEME
# -------------------------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #1f4e79;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.stButton>button:hover {
    background-color: #125a94;
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
st.title("💳 Credit Default Risk Predictor")
st.markdown("""
This application predicts whether a customer is likely to default on their credit card payment.

**Key Idea:**  
We prioritize **recall for defaulters** (high-risk customers), not just accuracy.
""")

st.divider()

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Information")

    limit_bal = st.number_input("Credit Limit", value=20000)
    age = st.number_input("Age", value=30)

    sex = st.selectbox("Sex", [1, 2])
    education = st.selectbox("Education Level", [1, 2, 3, 4])
    marriage = st.selectbox("Marital Status", [1, 2, 3])

with col2:
    st.subheader("📊 Financial Behavior")

    pay_0 = st.slider("Recent Payment Status (PAY_0)", -2, 8, 0)
    pay_2 = st.slider("PAY_2", -2, 8, 0)
    pay_3 = st.slider("PAY_3", -2, 8, 0)
    pay_4 = st.slider("PAY_4", -2, 8, 0)
    pay_5 = st.slider("PAY_5", -2, 8, 0)
    pay_6 = st.slider("PAY_6", -2, 8, 0)

    avg_bill = st.number_input("Average Bill Amount", value=5000.0)
    avg_payment = st.number_input("Average Payment", value=2000.0)

    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

st.divider()

# -------------------------------
# ENCODING
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
    st.subheader("📊 Prediction Result")

    # -------------------------------
    # RESULT
    # -------------------------------
    if prediction == 1:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk of Default")

    # -------------------------------
    # PROBABILITY
    # -------------------------------
    st.write("### Risk Probability")
    st.progress(float(probability))

    st.metric(
        label="Default Probability",
        value=f"{probability:.2%}"
    )

    # -------------------------------
    # INTERPRETATION
    # -------------------------------
    if probability > 0.7:
        st.warning("Very high risk — strong chance of default.")
    elif probability > 0.4:
        st.info("Moderate risk — monitor customer behavior.")
    else:
        st.success("Low risk — customer is likely safe.")

    # -------------------------------
    # FEATURE INSIGHTS
    # -------------------------------
    st.divider()
    st.subheader("🔍 Key Risk Drivers")

    reasons = []

    if pay_0 > 2:
        reasons.append("Recent payment delays are high")
    if delay_count > 3:
        reasons.append("Frequent past delays detected")
    if avg_payment < avg_bill:
        reasons.append("Payments are lower than bills")
    if avg_delay > 1:
        reasons.append("Average delay is significant")

    if reasons:
        for r in reasons:
            st.write(f"• {r}")
    else:
        st.write("No major risk signals detected")

# -------------------------------
# MODEL INSIGHT SECTION
# -------------------------------
st.divider()
st.subheader("📈 Model Insight")

st.info("""
This model demonstrates how **feature engineering improves real-world performance**.

• Accuracy remains ~81%  
• Default detection (recall) improved from **29% → 56%**  

👉 This shows that **accuracy alone is misleading** in imbalanced datasets.  
👉 Detecting high-risk customers is more important than overall accuracy.
""")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built for Capstone Project • Machine Learning + Streamlit Deployment")
