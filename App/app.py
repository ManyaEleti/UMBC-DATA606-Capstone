import os
import numpy as np
import joblib
import streamlit as st

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

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
Predict whether a customer is likely to default on their credit card payment.

**Key Idea:** We prioritize **recall for defaulters (high-risk customers)** over accuracy.
""")

st.divider()

# -------------------------------
# LAYOUT
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# LEFT COLUMN
# -------------------------------
with col1:
    st.subheader("👤 Customer Profile")

    # Credit Limit (clean input)
    limit_bal = st.text_input("Credit Limit ($)", value="0")
    try:
        limit_bal = float(limit_bal)
    except:
        limit_bal = 0

    # Age (better UX)
    age = st.number_input("Age", min_value=18, max_value=100, value=25)

    # Sex
    sex = st.selectbox("Sex", ["Select...", "Male", "Female"])

    # Education
    education = st.selectbox(
        "Education Level",
        ["Select...", "Graduate School", "University", "High School", "Others"]
    )

    # Marital Status
    marriage = st.selectbox(
        "Marital Status",
        ["Select...", "Married", "Single", "Others"]
    )

# -------------------------------
# PAYMENT STATUS OPTIONS
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
# RIGHT COLUMN
# -------------------------------
with col2:
    st.subheader("📊 Financial Behavior")

    pay_0_label = st.selectbox("Recent Payment Status (Last Month)", ["Select..."] + list(payment_options.keys()))
    pay_2_label = st.selectbox("Payment Status (2 Months Ago)", ["Select..."] + list(payment_options.keys()))
    pay_3_label = st.selectbox("Payment Status (3 Months Ago)", ["Select..."] + list(payment_options.keys()))
    pay_4_label = st.selectbox("Payment Status (4 Months Ago)", ["Select..."] + list(payment_options.keys()))
    pay_5_label = st.selectbox("Payment Status (5 Months Ago)", ["Select..."] + list(payment_options.keys()))
    pay_6_label = st.selectbox("Payment Status (6 Months Ago)", ["Select..."] + list(payment_options.keys()))

    pay_0 = payment_options.get(pay_0_label)
    pay_2 = payment_options.get(pay_2_label)
    pay_3 = payment_options.get(pay_3_label)
    pay_4 = payment_options.get(pay_4_label)
    pay_5 = payment_options.get(pay_5_label)
    pay_6 = payment_options.get(pay_6_label)

    avg_bill = st.number_input("Average Bill Amount ($)", value=0.0)
    avg_payment = st.number_input("Average Payment Amount ($)", value=0.0)
    avg_delay = st.number_input("Average Payment Delay", value=0.0)
    delay_count = st.number_input("Number of Delays", value=0)

st.divider()

# -------------------------------
# ENCODING
# -------------------------------
sex_val = 1 if sex == "Male" else 2 if sex == "Female" else None

edu_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
education_val = edu_map.get(education)

mar_map = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}
marriage_val = mar_map.get(marriage)

# One-hot encoding
sex_2 = 1 if sex_val == 2 else 0

education_2 = 1 if education_val == 2 else 0
education_3 = 1 if education_val == 3 else 0
education_4 = 1 if education_val == 4 else 0

marriage_2 = 1 if marriage_val == 2 else 0
marriage_3 = 1 if marriage_val == 3 else 0

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Risk"):

    if None in [
        sex_val, education_val, marriage_val,
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6
    ]:
        st.error("⚠️ Please complete all required fields.")
    else:
        features = np.array([[
            limit_bal, age,
            pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
            avg_bill, avg_payment, avg_delay, delay_count,
            sex_2, education_2, education_3, education_4,
            marriage_2, marriage_3
        ]])

        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        st.divider()
        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Default")
        else:
            st.success("✅ Low Risk of Default")

        st.write("### Risk Probability")
        st.progress(float(probability))

        st.metric("Default Probability", f"{probability:.2%}")

        if probability > 0.7:
            st.warning("Very high risk — strong chance of default.")
        elif probability > 0.4:
            st.info("Moderate risk — monitor closely.")
        else:
            st.success("Low risk — customer is likely safe.")

# -------------------------------
# FOOTER
# -------------------------------
st.divider()

st.subheader("📈 Model Insight")
st.info("""
This model demonstrates how **feature engineering improves performance**.

• Accuracy ~81%  
• Recall improved significantly  
• Focus on catching high-risk customers  

👉 In finance, missing a defaulter is more costly than a false alarm.
""")
