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

model_path = os.path.join(BASE_DIR, "model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Credit Default Risk Predictor")
st.markdown(
    """
Predict whether a customer is likely to default on their credit card payment.

**Key Idea:** We prioritize **recall for defaulters (high-risk customers)** over raw accuracy.
"""
)

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

    limit_bal = st.number_input("Credit Limit ($)", min_value=0, value=0)
    age = st.slider("Age", 18, 100, 18)

    # --- SEX ---
    sex = st.selectbox(
        "Sex",
        ["Select...", "Male", "Female"]
    )

    # --- EDUCATION ---
    education = st.selectbox(
        "Education Level",
        ["Select...", "Graduate School", "University", "High School", "Others"]
    )

    # --- MARRIAGE ---
    marriage = st.selectbox(
        "Marital Status",
        ["Select...", "Married", "Single", "Others"]
    )

# -------------------------------
# RIGHT COLUMN
# -------------------------------
with col2:
    st.subheader("📊 Financial Behavior")

    pay_0 = st.slider("Recent Payment Status (Last Month)", -2, 8, 0)
    pay_2 = st.slider("Payment Status (2 Months Ago)", -2, 8, 0)
    pay_3 = st.slider("Payment Status (3 Months Ago)", -2, 8, 0)
    pay_4 = st.slider("Payment Status (4 Months Ago)", -2, 8, 0)
    pay_5 = st.slider("Payment Status (5 Months Ago)", -2, 8, 0)
    pay_6 = st.slider("Payment Status (6 Months Ago)", -2, 8, 0)

    avg_bill = st.number_input("Average Bill Amount ($)", value=0.0)
    avg_payment = st.number_input("Average Payment Amount ($)", value=0.0)

    avg_delay = st.number_input("Average Payment Delay", value=0.0)
    delay_count = st.number_input("Number of Delays", value=0)

st.divider()

# -------------------------------
# ENCODING
# -------------------------------
sex_val = None
if sex == "Male":
    sex_val = 1
elif sex == "Female":
    sex_val = 2

edu_map = {
    "Graduate School": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}
education_val = edu_map.get(education, None)

mar_map = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}
marriage_val = mar_map.get(marriage, None)

# One-hot encoding
sex_2 = 1 if sex_val == 2 else 0

education_2 = 1 if education_val == 2 else 0
education_3 = 1 if education_val == 3 else 0
education_4 = 1 if education_val == 4 else 0

marriage_2 = 1 if marriage_val == 2 else 0
marriage_3 = 1 if marriage_val == 3 else 0

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

    # Validation
    if None in [sex_val, education_val, marriage_val]:
        st.error("⚠️ Please select all required fields.")
    else:
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        st.divider()
        st.subheader("📊 Prediction Result")

        # Result Display
        if prediction == 1:
            st.error("⚠️ High Risk of Default")
        else:
            st.success("✅ Low Risk of Default")

        # Probability
        st.write("### Risk Probability")
        st.progress(float(probability))

        st.metric(
            label="Default Probability",
            value=f"{probability:.2%}"
        )

        # Interpretation
        if probability > 0.7:
            st.warning("Very high risk — strong chance of default.")
        elif probability > 0.4:
            st.info("Moderate risk — monitor customer behavior.")
        else:
            st.success("Low risk — customer is likely safe.")

# -------------------------------
# FOOTER INSIGHT
# -------------------------------
st.divider()

st.subheader("📈 Model Insight")
st.info(
    """
This model demonstrates how **feature engineering improves real-world performance**.

• Accuracy remains ~81%  
• Default detection (recall) improved significantly  
• Focus is on catching high-risk customers  

👉 In finance, **missing a defaulter is more costly than a false alarm**
"""
)
