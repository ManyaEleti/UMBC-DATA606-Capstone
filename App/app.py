import streamlit as st
import numpy as np
import joblib

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide"
)

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("../model.pkl")
scaler = joblib.load("../scaler.pkl")

# -------------------------------
# HEADER
# -------------------------------
st.title("💳 Credit Default Risk Predictor")
st.markdown("Predict whether a customer is likely to default on their credit card payment.")

st.divider()

# -------------------------------
# LAYOUT (COLUMNS)
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# LEFT COLUMN (CUSTOMER INFO)
# -------------------------------
with col1:
    st.subheader("👤 Customer Information")

    limit_bal = st.number_input("Credit Limit", value=20000)
    age = st.number_input("Age", value=30)

    sex = st.selectbox("Sex", [1, 2])
    education = st.selectbox("Education Level", [1, 2, 3, 4])
    marriage = st.selectbox("Marital Status", [1, 2, 3])

# -------------------------------
# RIGHT COLUMN (BEHAVIOR)
# -------------------------------
with col2:
    st.subheader("📊 Financial Behavior")

    pay_0 = st.slider("PAY_0", -2, 8, 0)
    pay_2 = st.slider("PAY_2", -2, 8, 0)
    pay_3 = st.slider("PAY_3", -2, 8, 0)
    pay_4 = st.slider("PAY_4", -2, 8, 0)
    pay_5 = st.slider("PAY_5", -2, 8, 0)
    pay_6 = st.slider("PAY_6", -2, 8, 0)

    avg_bill = st.number_input("Avg Bill Amount", value=5000.0)
    avg_payment = st.number_input("Avg Payment", value=2000.0)

    avg_delay = st.number_input("Avg Delay", value=0.0)
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
# PREDICTION BUTTON
# -------------------------------
if st.button("🚀 Predict Risk"):

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.divider()

    st.subheader("📊 Prediction Result")

    # -------------------------------
    # RISK DISPLAY
    # -------------------------------
    if prediction == 1:
        st.error(f"⚠️ High Risk of Default")
    else:
        st.success(f"✅ Low Risk of Default")

    # -------------------------------
    # PROGRESS BAR (RISK METER)
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
