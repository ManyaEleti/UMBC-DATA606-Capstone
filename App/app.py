import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #0f172a;
    font-weight: 700;
}

.stButton > button {
    background: linear-gradient(90deg,#2563eb,#4338ca);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg,#1d4ed8,#3730a3);
    color: white;
}

.metric-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

.banner {
    background: linear-gradient(90deg,#2563eb,#1e40af);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="banner">
    <h1 style="color:white;">Credit Risk Intelligence Dashboard</h1>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<center><h4 style='color:#475569;'>Real-time credit risk scoring powered by machine learning</h4></center>",
    unsafe_allow_html=True
)

st.write("")
st.write("")

# ---------------- MODEL PERFORMANCE ----------------

st.subheader("Model Performance")

models = ["Logistic", "Balanced Logistic", "Random Forest"]
accuracy = [0.81, 0.76, 0.82]
recall = [0.29, 0.56, 0.48]
f1 = [0.43, 0.51, 0.53]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=models,
    y=accuracy,
    name="Accuracy"
))

fig.add_trace(go.Bar(
    x=models,
    y=recall,
    name="Recall"
))

fig.add_trace(go.Bar(
    x=models,
    y=f1,
    name="F1 Score"
))

fig.update_layout(
    barmode='group',
    height=500,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.success("Best model for risk detection: Balanced Logistic (highest recall)")

st.divider()

# ---------------- INPUT SECTION ----------------

col1, col2 = st.columns(2)

# ---------- LEFT COLUMN ----------
with col1:

    st.subheader("Customer Profile")

    limit_bal = st.number_input(
        "Credit Limit ($)",
        min_value=0,
        value=1000
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "University", "High School", "Others"]
    )

    marriage = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Others"]
    )

# ---------- RIGHT COLUMN ----------
with col2:

    st.subheader("Financial Behavior")

    status_options = {
        "No Bill / No Usage": -2,
        "Paid Duly": -1,
        "Use of Revolving Credit": 0,
        "1 Month Delay": 1,
        "2 Months Delay": 2,
        "3+ Months Delay": 3
    }

    pay_0_label = st.selectbox(
        "Recent Status",
        list(status_options.keys())
    )

    pay_2_label = st.selectbox(
        "2 Months Ago",
        list(status_options.keys())
    )

    pay_3_label = st.selectbox(
        "3 Months Ago",
        list(status_options.keys())
    )

    pay_4_label = st.selectbox(
        "4 Months Ago",
        list(status_options.keys())
    )

    pay_5_label = st.selectbox(
        "5 Months Ago",
        list(status_options.keys())
    )

    pay_6_label = st.selectbox(
        "6 Months Ago",
        list(status_options.keys())
    )

    avg_bill = st.number_input(
        "Avg Bill ($)",
        min_value=0.0,
        value=200.0
    )

    avg_pay = st.number_input(
        "Avg Payment ($)",
        min_value=0.0,
        value=100.0
    )

    avg_delay = st.number_input(
        "Avg Delay",
        min_value=0.0,
        value=1.0
    )

    delay_count = st.number_input(
        "Delay Count",
        min_value=0,
        value=1
    )

# ---------------- FEATURE ENGINEERING ----------------

sex_val = 1 if sex == "Male" else 2

edu_map = {
    "Graduate": 1,
    "University": 2,
    "High School": 3,
    "Others": 4
}

mar_map = {
    "Married": 1,
    "Single": 2,
    "Others": 3
}

education_val = edu_map[education]
marriage_val = mar_map[marriage]

pay_0 = status_options[pay_0_label]
pay_2 = status_options[pay_2_label]
pay_3 = status_options[pay_3_label]
pay_4 = status_options[pay_4_label]
pay_5 = status_options[pay_5_label]
pay_6 = status_options[pay_6_label]

# ---------------- PREDICTION ----------------

st.write("")
st.write("")

if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        "LIMIT_BAL": limit_bal,
        "SEX": sex_val,
        "EDUCATION": education_val,
        "MARRIAGE": marriage_val,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "AVG_BILL": avg_bill,
        "AVG_PAY": avg_pay,
        "AVG_DELAY": avg_delay,
        "DELAY_COUNT": delay_count
    }])

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.write("")
    st.subheader("Prediction")

    if prediction == 1:
        st.error("High Risk Customer")
    else:
        st.success("Low Risk Customer")

    st.metric(
        label="Default Probability",
        value=f"{probability*100:.2f}%"
    )

    st.progress(float(probability))

    st.subheader("Insight")

    if probability > 0.7:
        st.warning("Customer shows strong signs of repayment risk.")
    elif probability > 0.4:
        st.info("Monitor customer closely.")
    else:
        st.success("Stable repayment behavior.")

    st.subheader("Recommended Action")

    if probability > 0.7:
        st.error("Reduce credit exposure and review account.")
    elif probability > 0.4:
        st.warning("Monitor customer closely.")
    else:
        st.success("Customer eligible for normal credit activity.")
