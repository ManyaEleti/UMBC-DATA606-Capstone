import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# =========================
# CSS (FINAL POLISHED UI)
# =========================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fafc, #eef2f7);
}

/* Header */
.main-title {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    padding: 32px;
    border-radius: 16px;
    color: white;
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #475569;
    margin-top: 8px;
    margin-bottom: 20px;
}

/* Cards */
.section-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    margin-bottom: 10px;
}

/* Prevent empty cards */
.section-card:empty {
    display: none;
}

/* Text */
h1, h2, h3 {
    color: #0f172a !important;
}
label, p {
    color: #334155 !important;
}

/* =========================
   UNIFIED INPUT STYLING
   ========================= */

div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input, select, textarea {

    background-color: #dbe2ea !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}

/* Fix inner input */
input {
    background-color: #dbe2ea !important;
}

/* Remove white patches */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: transparent !important;
}

/* Focus effect */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border: 1px solid #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: white;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 20px;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #059669;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">Credit Risk Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Real-time credit risk scoring powered by machine learning</div>',
    unsafe_allow_html=True
)

# =========================
# MODEL PERFORMANCE
# =========================
st.subheader("Model Performance")

models = ["Logistic", "Balanced Logistic", "Random Forest"]
accuracy = [0.81, 0.76, 0.82]
recall = [0.29, 0.56, 0.48]
f1 = [0.43, 0.51, 0.53]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        name='Accuracy',
        x=models,
        y=accuracy,
        marker_color='#3b82f6',
        width=0.25
    )
)

fig.add_trace(
    go.Bar(
        name='Recall',
        x=models,
        y=recall,
        marker_color='#10b981',
        width=0.25
    )
)

fig.add_trace(
    go.Bar(
        name='F1 Score',
        x=models,
        y=f1,
        marker_color='#8b5cf6',
        width=0.25
    )
)

fig.update_layout(
    barmode='group',
    height=400,
    paper_bgcolor='white',
    plot_bgcolor='white',
    font=dict(color='#1e293b'),
    yaxis=dict(gridcolor='#e2e8f0')
)

st.plotly_chart(fig, use_container_width=True)

best_model = models[recall.index(max(recall))]
st.success(f"Best model for risk detection: {best_model} (highest recall)")

st.divider()

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Customer Profile")

    limit_bal = st.number_input(
        "Credit Limit ($)",
        value=0,
        format="%d"
    )

    age = st.number_input(
        "Age",
        value=25
    )

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "University", "High School", "Other"]
    )

    marriage = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Other"]
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    st.subheader("Financial Behavior")

    pay_0 = st.selectbox(
        "Recent Status",
        [-2, -1, 0, 1, 2, 3]
    )

    pay_2 = st.selectbox(
        "2 Months Ago",
        [-2, -1, 0, 1, 2]
    )

    pay_3 = st.selectbox(
        "3 Months Ago",
        [-2, -1, 0, 1, 2]
    )

    pay_4 = st.selectbox(
        "4 Months Ago",
        [-2, -1, 0, 1, 2]
    )

    pay_5 = st.selectbox(
        "5 Months Ago",
        [-2, -1, 0, 1, 2]
    )

    pay_6 = st.selectbox(
        "6 Months Ago",
        [-2, -1, 0, 1, 2]
    )

    avg_bill = st.number_input(
        "Avg Bill ($)",
        value=0.0,
        format="%.2f"
    )

    avg_payment = st.number_input(
        "Avg Payment ($)",
        value=0.0,
        format="%.2f"
    )

    avg_delay = st.number_input(
        "Avg Delay",
        value=0.0,
        format="%.2f"
    )

    delay_count = st.number_input(
        "Delay Count",
        value=0
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =========================
# ENCODING
# =========================
sex_2 = 1 if sex == "Female" else 0

education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0

marriage_2 = 1 if marriage == "Single" else 0
marriage_3 = 1 if marriage == "Other" else 0

features = np.array([[
    limit_bal,
    age,
    pay_0,
    pay_2,
    pay_3,
    pay_4,
    pay_5,
    pay_6,
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

# =========================
# PREDICTION
# =========================
if st.button("Predict Risk"):

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    prob = model.predict_proba(features_scaled)[0][1]

    st.subheader("Prediction")

    if prediction == 1:
        st.error("High Risk Customer")
    else:
        st.success("Low Risk Customer")

    st.metric(
        "Default Probability (%)",
        f"{prob:.2%}"
    )

    st.progress(float(prob))

    st.subheader("Insight")

    if delay_count > 2:
        st.warning("High delay frequency increases risk")
    else:
        st.success("Stable repayment behavior")

    st.subheader("Recommended Action")

    if prob > 0.7:
        st.error("Reduce credit exposure")

    elif prob > 0.4:
        st.warning("Monitor customer closely")

    else:
        st.success("No action needed")
