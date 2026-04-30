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
# STRIPE / BLOOMBERG UI CSS
# =========================
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #0b1220;
}

/* Header */
.main-title {
    background: linear-gradient(135deg, #111827, #1f2937);
    padding: 30px;
    border-radius: 14px;
    color: #f9fafb;
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    border: 1px solid #1f2937;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-top: 8px;
    margin-bottom: 25px;
}

/* Cards */
.section-card {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1f2937;
    margin-bottom: 15px;
}

/* Text */
h1, h2, h3 {
    color: #f9fafb !important;
}
label, p {
    color: #d1d5db !important;
}

/* Inputs */
input, select {
    background-color: #0f172a !important;
    color: #e5e7eb !important;
    border: 1px solid #1f2937 !important;
}

/* Button */
.stButton > button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    font-weight: 600;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #22c55e;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">💳 Credit Risk Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time credit risk scoring powered by machine learning</div>', unsafe_allow_html=True)

# =========================
# MODEL PERFORMANCE (TOP)
# =========================
st.subheader("🏆 Model Performance")

models = ["Logistic", "Balanced Logistic", "Random Forest"]
accuracy = [0.81, 0.76, 0.82]
recall = [0.29, 0.56, 0.48]
f1 = [0.43, 0.51, 0.53]

fig = go.Figure()

fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='#3b82f6', width=0.25))
fig.add_trace(go.Bar(name='Recall', x=models, y=recall, marker_color='#22c55e', width=0.25))
fig.add_trace(go.Bar(name='F1 Score', x=models, y=f1, marker_color='#a78bfa', width=0.25))

fig.update_layout(
    barmode='group',
    height=400,
    paper_bgcolor='#0b1220',
    plot_bgcolor='#0b1220',
    font=dict(color='#e5e7eb'),
    yaxis=dict(gridcolor='#1f2937')
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

    limit_bal = st.number_input("Credit Limit", value=0)
    age = st.number_input("Age", value=25)

    sex = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Married", "Single", "Other"])

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Financial Behavior")

    pay_0 = st.selectbox("Recent Status", [-2, -1, 0, 1, 2, 3])
    pay_2 = st.selectbox("2 Months Ago", [-2, -1, 0, 1, 2])
    pay_3 = st.selectbox("3 Months Ago", [-2, -1, 0, 1, 2])
    pay_4 = st.selectbox("4 Months Ago", [-2, -1, 0, 1, 2])
    pay_5 = st.selectbox("5 Months Ago", [-2, -1, 0, 1, 2])
    pay_6 = st.selectbox("6 Months Ago", [-2, -1, 0, 1, 2])

    avg_bill = st.number_input("Avg Bill", value=0.0)
    avg_payment = st.number_input("Avg Payment", value=0.0)
    avg_delay = st.number_input("Avg Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

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
    limit_bal, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
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

    st.metric("Default Probability", f"{prob:.2%}")
    st.progress(float(prob))

    st.subheader("Insight")

    if delay_count > 2:
        st.warning("High delay frequency is the main risk driver")
    else:
        st.success("Customer behavior looks stable")

    st.subheader("Recommended Action")

    if prob > 0.7:
        st.error("Reduce credit exposure")
    elif prob > 0.4:
        st.warning("Monitor closely")
    else:
        st.success("No action needed")
