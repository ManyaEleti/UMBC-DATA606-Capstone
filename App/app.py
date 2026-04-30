import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# -----------------------
# LOAD MODEL
# -----------------------
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -----------------------
# HEADER (DARK GREEN FINTECH)
# -----------------------
st.markdown("""
<style>
.main-title {
    background: linear-gradient(90deg, #064e3b, #065f46);
    padding: 22px;
    border-radius: 14px;
    color: white;
    font-size: 34px;
    font-weight: 700;
    text-align: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.section-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💳 Credit Risk Intelligence Dashboard</div>', unsafe_allow_html=True)

st.markdown("AI-powered system to detect **high-risk customers** using behavioral patterns.")

st.divider()

# =====================================================
# 🏆 MODEL PERFORMANCE (THIN BAR GRAPH)
# =====================================================

st.subheader("🏆 Model Performance Analysis")

models = ["Logistic", "Balanced Logistic", "Random Forest"]

accuracy = [0.81, 0.76, 0.82]
recall = [0.29, 0.56, 0.48]
f1 = [0.43, 0.51, 0.53]

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Accuracy',
    x=models,
    y=accuracy,
    marker_color='#2563eb',
    width=0.25
))

fig.add_trace(go.Bar(
    name='Recall',
    x=models,
    y=recall,
    marker_color='#059669',
    width=0.25
))

fig.add_trace(go.Bar(
    name='F1 Score',
    x=models,
    y=f1,
    marker_color='#7c3aed',
    width=0.25
))

fig.update_layout(
    barmode='group',
    template='plotly_white',
    height=420,
    margin=dict(l=20, r=20, t=40, b=20),
    yaxis=dict(title="Score"),
    xaxis=dict(title="Model"),
    legend_title="Metrics"
)

st.plotly_chart(fig, use_container_width=True)

best_model = models[recall.index(max(recall))]
st.success(f"🏆 Best Model for Risk Detection: {best_model} (Highest Recall)")

st.divider()

# =====================================================
# 👤 INPUT SECTION
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("👤 Customer Profile")

    limit_bal = st.number_input("Credit Limit ($)", value=0)
    age = st.number_input("Age", value=25)

    sex = st.selectbox("Gender", ["Select...", "Male", "Female"])
    education = st.selectbox("Education", ["Select...", "Graduate", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status", ["Select...", "Married", "Single", "Other"])

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📊 Financial Behavior")

    pay_0 = st.selectbox("Recent Payment Status", ["Select...", -2, -1, 0, 1, 2, 3])
    pay_2 = st.selectbox("2 Months Ago", ["Select...", -2, -1, 0, 1, 2])
    pay_3 = st.selectbox("3 Months Ago", ["Select...", -2, -1, 0, 1, 2])
    pay_4 = st.selectbox("4 Months Ago", ["Select...", -2, -1, 0, 1, 2])
    pay_5 = st.selectbox("5 Months Ago", ["Select...", -2, -1, 0, 1, 2])
    pay_6 = st.selectbox("6 Months Ago", ["Select...", -2, -1, 0, 1, 2])

    avg_bill = st.number_input("Average Bill", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Frequency", value=0)

    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =====================================================
# 🔄 ENCODING
# =====================================================

sex_2 = 1 if sex == "Female" else 0
education_2 = 1 if education == "University" else 0
education_3 = 1 if education == "High School" else 0
education_4 = 1 if education == "Other" else 0
marriage_2 = 1 if marriage == "Single" else 0
marriage_3 = 1 if marriage == "Other" else 0

def safe_val(x):
    return 0 if x == "Select..." else x

pay_0 = safe_val(pay_0)
pay_2 = safe_val(pay_2)
pay_3 = safe_val(pay_3)
pay_4 = safe_val(pay_4)
pay_5 = safe_val(pay_5)
pay_6 = safe_val(pay_6)

# =====================================================
# 🚀 PREDICTION
# =====================================================

features = np.array([[
    limit_bal, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    avg_bill, avg_payment, avg_delay, delay_count,
    sex_2, education_2, education_3, education_4,
    marriage_2, marriage_3
]])

if st.button("🚀 Predict Risk"):

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk Customer")
    else:
        st.success("✅ Low Risk Customer")

    st.progress(float(probability))
    st.metric("Default Probability", f"{probability:.2%}")

    # INSIGHT
    st.subheader("🧠 Insight")

    if delay_count > 2:
        st.warning("Frequent delays strongly increase risk")
    elif avg_delay > 1:
        st.info("Moderate delay behavior detected")
    else:
        st.success("Healthy repayment behavior")

    # ACTION
    st.subheader("💼 Recommended Action")

    if probability > 0.7:
        st.error("Reduce credit exposure immediately")
    elif probability > 0.4:
        st.warning("Monitor customer closely")
    else:
        st.success("Customer is safe")
