import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

# -------------------------------
# COLOR PALETTE (FINTECH STYLE)
# -------------------------------
PRIMARY = "#0f172a"   # dark navy
BLUE = "#1e3a8a"      # deep blue
ACCENT = "#2563eb"    # bright blue
RED = "#dc2626"       # risk
GREEN = "#059669"     # safe
GRAY = "#64748b"      # neutral

# -------------------------------
# CLEAN UI CSS
# -------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
}

.header {
    background: linear-gradient(135deg, #0f172a, #1e3a8a);
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white;
    border-radius: 10px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="header">💳 Credit Risk Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown("AI-powered system to detect **high-risk customers** using machine learning")
st.markdown("---")

# =========================================================
# 📊 DASHBOARD (TOP)
# =========================================================
st.markdown("## 📊 Model Performance Dashboard")

k1, k2, k3 = st.columns(3)
k1.metric("Accuracy", "81%")
k2.metric("Recall", "56%")
k3.metric("Customers", "30K+")

st.markdown("---")

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.markdown("### 📊 Key Risk Drivers")

importance = pd.DataFrame({
    "Feature": ["Delay Count", "Avg Delay", "Avg Payment", "Avg Bill"],
    "Importance": [0.35, 0.25, 0.20, 0.20]
})

fig1 = px.bar(
    importance,
    x="Feature",
    y="Importance",
    color="Importance",
    color_continuous_scale=[PRIMARY, BLUE, ACCENT]
)

fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# MODEL COMPARISON
# -------------------------------
st.markdown("### 📊 Model Comparison")

df = pd.DataFrame({
    "Model": ["Logistic", "Balanced Logistic", "Random Forest"],
    "Accuracy": [0.81, 0.76, 0.82],
    "Recall": [0.29, 0.56, 0.48],
    "F1": [0.43, 0.51, 0.53]
})

df_melt = df.melt(id_vars="Model")

fig2 = px.bar(
    df_melt,
    x="Model",
    y="value",
    color="variable",
    barmode="group",
    color_discrete_map={
        "Accuracy": BLUE,
        "Recall": RED,
        "F1": GREEN
    }
)

fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
st.markdown("### 📊 Confusion Matrix")

cm = np.array([[3828, 845],
               [583, 744]])

fig3 = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale=["#e2e8f0", ACCENT, PRIMARY]
)

fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# ROC CURVE
# -------------------------------
st.markdown("### 📈 ROC Curve")

roc_df = pd.DataFrame({
    "FPR": [0, 0.1, 0.2, 0.3, 1],
    "TPR": [0, 0.5, 0.65, 0.75, 1]
})

fig4 = px.line(roc_df, x="FPR", y="TPR")
fig4.update_traces(line=dict(color=BLUE, width=4))

fig4.add_shape(
    type="line",
    line=dict(color=GRAY, dash="dash"),
    x0=0, x1=1, y0=0, y1=1
)

fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# =========================================================
# INPUT SECTION
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")
    limit_bal = st.number_input("Credit Limit", value=0)
    age = st.slider("Age", 18, 75, 25)

    sex = st.selectbox("Sex", ["Select...", "Male", "Female"])
    education = st.selectbox("Education",
        ["Select...", "Graduate School", "University", "High School", "Other"])
    marriage = st.selectbox("Marital Status",
        ["Select...", "Married", "Single", "Other"])

with col2:
    st.subheader("📊 Financial Behavior")

    options = ["Select...", "On Time", "1 Month Delay", "2+ Months Delay"]

    def encode(x):
        return {"On Time": 0, "1 Month Delay": 1, "2+ Months Delay": 2}.get(x, 0)

    pay_0 = encode(st.selectbox("Last Month", options))
    pay_2 = encode(st.selectbox("2 Months Ago", options))
    pay_3 = encode(st.selectbox("3 Months Ago", options))
    pay_4 = encode(st.selectbox("4 Months Ago", options))
    pay_5 = encode(st.selectbox("5 Months Ago", options))
    pay_6 = encode(st.selectbox("6 Months Ago", options))

    avg_bill = st.number_input("Average Bill", value=0.0)
    avg_payment = st.number_input("Average Payment", value=0.0)
    avg_delay = st.number_input("Average Delay", value=0.0)
    delay_count = st.number_input("Delay Count", value=0)

# -------------------------------
# FEATURE VECTOR
# -------------------------------
features = np.array([[limit_bal, age,
    pay_0, pay_2, pay_3, pay_4, pay_5, pay_6,
    avg_bill, avg_payment, avg_delay, delay_count,
    1 if sex=="Female" else 0,
    1 if education=="University" else 0,
    1 if education=="High School" else 0,
    1 if education=="Other" else 0,
    1 if marriage=="Single" else 0,
    1 if marriage=="Other" else 0
]])

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict Risk"):
    scaled = scaler.transform(features)
    prob = model.predict_proba(scaled)[0][1]

    st.markdown("## 📊 Prediction Result")

    if prob > 0.7:
        st.error("🔴 High Risk")
    elif prob > 0.4:
        st.warning("🟠 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    st.metric("Default Probability", f"{prob:.2%}")
