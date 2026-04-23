import streamlit as st
import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "../model.pkl"))

st.title("📊 Model Insights")

st.markdown("### Feature Importance (What drives risk?)")

if hasattr(model, "coef_"):
    importance = model.coef_[0]
else:
    importance = model.feature_importances_

features = [
    "limit_bal", "age", "pay_0", "pay_2",
    "avg_bill", "avg_payment"
]

df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(df.set_index("Feature"))

st.markdown("""
### 🧠 Interpretation

- Payment delays are the strongest predictors
- Financial behavior > demographics
- Feature engineering improved recall significantly
""")
