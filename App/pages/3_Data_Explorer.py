import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(BASE_DIR, "../../Data/credit_feature_engineered.csv"))

st.title("📂 Data Explorer")

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Distribution of Default")
st.bar_chart(df["default"].value_counts())

st.write("### Correlation")
st.write(df.corr())
