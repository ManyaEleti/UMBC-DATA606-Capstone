st.markdown("""
<style>

/* ===== UNIFIED INPUT STYLE ===== */
div[data-baseweb="input"],
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input, select, textarea {

    background-color: #dbe2ea !important;  /* darker neutral */
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}

/* Fix inner text field */
input {
    background-color: #dbe2ea !important;
}

/* Dropdown selected value */
[data-baseweb="select"] div {
    background-color: #dbe2ea !important;
}

/* Focus effect (VERY IMPORTANT for premium feel) */
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border: 1px solid #2563eb !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15);
}

/* Remove weird white patches */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)
