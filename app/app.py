"""
Credit Default Risk — Demo App (skeleton)

This is intentionally a SKELETON for Phase 1. In Phase 4 we wire it to
the real trained model. It exists now so the project structure is
complete and you can `streamlit run app/app.py` to confirm the env works.

Run:  streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="Credit Default Risk", page_icon="🏦", layout="centered")

st.title("🏦 Credit Default Risk Predictor")
st.caption("Demo skeleton — model wiring happens in Phase 4.")

st.subheader("Applicant details")

col1, col2 = st.columns(2)
with col1:
    loan_amnt = st.number_input("Loan amount (₹/$)", min_value=500, value=10000, step=500)
    annual_inc = st.number_input("Annual income", min_value=0, value=60000, step=1000)
with col2:
    dti = st.slider("Debt-to-income ratio (%)", 0.0, 50.0, 18.0)
    emp_length = st.selectbox("Employment length (years)", ["<1", "1-3", "4-6", "7-9", "10+"])

if st.button("Assess risk", type="primary"):
    # Placeholder logic — replaced by real model.predict_proba in Phase 4.
    st.info(
        "Model not wired yet. After Phase 3 this will show a calibrated "
        "probability of default, a High/Medium/Low band, and the top "
        "factors driving the decision (via SHAP)."
    )

st.divider()
st.markdown(
    "Built by **Amith Choudhary** · Lending Club data · "
    "scikit-learn / XGBoost · [GitHub repo](#)"
)
