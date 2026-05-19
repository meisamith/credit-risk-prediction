import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Default Risk", page_icon="🏦", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load('models/credit_risk_model.joblib')

model = load_model()

st.title("🏦 Credit Default Risk Predictor")
st.caption("Lending Club data · XGBoost · ROC-AUC 0.72")

st.subheader("Applicant Details")

col1, col2 = st.columns(2)
with col1:
    loan_amnt = st.number_input("Loan Amount ($)", 500, 40000, 10000, 500)
    annual_inc = st.number_input("Annual Income ($)", 0, 500000, 60000, 1000)
    dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0)
    fico = st.slider("FICO Score", 600, 850, 700)
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 13.0)

with col2:
    term = st.selectbox("Loan Term (months)", [36, 60])
    grade = st.selectbox("Loan Grade", ['A','B','C','D','E','F','G'])
    emp_length = st.selectbox("Employment Length (years)", [0,1,2,3,4,5,6,7,8,9,10])
    home_ownership = st.selectbox("Home Ownership", ['RENT','MORTGAGE','OWN','OTHER'])
    purpose = st.selectbox("Loan Purpose", ['debt_consolidation','credit_card','home_improvement','other','major_purchase','small_business'])

if st.button("Assess Risk", type="primary"):
    input_data = pd.DataFrame([{
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate,
        'installment': round((loan_amnt * (int_rate/1200)) / (1 - (1 + int_rate/1200)**-term), 2),
        'grade': grade, 'emp_length': float(emp_length),
        'home_ownership': home_ownership, 'annual_inc': annual_inc,
        'verification_status': 'Not Verified', 'purpose': purpose,
        'dti': dti, 'delinq_2yrs': 0.0, 'fico_range_low': float(fico - 4),
        'fico_range_high': float(fico), 'inq_last_6mths': 0.0,
        'open_acc': 8.0, 'pub_rec': 0.0, 'revol_bal': 10000.0,
        'revol_util': 40.0, 'total_acc': 20.0, 'mort_acc': 0.0,
        'pub_rec_bankruptcies': 0.0
    }])

    proba = model.predict_proba(input_data)[0][1]

    if proba < 0.15:
        band, color = "🟢 Low Risk", "success"
    elif proba < 0.40:
        band, color = "🟡 Medium Risk", "warning"
    else:
        band, color = "🔴 High Risk", "error"

    st.metric("Probability of Default", f"{proba:.1%}")
    getattr(st, color)(f"{band} — {'Recommend approval' if proba < 0.40 else 'Recommend rejection'}")

st.divider()
st.caption("Built by Amith Choudhary · JSS STU · github.com/meisamith/credit-risk-prediction")
