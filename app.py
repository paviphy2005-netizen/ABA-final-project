import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

st.title("📊 MSME Loan Default Prediction Dashboard")

# File check
required_files = ['model.pkl', 'scaler.pkl', 'encoders.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files: {missing_files}")
    st.stop()

# Load files
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Sidebar inputs
st.sidebar.header("Enter MSME Details")

# ✅ ADD AGE (FIX)
Age = st.sidebar.slider("Age", 18, 80, 30)

Income = st.sidebar.slider("Income", 10000, 1000000, 50000)
LoanAmount = st.sidebar.slider("Loan Amount", 1000, 500000, 10000)
CreditScore = st.sidebar.slider("Credit Score", 300, 900, 650)
MonthsEmployed = st.sidebar.slider("Months Employed", 0, 300, 24)
NumCreditLines = st.sidebar.slider("Credit Lines", 1, 10, 3)
InterestRate = st.sidebar.slider("Interest Rate", 1.0, 25.0, 10.0)
LoanTerm = st.sidebar.slider("Loan Term", 6, 360, 60)
DTIRatio = st.sidebar.slider("DTI Ratio", 0.0, 1.0, 0.3)

Education = st.sidebar.selectbox("Education", ["Graduate","Postgraduate","Others"])
EmploymentType = st.sidebar.selectbox("Employment", ["Salaried","Self-Employed"])
MaritalStatus = st.sidebar.selectbox("Marital", ["Single","Married"])
HasMortgage = st.sidebar.selectbox("Mortgage", ["Yes","No"])

# ✅ INCLUDE AGE HERE
input_data = pd.DataFrame({
    'Age':[Age],
    'Income':[Income],
    'LoanAmount':[LoanAmount],
    'CreditScore':[CreditScore],
    'MonthsEmployed':[MonthsEmployed],
    'NumCreditLines':[NumCreditLines],
    'InterestRate':[InterestRate],
    'LoanTerm':[LoanTerm],
    'DTIRatio':[DTIRatio],
    'Education':[Education],
    'EmploymentType':[EmploymentType],
    'MaritalStatus':[MaritalStatus],
    'HasMortgage':[HasMortgage]
})

# Encoding
for col in encoders:
    try:
        if input_data[col][0] in encoders[col].classes_:
            input_data[col] = encoders[col].transform(input_data[col])
        else:
            input_data[col] = 0
    except:
        input_data[col] = 0

# Prediction
if st.button("Predict Risk"):
    try:
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        st.write(f"### Default Probability: {prob*100:.2f}%")

        if prob > 0.5:
            st.error("High Risk")
        else:
            st.success("Low Risk")

        fig = px.pie(values=[prob,1-prob], names=["Risk","Safe"])
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
