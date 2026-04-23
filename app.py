import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
encoders = pickle.load(open('encoders.pkl','rb'))

st.title("📊 MSME Loan Default Dashboard")

# Inputs
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

# Dataframe
input_data = pd.DataFrame({
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

# Encode
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

# Predict
if st.button("Predict Risk"):
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"### Default Probability: {prob*100:.2f}%")

    if prob > 0.5:
        st.error("High Risk")
    else:
        st.success("Low Risk")

    fig = px.pie(values=[prob,1-prob], names=["Risk","Safe"])
    st.plotly_chart(fig)
