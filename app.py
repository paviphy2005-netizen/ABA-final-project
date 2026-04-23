import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("📊 MSME Loan Default Prediction Dashboard")

# Sidebar Inputs
st.sidebar.header("Enter MSME Details")

Age = st.sidebar.slider("Age", 18, 80, 30)
Income = st.sidebar.slider("Income", 10000, 1000000, 50000)
LoanAmount = st.sidebar.slider("Loan Amount", 1000, 500000, 10000)
CreditScore = st.sidebar.slider("Credit Score", 300, 900, 650)
MonthsEmployed = st.sidebar.slider("Months Employed", 0, 300, 24)
NumCreditLines = st.sidebar.slider("Number of Credit Lines", 1, 10, 3)
InterestRate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.0)
LoanTerm = st.sidebar.slider("Loan Term (Months)", 6, 360, 60)
DTIRatio = st.sidebar.slider("DTI Ratio", 0.0, 1.0, 0.3)

Education = st.sidebar.selectbox("Education", ["Graduate", "Postgraduate", "Others"])
EmploymentType = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-Employed"])
MaritalStatus = st.sidebar.selectbox("Marital Status", ["Single", "Married"])
HasMortgage = st.sidebar.selectbox("Has Mortgage", ["Yes", "No"])
HasDependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
LoanPurpose = st.sidebar.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
HasCoSigner = st.sidebar.selectbox("Has Co-Signer", ["No", "Yes"])

# Encode manually (must match training)
edu_map = {"Graduate":0, "Postgraduate":1, "Others":2}
emp_map = {"Salaried":0, "Self-Employed":1}
mar_map = {"Single":0, "Married":1}
mort_map = {"No":0, "Yes":1}
dep_map = {"No":0, "Yes":1}
pur_map = {"Auto":0, "Business":1, "Education":2, "Home":3, "Other":4}
cos_map = {"No":0, "Yes":1}

# Input dataframe
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
    'Education':[edu_map[Education]],
    'EmploymentType':[emp_map[EmploymentType]],
    'MaritalStatus':[mar_map[MaritalStatus]],
    'HasMortgage':[mort_map[HasMortgage]],
    'HasDependents':[dep_map[HasDependents]],
    'LoanPurpose':[pur_map[LoanPurpose]],
    'HasCoSigner':[cos_map[HasCoSigner]]
})

# Scale
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Risk"):
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📌 Default Probability")
    st.write(f"{prob*100:.2f}%")

    if prob > 0.5:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk")

    # Gauge Chart
    fig = px.pie(
        values=[prob, 1-prob],
        names=["Default Risk", "Safe"],
        title="Risk Distribution"
    )
    st.plotly_chart(fig)

# Visualization Section
st.subheader("📊 Interactive Visualizations")

# Dummy dataset for demo visuals
data = pd.DataFrame({
    'Feature': input_data.columns,
    'Value': input_data.iloc[0]
})

# Bar chart
fig_bar = px.bar(data, x='Feature', y='Value', title="Input Feature Distribution")
st.plotly_chart(fig_bar)

# Line chart
fig_line = px.line(data, x='Feature', y='Value', title="Trend View")
st.plotly_chart(fig_line)