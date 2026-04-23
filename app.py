import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model components
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

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

# Create input dataframe (RAW values first)
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
    'HasMortgage':[HasMortgage],
    'HasDependents':[HasDependents],
    'LoanPurpose':[LoanPurpose],
    'HasCoSigner':[HasCoSigner]
})

# Apply encoding (same as training)
for col in encoders:
    input_data[col] = encoders[col].transform(input_data[col])

# Prediction button
if st.button("Predict Risk"):
    try:
        input_scaled = scaler.transform(input_data)
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("📌 Default Probability")
        st.write(f"{prob*100:.2f}%")

        if prob > 0.5:
            st.error("⚠️ High Risk of Default")
        else:
            st.success("✅ Low Risk")

        # Pie Chart
        fig = px.pie(
            values=[prob, 1-prob],
            names=["Default Risk", "Safe"],
            title="Risk Distribution"
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error: {e}")

# Visualization Section
st.subheader("📊 Interactive Visualizations")

data = pd.DataFrame({
    'Feature': input_data.columns,
    'Value': input_data.iloc[0]
})

# Bar Chart
fig_bar = px.bar(data, x='Feature', y='Value', title="Input Feature Distribution")
st.plotly_chart(fig_bar)

# Line Chart
fig_line = px.line(data, x='Feature', y='Value', title="Trend View")
st.plotly_chart(fig_line)
