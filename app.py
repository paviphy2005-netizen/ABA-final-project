import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import os

st.title("📊 MSME Loan Default Prediction Dashboard")

# ✅ CHECK FILES FIRST (VERY IMPORTANT)
required_files = ['model.pkl', 'scaler.pkl', 'encoders.pkl']

missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"❌ Missing files in GitHub repo: {missing_files}")
    st.info("👉 Upload model.pkl, scaler.pkl, encoders.pkl to your repository")
    st.stop()

# ✅ LOAD FILES SAFELY
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

st.success("✅ Model loaded successfully")

# Sidebar Inputs
st.sidebar.header("Enter MSME Details")

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

# Input Data
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

# ✅ SAFE ENCODING
for col in encoders:
    try:
        if input_data[col][0] in encoders[col].classes_:
            input_data[col] = encoders[col].transform(input_data[col])
        else:
            input_data[col] = 0
    except Exception:
        input_data[col] = 0

# Prediction
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

        # Pie chart
        fig = px.pie(
            values=[prob, 1-prob],
            names=["Default Risk", "Safe"],
            title="Risk Distribution"
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Visualization
st.subheader("📊 Input Feature Visualization")

data = pd.DataFrame({
    'Feature': input_data.columns,
    'Value': input_data.iloc[0]
})

fig_bar = px.bar(data, x='Feature', y='Value', title="Feature Distribution")
st.plotly_chart(fig_bar)

fig_line = px.line(data, x='Feature', y='Value', title="Trend View")
st.plotly_chart(fig_line)
