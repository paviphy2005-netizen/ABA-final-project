import streamlit as st
import pandas as pd
import pickle

st.title("MSME Loan Risk Debug")

try:
    model = pickle.load(open('model (2).pkl','rb'))
    scaler = pickle.load(open('scaler (2).pkl','rb'))
    encoders = pickle.load(open('encoders (3).pkl','rb'))

    st.success("Files loaded successfully")

    # Dummy input to test
    input_data = pd.DataFrame({
        'Income':[50000],
        'LoanAmount':[10000],
        'CreditScore':[650],
        'MonthsEmployed':[24],
        'NumCreditLines':[3],
        'InterestRate':[10],
        'LoanTerm':[60],
        'DTIRatio':[0.3],
        'Education':['Graduate'],
        'EmploymentType':['Salaried'],
        'MaritalStatus':['Single'],
        'HasMortgage':['No']
    })

    # Encoding
    for col in encoders:
        input_data[col] = encoders[col].transform(input_data[col])

    # Scaling + prediction
    scaled = scaler.transform(input_data)
    prob = model.predict_proba(scaled)[0][1]

    st.write("Prediction working:", prob)

except Exception as e:
    st.error(f"ERROR: {e}")
