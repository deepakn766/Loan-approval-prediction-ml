import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Approval Prediction System")
st.write("Predict whether a loan application should be **Approved or Rejected** based on applicant financial data.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl", "rb") as f:   # read binary
        return pickle.load(f)

model = load_model()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Applicant Details")

person_gender = st.selectbox("Gender", ["Male", "Female"]).lower()


person_education = st.sidebar.selectbox(
    "Education Level", ["HIGH SCHOOL", "BACHELORS", "MASTER", "PHD", "OTHER"]
).title()

person_income = st.number_input("person_income")

person_emp_exp = st.sidebar.slider(
    "Employment Experience (Years)", 0, 40
)

person_home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
).upper()

loan_amnt = st.number_input("Loan Amount")

loan_int_rate = st.number_input(
    "Interest Rate (%)"
)

loan_intent = st.sidebar.selectbox(
    "Loan Purpose",
    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
).upper()

loan_percent_income = st.number_input(
    "Loan Percent of Income"
)

previous_loan_defaults_on_file = st.sidebar.selectbox(
    "Previous Default on Record", ["YES", "NO"]
).title()

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (Years)"
)

credit_score = st.number_input(
    "Credit Score"
)

# -----------------------------
# Create input dataframe
# -----------------------------
input_data = pd.DataFrame({
    "person_gender": [person_gender],
    "person_education": [person_education],
    "person_income": [person_income],
    "person_emp_exp": [person_emp_exp],
    "person_home_ownership": [person_home_ownership],
    "loan_amnt": [loan_amnt],
    "loan_int_rate": [loan_int_rate],
    "loan_percent_income": [loan_percent_income],
    "loan_intent": [loan_intent],
    "cb_person_cred_hist_length": [cb_person_cred_hist_length],
    "credit_score": [credit_score],
    "previous_loan_defaults_on_file": [previous_loan_defaults_on_file]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.metric(
        label="Approval Probability",
        value=f"{probability * 100:.2f}%"
    )

    st.info(
        "⚠️ This prediction is based on historical data and should be used as a decision-support tool."
    )