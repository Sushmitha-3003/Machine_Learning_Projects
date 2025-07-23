import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("bagging_model.pkl")     
scaler = joblib.load("scaler.pkl")           

st.title("🩺 Diabetes Prediction App")

# User input
st.subheader("Enter Patient Information:")
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=0)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    
    # Scale input
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]  # Probability of class 1 (diabetic)

    # Show result
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error(f"🧪 The model predicts that the patient **has diabetes** with a probability of {probability:.2f}.")
    else:
        st.success(f"✅ The model predicts that the patient **does not have diabetes** with a probability of {1 - probability:.2f}.")
