import streamlit as st
import joblib
import os
import numpy as np

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load the model and scalar safely
model_path = os.path.join(os.path.dirname(__file__), "bagging_model.pkl")
scalar_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")  # You used 'scalar'

model = joblib.load(model_path)
scalar = joblib.load(scalar_path)

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Enter the health details below to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=140, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=33)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])
    
    scaled_input = scalar.transform(input_data)
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts that the person is likely to have diabetes.")
    else:
        st.success("✅ The model predicts that the person is unlikely to have diabetes.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
