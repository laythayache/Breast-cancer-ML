import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Cancer Prediction App 🧪")

# Define input fields
feature_names = [
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean',
    'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 
    'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst',
    'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst'
]

# Create inputs
input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)[0][prediction[0]]
    
    if prediction[0] == 1:
        st.error(f"⚠️ Cancerous (Malignant) with confidence: {prediction_proba:.2%}")
    else:
        st.success(f"✅ Non-Cancerous (Benign) with confidence: {prediction_proba:.2%}")
