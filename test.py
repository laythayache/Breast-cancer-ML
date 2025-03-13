import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make a prediction
def predict():
    try:
        # Get values from input fields
        input_data = [float(entry.get()) for entry in entries]
        input_data_scaled = scaler.transform([input_data])
        
        # Predict
        prediction = model.predict(input_data_scaled)
        prediction_proba = model.predict_proba(input_data_scaled)[0][prediction[0]]
        
        if prediction[0] == 1:
            result = f"Cancerous (Malignant)\nConfidence: {prediction_proba:.2%}"
        else:
            result = f"Non-Cancerous (Benign)\nConfidence: {prediction_proba:.2%}"
        
        messagebox.showinfo("Prediction Result", result)
    
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {e}")

# GUI setup
root = tk.Tk()
root.title("Cancer Prediction App")

# Define input labels (based on feature names)
feature_names = [
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean',
    'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 
    'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst',
    'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst'
]

# Create labels and entry fields
entries = []
for i, feature in enumerate(feature_names):
    tk.Label(root, text=feature).grid(row=i, column=0, padx=5, pady=2)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=2)
    entries.append(entry)

# Predict button
predict_button = tk.Button(root, text="Predict", command=predict, bg="green", fg="white", padx=5, pady=5)
predict_button.grid(row=len(feature_names), column=0, columnspan=2, pady=10)

# Run the app
root.mainloop()
