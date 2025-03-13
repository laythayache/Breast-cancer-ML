import sys
import numpy as np
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit,
                             QPushButton, QVBoxLayout, QWidget, QMessageBox, QGridLayout)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Load model and scaler
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names for the input fields
feature_names = [
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean',
    'Smoothness Mean', 'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE', 'Concave Points SE', 
    'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst',
    'Smoothness Worst', 'Compactness Worst', 'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst'
]

class CancerPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cancer Prediction App 🧪")
        self.setGeometry(100, 100, 600, 800)  # Window size
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")  # Dark mode theme

        # Create layout
        layout = QGridLayout()
        self.inputs = []

        # Create input fields for each feature
        for i, feature in enumerate(feature_names):
            label = QLabel(feature)
            label.setFont(QFont("Arial", 10))
            label.setStyleSheet("color: #cccccc; padding: 2px;")
            layout.addWidget(label, i // 2, (i % 2) * 2)

            entry = QLineEdit()
            entry.setStyleSheet("background-color: #2e2e2e; color: #ffffff; border: 1px solid #555555; padding: 4px;")
            layout.addWidget(entry, i // 2, (i % 2) * 2 + 1)
            self.inputs.append(entry)

        # Create Predict Button
        self.predict_button = QPushButton("PREDICT")
        self.predict_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.predict_button.setStyleSheet("""
            background-color: #ff9800;
            color: #ffffff;
            padding: 10px;
            border-radius: 5px;
        """)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button, len(feature_names) // 2 + 1, 0, 1, 2)

        # Create Main Container
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def predict(self):
        try:
            # Get input values from user
            input_data = [float(entry.text()) for entry in self.inputs]
            input_data_scaled = scaler.transform([input_data])

            # Make prediction
            prediction = model.predict(input_data_scaled)
            prediction_proba = model.predict_proba(input_data_scaled)[0][prediction[0]]

            if prediction[0] == 1:
                result = f"⚠️ Cancerous (Malignant)\nConfidence: {prediction_proba:.2%}"
                result_type = "Warning"
            else:
                result = f"✅ Non-Cancerous (Benign)\nConfidence: {prediction_proba:.2%}"
                result_type = "Information"

            # Show prediction result in a message box
            QMessageBox.information(self, result_type, result)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid input: {e}")

# Run the App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CancerPredictionApp()
    window.show()
    sys.exit(app.exec_())
