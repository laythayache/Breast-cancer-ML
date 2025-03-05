# Breast Cancer Diagnosis Classification: Multiple Models

This repository demonstrates a machine learning project using the Breast Cancer Wisconsin (Diagnostic) dataset. The goal is to classify tumors as malignant (M) or benign (B) using different machine learning models. 

The project includes **7 Python scripts**, each using a different method/algorithm:
1. **logistic_regression.py**: Logistic Regression
2. **random_forest.py**: Random Forest Classifier
3. **svm_classifier.py**: Support Vector Machine (SVM)
4. **knn_classifier.py**: K-Nearest Neighbors (KNN)
5. **naive_bayes.py**: Gaussian Naive Bayes
6. **mlp_classifier.py**: Multi-Layer Perceptron (Neural Network)
7. **gradient_boosting.py**: Gradient Boosting Classifier

## Project Structure

- `data.csv`  
  The CSV file containing the dataset. Make sure it follows the sample format provided.

- `README.md`  
  This file.

- Each model file (e.g., `logistic_regression.py`, etc.) is a standalone script.

## Requirements

- Python 3.6+
- Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

Install the required libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
