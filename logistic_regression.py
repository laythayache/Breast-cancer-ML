"""
Breast Cancer Diagnosis Classification using Logistic Regression

This script loads the data, preprocesses it, and trains a Logistic Regression model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Drop the 'id' column
    df = df.drop(columns=['id'])
    # Encode diagnosis: M -> 1, B -> 0
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    # Separate features and target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def main():
    # Load data from CSV
    df = load_data("Cancer_Data.csv")
    X, y = preprocess_data(df)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Train Logistic Regression
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy: {:.4f}".format(accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    main()
