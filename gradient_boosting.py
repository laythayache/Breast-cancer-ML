"""
Breast Cancer Diagnosis Classification using Gradient Boosting

This script loads the data, preprocesses it, and trains a Gradient Boosting classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.drop(columns=['id'])
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def main():
    df = load_data("data.csv")
    X, y = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Train Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Gradient Boosting Accuracy: {:.4f}".format(accuracy))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu")
    plt.title("Gradient Boosting Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    main()
