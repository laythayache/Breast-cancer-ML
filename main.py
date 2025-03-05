#!/usr/bin/env python
"""
main.py

This script loads the Breast Cancer Wisconsin (Diagnostic) dataset (data.csv),
preprocesses the data, and then trains and evaluates seven different classification models:
1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Gaussian Naive Bayes
6. Multi-Layer Perceptron (MLP)
7. Gradient Boosting

For each model, the script creates a PNG image that displays:
    - A heatmap of the confusion matrix.
    - A text box showing the accuracy and a classification report.

After running all models, the script creates a bar chart comparing their accuracies
and saves it as "model_comparison.png".
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import different model classes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# For reproducibility
RANDOM_STATE = 42

def load_and_preprocess(filepath):
    """
    Load the CSV file and preprocess the data.
    - Drops the 'id' column.
    - Encodes 'diagnosis' (M -> 1, B -> 0).
    - Scales the features.
    
    Returns:
        X_scaled: Scaled features (numpy array)
        y: Target vector (pandas Series)
    """
    df = pd.read_csv(filepath)
    # Drop the non-predictive 'id' column
    df = df.drop(columns=['id'])
    # Encode diagnosis labels (M: malignant -> 1, B: benign -> 0)
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def evaluate_and_save(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train the given model, evaluate it, and save a figure that shows:
        - A confusion matrix heatmap.
        - A text box with accuracy and classification report.
    
    Args:
        model: The ML model instance.
        model_name: A string name for the model.
        X_train, X_test, y_train, y_test: Training and testing data.
        
    Returns:
        accuracy (float): Accuracy score on the test set.
    """
    # Train model
    model.fit(X_train, y_train)
    # Predict test set
    y_pred = model.predict(X_test)
    
    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a figure with 2 subplots: one for confusion matrix and one for text
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left subplot: Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    
    # Right subplot: Text box with accuracy and classification report
    axes[1].axis('off')  # Turn off axis
    eval_text = f"Accuracy: {acc:.4f}\n\nClassification Report:\n{class_report}"
    # Place text in the middle of the subplot
    axes[1].text(0.05, 0.95, eval_text, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5))
    
    # Save the figure as a PNG file
    filename = f"{model_name.replace(' ', '_').lower()}_evaluation.png"
    plt.suptitle(f"{model_name} Evaluation", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)
    
    print(f"{model_name} - Accuracy: {acc:.4f}")
    return acc

def main():
    # Load and preprocess the data from data.csv
    X, y = load_and_preprocess("data.csv")
    
    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Define the models to be evaluated
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
        "SVM": SVC(random_state=RANDOM_STATE, kernel='rbf', C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(random_state=RANDOM_STATE, hidden_layer_sizes=(100,), max_iter=300),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=100)
    }
    
    # Dictionary to store accuracy scores
    model_accuracies = {}
    
    # Evaluate each model and save its evaluation plot
    for name, model in models.items():
        print(f"Evaluating {name} ...")
        acc = evaluate_and_save(model, name, X_train, X_test, y_train, y_test)
        model_accuracies[name] = acc
    
    # Create a bar chart comparing the accuracies of all models
    plt.figure(figsize=(10, 6))
    model_names = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())
    sns.barplot(x=accuracies, y=model_names, palette="viridis")
    plt.xlabel("Accuracy")
    plt.title("Comparison of Model Accuracies")
    for i, v in enumerate(accuracies):
        plt.text(v + 0.005, i, f"{v:.4f}", color='black', va="center")
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    
    print("\nAll model evaluations complete. Comparison chart saved as 'model_comparison.png'.")

if __name__ == "__main__":
    main()
