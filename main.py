#!/usr/bin/env python
"""
Breast Cancer Classification with Improved Data Handling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_and_preprocess(filepath):
    """
    Load and preprocess data with robust missing value handling
    """
    # Load data with error checking
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"\nInitial dataset shape: {df.shape}")
    print("First 3 rows of raw data:")
    print(df.head(3))

    # Initial preprocessing
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"\nMissing values detected: {missing_values}")
        print("Columns with missing values:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        feature_cols = df.columns[df.columns != 'diagnosis']
        df[feature_cols] = imputer.fit_transform(df[feature_cols])
        print("\nMissing values after imputation:", df.isnull().sum().sum())
    else:
        print("\nNo missing values detected")

    # Validate dataset integrity
    if df.empty:
        raise ValueError("Dataset is empty after preprocessing")
    
    # Encode target variable
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    
    # Final dataset verification
    print("\nProcessed dataset shape:", df.shape)
    print("Class distribution:")
    print(df['diagnosis'].value_counts())
    
    # Split features and target
    X = df.drop(columns=['diagnosis'])
    y = df['diagnosis']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def evaluate_and_save(model, model_name, X_train, X_test, y_train, y_test):
    """Train, evaluate, and save model results"""
    # Training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f'{model_name} Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Metrics display
    ax2.axis('off')
    metrics_text = f"Accuracy: {acc:.4f}\n\nClassification Report:\n{report}"
    ax2.text(0.05, 0.95, metrics_text, fontsize=10, 
            verticalalignment='top', 
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Save figure
    plt.suptitle(f"{model_name} Evaluation", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{model_name.replace(' ', '_')}_results.png")
    plt.close()
    
    return acc

def main():
    """Main execution flow"""
    # Load and prepare data
    X, y = load_and_preprocess("Cancer_Data.csv")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "SVM": SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), 
                                      max_iter=500, 
                                      random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, 
                                                      random_state=RANDOM_STATE)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 40}")
        print(f"Training {name}")
        accuracy = evaluate_and_save(model, name, X_train, X_test, y_train, y_test)
        results[name] = accuracy
        print(f"{name} accuracy: {accuracy:.4f}")
    
    # Create comparison chart
    plt.figure(figsize=(12, 6))
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    sns.barplot(x=list(sorted_results.values()), y=list(sorted_results.keys()), palette="viridis")
    plt.title("Model Performance Comparison")
    plt.xlabel("Accuracy Score")
    plt.xlim(0.8, 1.0)
    
    # Add value labels
    for i, v in enumerate(sorted_results.values()):
        plt.text(v + 0.005, i, f"{v:.4f}", color='black', va='center')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()
    
    print("\nExecution completed successfully!")
    print(f"Best model: {max(results, key=results.get)} ({max(results.values()):.4f})")

if __name__ == "__main__":
    main()