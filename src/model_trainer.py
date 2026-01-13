"""
Model Trainer for Churn Prediction

Trains Logistic Regression and Random Forest models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("🚀 Training Logistic Regression...")
    
    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_acc = model.score(X_train, y_train)
    print(f"✅ Training Accuracy: {train_acc:.4f}")
    
    return model


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("🚀 Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Training accuracy
    train_acc = model.score(X_train, y_train)
    print(f"✅ Training Accuracy: {train_acc:.4f}")
    
    return model


def make_predictions(model, X_test):
    """
    Make predictions with a trained model.
    
    Args:
        model: Trained classifier
        X_test: Test features
        
    Returns:
        predictions, probabilities
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    return predictions, probabilities
