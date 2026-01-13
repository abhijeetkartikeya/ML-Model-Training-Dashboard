"""
Data Preprocessor for Churn Prediction

Cleans and prepares data for machine learning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_data(df, test_size=0.2, drop_columns=None):
    """
    Prepare data for training by splitting into train/test sets.
    
    Args:
        df: Input DataFrame
        test_size: Percentage of data for testing (default 20%)
        drop_columns: List of columns to drop (like customerID)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Make a copy
    data = df.copy()
    
    # Drop unnecessary columns
    if drop_columns:
        data = data.drop(columns=drop_columns, errors='ignore')
    
    # Separate features and target
    if 'Churn' in data.columns:
        # Convert Churn to 0/1
        data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
        
        X = data.drop('Churn', axis=1)
        y = data['Churn']
    else:
        raise ValueError("Dataset must have 'Churn' column")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=42,
        stratify=y
    )
    
    print(f"✅ Train set: {len(X_train)} rows")
    print(f"✅ Test set: {len(X_test)} rows")
    
    return X_train, X_test, y_train, y_test


def preprocess_features(X_train, X_test):
    """
    Preprocess features: encode categories and scale numbers.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train_processed, X_test_processed (as numpy arrays)
    """
    # Make copies
    train = X_train.copy()
    test = X_test.copy()
    
    # Handle missing values in TotalCharges
    if 'TotalCharges' in train.columns:
        train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce')
        test['TotalCharges'] = pd.to_numeric(test['TotalCharges'], errors='coerce')
        
        train['TotalCharges'] = train['TotalCharges'].fillna(train['TotalCharges'].median())
        test['TotalCharges'] = test['TotalCharges'].fillna(train['TotalCharges'].median())
    
    # Identify numeric and categorical columns
    numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        label_encoders[col] = le
    
    # Scale numeric features
    scaler = StandardScaler()
    if numeric_cols:
        train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
        test[numeric_cols] = scaler.transform(test[numeric_cols])
    
    print(f"✅ Preprocessed {len(train.columns)} features")
    
    # Convert to numpy arrays
    return train.values, test.values
