"""
Test script to verify Neural Network and XGBoost models work correctly.

This script tests both models independently to identify any issues.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd

print("="*80)
print("Testing Neural Network and XGBoost Models".center(80))
print("="*80)
print()

# Load data
print("📁 Loading sample data...")
from src.data.data_loader import load_sample_data
from src.data.preprocessor import prepare_data, DataPreprocessor

df = load_sample_data()
print(f"✅ Loaded {len(df)} customers")
print()

# Prepare data
print("🎯 Preparing data...")
X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2, drop_columns=['customerID'])

preprocessor = DataPreprocessor(
    imputation_strategy='mean',
    scaling_method='standard',
    encoding_method='label'
)

X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

print(f"✅ Training set: {X_train_processed.shape}")
print(f"✅ Test set: {X_test_processed.shape}")
print()

# Test XGBoost
print("="*80)
print("Testing XGBoost Model".center(80))
print("="*80)
try:
    from src.models.xgboost_model import XGBoostModel
    
    print("🤖 Creating XGBoost model...")
    xgb_model = XGBoostModel()
    
    print("🚀 Training XGBoost model...")
    xgb_results = xgb_model.train(
        X_train_processed,
        y_train.values,
        X_test_processed,
        y_test.values
    )
    
    print(f"✅ XGBoost trained successfully!")
    print(f"   Training accuracy: {xgb_results['train_accuracy']:.4f}")
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_processed)
    y_proba = xgb_model.predict_proba(X_test_processed)
    
    print(f"✅ Predictions made successfully!")
    print(f"   Predicted {len(y_pred)} samples")
    print(f"   Probability shape: {y_proba.shape}")
    print()
    
except Exception as e:
    print(f"❌ XGBoost Error: {e}")
    import traceback
    traceback.print_exc()
    print()

# Test Neural Network
print("="*80)
print("Testing Neural Network Model".center(80))
print("="*80)
try:
    from src.models.neural_network import NeuralNetworkModel
    
    print("🤖 Creating Neural Network model...")
    nn_model = NeuralNetworkModel()
    
    print("🚀 Training Neural Network model (this may take a minute)...")
    nn_results = nn_model.train(
        X_train_processed,
        y_train.values,
        X_test_processed,
        y_test.values,
        epochs=10,  # Reduced for testing
        batch_size=32
    )
    
    print(f"✅ Neural Network trained successfully!")
    print(f"   Training accuracy: {nn_results['train_accuracy']:.4f}")
    print(f"   Training loss: {nn_results['train_loss']:.4f}")
    print(f"   Epochs trained: {nn_results['epochs_trained']}")
    
    # Make predictions
    y_pred = nn_model.predict(X_test_processed)
    y_proba = nn_model.predict_proba(X_test_processed)
    
    print(f"✅ Predictions made successfully!")
    print(f"   Predicted {len(y_pred)} samples")
    print(f"   Probability shape: {y_proba.shape}")
    print()
    
except Exception as e:
    print(f"❌ Neural Network Error: {e}")
    import traceback
    traceback.print_exc()
    print()

print("="*80)
print("Test Complete!".center(80))
print("="*80)
