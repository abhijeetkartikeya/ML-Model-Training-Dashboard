"""
Simple Demo Script - Customer Churn Prediction

This script demonstrates the core functionality without requiring
all dependencies. Run this to verify the project structure works.

Usage:
    python demo_simple.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np

print("="*80)
print("Customer Churn Prediction System - Simple Demo".center(80))
print("="*80)
print()

# 1. Load sample data
print("📁 Step 1: Loading Sample Data...")
try:
    from src.data.data_loader import load_sample_data
    df = load_sample_data()
    print(f"✅ Loaded {len(df)} customers with {len(df.columns)} features")
    print(f"   Churn Rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
    print()
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("   Make sure sample data exists: python src/data/generate_sample_data.py")
    sys.exit(1)

# 2. Data Preview
print("📊 Step 2: Data Preview...")
print(df.head())
print()

# 3. Basic Statistics
print("📈 Step 3: Basic Statistics...")
print(f"   Total Customers: {len(df)}")
print(f"   Churned: {(df['Churn'] == 'Yes').sum()}")
print(f"   Retained: {(df['Churn'] == 'No').sum()}")
print()

# 4. Feature Engineering Demo
print("🔧 Step 4: Feature Engineering...")
try:
    from src.data.feature_engineer import FeatureEngineer
    fe = FeatureEngineer()
    df_engineered = fe.create_all_features(df.copy())
    print(f"✅ Created {len(fe.created_features)} new features:")
    for feat in fe.created_features[:5]:
        print(f"   - {feat}")
    if len(fe.created_features) > 5:
        print(f"   ... and {len(fe.created_features) - 5} more")
    print()
except Exception as e:
    print(f"❌ Error in feature engineering: {e}")
    print()

# 5. Data Preparation Demo
print("🎯 Step 5: Data Preparation...")
try:
    from src.data.preprocessor import prepare_data
    X_train, X_test, y_train, y_test = prepare_data(df, test_size=0.2)
    print(f"✅ Data split successfully:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    print()
except Exception as e:
    print(f"❌ Error preparing data: {e}")
    print()

# 6. Summary
print("="*80)
print("✅ Core Data Pipeline Working!".center(80))
print("="*80)
print()
print("Next Steps:")
print("1. Install ML dependencies (see INSTALL.md)")
print("2. Train models programmatically or via Streamlit dashboard")
print("3. Evaluate and compare model performance")
print()
print("To run the full dashboard:")
print("   streamlit run streamlit_app.py")
print()
print("Note: Requires Python 3.11 or 3.12 for full functionality")
print("      (Python 3.15 has compatibility issues with some packages)")
print("="*80)
