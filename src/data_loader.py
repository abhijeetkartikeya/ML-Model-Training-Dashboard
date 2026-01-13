"""
Simple Data Loader for Churn Prediction

Loads customer data from CSV files.
"""

import pandas as pd
from pathlib import Path


def load_sample_data():
    """
    Load the sample churn dataset.
    
    Returns:
        DataFrame with customer data
    """
    # Path to sample data
    data_path = Path(__file__).parent.parent / "data" / "sample_churn_data.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Sample data not found at: {data_path}")
    
    # Load CSV
    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} customers with {len(df.columns)} features")
    
    return df


def load_custom_data(filepath):
    """
    Load customer data from a custom CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with customer data
    """
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} customers from {filepath}")
    
    return df
