"""
Generate Simple 2-Feature Customer Churn Dataset

Creates a minimal dataset with only 2 features for fast training.

Author: Kartikeya
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 7043

# Generate customer IDs
customer_ids = [f"C{str(i).zfill(7)}" for i in range(1, n_samples + 1)]

# Feature 1: Tenure (months with company)
tenure = np.random.exponential(scale=20, size=n_samples).astype(int)
tenure = np.clip(tenure, 0, 72)

# Feature 2: Monthly Charges
monthly_charges = np.random.uniform(18, 120, n_samples)
monthly_charges = np.round(monthly_charges, 2)

# Churn (target variable)
# Simple logic: Higher churn for low tenure + high charges
churn_prob = np.zeros(n_samples)

# Base probability
churn_prob += 0.20

# Tenure influence (inverse relationship)
churn_prob += np.where(tenure < 6, 0.35, 0)
churn_prob += np.where((tenure >= 6) & (tenure < 12), 0.20, 0)
churn_prob -= np.where(tenure > 24, 0.25, 0)

# Monthly charges influence (higher charges = higher churn)
churn_prob += np.where(monthly_charges > 80, 0.15, 0)
churn_prob += np.where(monthly_charges > 100, 0.10, 0)

# Clip probabilities
churn_prob = np.clip(churn_prob, 0, 0.85)

# Generate churn
churn = np.random.binomial(1, churn_prob)
churn = np.where(churn == 1, 'Yes', 'No')

# Create DataFrame with only 3 columns (2 features + target)
df = pd.DataFrame({
    'customerID': customer_ids,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'Churn': churn
})

# Save to CSV
output_path = Path(__file__).parent.parent.parent / 'data' / 'sample_churn_data.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Simple 2-feature dataset created: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nFeatures: tenure, MonthlyCharges")
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
print(f"\nTenure stats:")
print(df['tenure'].describe())
print(f"\nMonthly Charges stats:")
print(df['MonthlyCharges'].describe())
