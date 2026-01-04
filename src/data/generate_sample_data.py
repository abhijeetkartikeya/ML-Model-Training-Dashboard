"""
Generate Sample Customer Churn Dataset

Creates a realistic telecom customer churn dataset for demonstration.

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

# Demographics
gender = np.random.choice(['Male', 'Female'], n_samples)
senior_citizen = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48])
dependents = np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70])

# Account information
tenure = np.random.exponential(scale=20, size=n_samples).astype(int)
tenure = np.clip(tenure, 0, 72)

contract = np.random.choice(
    ['Month-to-month', 'One year', 'Two year'],
    n_samples,
    p=[0.55, 0.21, 0.24]
)

paperless_billing = np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41])
payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    n_samples,
    p=[0.34, 0.23, 0.22, 0.21]
)

# Services
phone_service = np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == 'Yes',
    np.random.choice(['Yes', 'No'], n_samples, p=[0.42, 0.58]),
    'No phone service'
)

internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22])

# Internet-dependent services
def generate_internet_service(internet_service_array, service_prob=0.5):
    result = []
    for internet in internet_service_array:
        if internet == 'No':
            result.append('No internet service')
        else:
            result.append(np.random.choice(['Yes', 'No'], p=[service_prob, 1-service_prob]))
    return np.array(result)

online_security = generate_internet_service(internet_service, 0.29)
online_backup = generate_internet_service(internet_service, 0.34)
device_protection = generate_internet_service(internet_service, 0.34)
tech_support = generate_internet_service(internet_service, 0.29)
streaming_tv = generate_internet_service(internet_service, 0.38)
streaming_movies = generate_internet_service(internet_service, 0.39)

# Charges
monthly_charges = np.random.uniform(18, 120, n_samples)
monthly_charges = np.round(monthly_charges, 2)

# Adjust charges based on services
monthly_charges = np.where(internet_service == 'Fiber optic', monthly_charges * 1.2, monthly_charges)
monthly_charges = np.where(contract == 'Two year', monthly_charges * 0.9, monthly_charges)
monthly_charges = np.round(monthly_charges, 2)

# Total charges
total_charges = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
total_charges = np.maximum(total_charges, monthly_charges)
total_charges = np.round(total_charges, 2)

# Some customers have missing total charges (new customers)
total_charges_str = total_charges.astype(str)
new_customer_mask = tenure == 0
total_charges_str[new_customer_mask] = ' '  # Empty string for new customers

# Churn (target variable)
# Higher churn probability for:
# - Month-to-month contracts
# - Fiber optic users (higher price)
# - Electronic check payment
# - No online security
# - Short tenure

churn_prob = np.zeros(n_samples)

# Base probability
churn_prob += 0.15

# Contract influence
churn_prob += np.where(contract == 'Month-to-month', 0.30, 0)
churn_prob += np.where(contract == 'One year', 0.05, 0)

# Tenure influence (inverse relationship)
churn_prob += np.where(tenure < 6, 0.25, 0)
churn_prob += np.where((tenure >= 6) & (tenure < 12), 0.15, 0)
churn_prob -= np.where(tenure > 24, 0.20, 0)

# Service influence
churn_prob += np.where(internet_service == 'Fiber optic', 0.15, 0)
churn_prob += np.where(online_security == 'No', 0.10, 0)
churn_prob += np.where(tech_support == 'No', 0.08, 0)

# Payment method influence
churn_prob += np.where(payment_method == 'Electronic check', 0.12, 0)

# Senior citizen
churn_prob += np.where(senior_citizen == 1, 0.05, 0)

# Clip probabilities
churn_prob = np.clip(churn_prob, 0, 0.85)

# Generate churn
churn = np.random.binomial(1, churn_prob)
churn = np.where(churn == 1, 'Yes', 'No')

# Create DataFrame
df = pd.DataFrame({
    'customerID': customer_ids,
    'gender': gender,
    'SeniorCitizen': senior_citizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges_str,
    'Churn': churn
})

# Save to CSV
output_path = Path(__file__).parent.parent.parent / 'data' / 'sample_churn_data.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Sample dataset created: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nChurn distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn rate: {(df['Churn'] == 'Yes').sum() / len(df) * 100:.2f}%")
