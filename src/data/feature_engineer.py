"""
Feature Engineering Module

Creates domain-specific features for customer churn prediction.

Author: Kartikeya
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestClassifier

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineering for customer churn prediction.
    
    Creates domain-specific features based on customer behavior patterns.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.created_features: List[str] = []
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with new features
        """
        logger.info("Creating engineered features...")
        
        df = df.copy()
        
        # Tenure-based features
        df = self.create_tenure_features(df)
        
        # Spending features
        df = self.create_spending_features(df)
        
        # Service usage features
        df = self.create_service_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        logger.info(f"Created {len(self.created_features)} new features")
        return df
    
    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features.
        
        Features:
        - Tenure groups (new, medium, long-term customers)
        - Is new customer (< 6 months)
        - Is loyal customer (> 24 months)
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with tenure features
        """
        if 'tenure' not in df.columns:
            return df
        
        df = df.copy()
        
        # Tenure groups
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 100],
            labels=['0-1 year', '1-2 years', '2-4 years', '4+ years']
        )
        
        # Binary indicators
        df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
        df['is_loyal_customer'] = (df['tenure'] > 24).astype(int)
        
        self.created_features.extend(['tenure_group', 'is_new_customer', 'is_loyal_customer'])
        logger.info("Created tenure features")
        
        return df
    
    def create_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spending-related features.
        
        Features:
        - Average monthly spending (TotalCharges / tenure)
        - Spending ratio (MonthlyCharges / median)
        - High spender indicator
        - Charges per service
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with spending features
        """
        if 'MonthlyCharges' not in df.columns:
            return df
        
        df = df.copy()
        
        # Average monthly spending
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            # Handle TotalCharges as object (convert to numeric)
            if df['TotalCharges'].dtype == 'object':
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            df['avg_monthly_spending'] = df.apply(
                lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
                axis=1
            )
            self.created_features.append('avg_monthly_spending')
        
        # Spending ratio compared to median
        median_charges = df['MonthlyCharges'].median()
        df['spending_ratio'] = df['MonthlyCharges'] / median_charges
        self.created_features.append('spending_ratio')
        
        # High spender (top 25%)
        high_spender_threshold = df['MonthlyCharges'].quantile(0.75)
        df['is_high_spender'] = (df['MonthlyCharges'] >= high_spender_threshold).astype(int)
        self.created_features.append('is_high_spender')
        
        logger.info("Created spending features")
        
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service usage features.
        
        Features:
        - Total number of services
        - Has internet service
        - Has phone service
        - Service diversity score
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with service features
        """
        df = df.copy()
        
        # Service columns
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Filter to existing columns
        existing_service_cols = [col for col in service_cols if col in df.columns]
        
        if existing_service_cols:
            # Total number of services (count 'Yes' values)
            df['total_services'] = df[existing_service_cols].apply(
                lambda row: sum(str(val).lower() in ['yes', 'dsl', 'fiber optic'] for val in row),
                axis=1
            )
            self.created_features.append('total_services')
            
            # Has no services
            df['has_no_services'] = (df['total_services'] == 0).astype(int)
            self.created_features.append('has_no_services')
        
        # Internet service indicator
        if 'InternetService' in df.columns:
            df['has_internet'] = (df['InternetService'] != 'No').astype(int)
            df['has_fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
            self.created_features.extend(['has_internet', 'has_fiber_optic'])
        
        # Phone service indicator
        if 'PhoneService' in df.columns:
            df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
            self.created_features.append('has_phone')
        
        logger.info("Created service features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Features:
        - Tenure × MonthlyCharges
        - SeniorCitizen × MonthlyCharges
        - Contract type × Tenure
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Tenure × MonthlyCharges
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['tenure_monthly_charges'] = df['tenure'] * df['MonthlyCharges']
            self.created_features.append('tenure_monthly_charges')
        
        # SeniorCitizen × MonthlyCharges
        if 'SeniorCitizen' in df.columns and 'MonthlyCharges' in df.columns:
            df['senior_monthly_charges'] = df['SeniorCitizen'] * df['MonthlyCharges']
            self.created_features.append('senior_monthly_charges')
        
        logger.info("Created interaction features")
        
        return df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'mutual_info',
    k: int = 20
) -> List[str]:
    """
    Select top k features using specified method.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Selection method ('f_classif', 'mutual_info', 'rfe')
        k: Number of features to select
    
    Returns:
        List of selected feature names
    
    Example:
        >>> selected_features = select_features(X_train, y_train, method='mutual_info', k=15)
    """
    logger.info(f"Selecting top {k} features using {method}...")
    
    # Ensure X is numerical
    X_numerical = X.select_dtypes(include=[np.number])
    
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=min(k, X_numerical.shape[1]))
        selector.fit(X_numerical, y)
        selected_features = X_numerical.columns[selector.get_support()].tolist()
    
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X_numerical.shape[1]))
        selector.fit(X_numerical, y)
        selected_features = X_numerical.columns[selector.get_support()].tolist()
    
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        selector = RFE(estimator, n_features_to_select=min(k, X_numerical.shape[1]))
        selector.fit(X_numerical, y)
        selected_features = X_numerical.columns[selector.get_support()].tolist()
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    logger.info(f"Selected {len(selected_features)} features: {selected_features}")
    
    return selected_features


def get_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Get feature importance using Random Forest.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance scores
    
    Example:
        >>> importance_df = get_feature_importance(X_train, y_train)
    """
    logger.info("Calculating feature importance...")
    
    # Ensure X is numerical
    X_numerical = X.select_dtypes(include=[np.number])
    
    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_numerical, y)
    
    # Get importance
    importance_df = pd.DataFrame({
        'feature': X_numerical.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    logger.info(f"Top {top_n} important features calculated")
    
    return importance_df
