"""
Data Preprocessing Module

Handles data cleaning, encoding, scaling, and preparation for ML models.

Author: Kartikeya
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from src.config import Config
from src.utils.logger import get_logger
from src.utils.helpers import save_pickle, load_pickle

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline.
    
    Handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Class imbalance
    - Train-test splitting
    """
    
    def __init__(
        self,
        imputation_strategy: str = 'mean',
        scaling_method: str = 'standard',
        encoding_method: str = 'label',
        handle_imbalance: bool = False,
        imbalance_method: str = 'smote'
    ):
        """
        Initialize DataPreprocessor.
        
        Args:
            imputation_strategy: Strategy for missing values ('mean', 'median', 'mode', 'knn')
            scaling_method: Scaling method ('standard', 'minmax', 'robust', 'none')
            encoding_method: Encoding for categorical features ('label', 'onehot')
            handle_imbalance: Whether to handle class imbalance
            imbalance_method: Method for handling imbalance ('smote', 'oversample', 'undersample')
        """
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.handle_imbalance = handle_imbalance
        self.imbalance_method = imbalance_method
        
        # Preprocessing objects
        self.numerical_imputer: Optional[Union[SimpleImputer, KNNImputer]] = None
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler, RobustScaler]] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoder: Optional[OneHotEncoder] = None
        self.feature_names: Optional[List[str]] = None
        
        logger.info(f"DataPreprocessor initialized with: imputation={imputation_strategy}, "
                   f"scaling={scaling_method}, encoding={encoding_method}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'DataPreprocessor':
        """
        Fit preprocessing pipeline on training data.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional, needed for SMOTE)
        
        Returns:
            Self for method chaining
        """
        logger.info("Fitting preprocessing pipeline...")
        
        # Separate numerical and categorical features
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numerical features: {len(numerical_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        
        # Fit numerical imputer
        if numerical_features:
            self._fit_numerical_imputer(X[numerical_features])
        
        # Fit categorical imputer
        if categorical_features:
            self._fit_categorical_imputer(X[categorical_features])
        
        # Fit encoders
        if categorical_features:
            self._fit_encoders(X[categorical_features])
        
        # Transform for fitting scaler
        X_transformed = self._transform_features(X)
        
        # Fit scaler
        if self.scaling_method != 'none':
            self._fit_scaler(X_transformed)
        
        # Store feature names
        if isinstance(X_transformed, pd.DataFrame):
            self.feature_names = X_transformed.columns.tolist()
        
        logger.info("Preprocessing pipeline fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessing pipeline.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Transformed numpy array
        """
        logger.info("Transforming data...")
        
        # Transform features
        X_transformed = self._transform_features(X)
        
        # Scale features
        if self.scaler is not None:
            X_transformed = self.scaler.transform(X_transformed)
        
        logger.info(f"Data transformed: shape {X_transformed.shape}")
        return X_transformed
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Fit and transform data in one step.
        
        Args:
            X: Feature DataFrame
            y: Target Series (optional)
        
        Returns:
            Transformed numpy array
        """
        return self.fit(X, y).transform(X)
    
    def _fit_numerical_imputer(self, X_numerical: pd.DataFrame) -> None:
        """Fit imputer for numerical features."""
        if self.imputation_strategy == 'knn':
            self.numerical_imputer = KNNImputer(n_neighbors=5)
        else:
            strategy = self.imputation_strategy if self.imputation_strategy in ['mean', 'median'] else 'mean'
            self.numerical_imputer = SimpleImputer(strategy=strategy)
        
        self.numerical_imputer.fit(X_numerical)
        logger.info(f"Numerical imputer fitted with strategy: {self.imputation_strategy}")
    
    def _fit_categorical_imputer(self, X_categorical: pd.DataFrame) -> None:
        """Fit imputer for categorical features."""
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.categorical_imputer.fit(X_categorical)
        logger.info("Categorical imputer fitted with strategy: most_frequent")
    
    def _fit_encoders(self, X_categorical: pd.DataFrame) -> None:
        """Fit encoders for categorical features."""
        if self.encoding_method == 'label':
            for col in X_categorical.columns:
                le = LabelEncoder()
                # Handle missing values before encoding
                non_null_values = X_categorical[col].dropna()
                le.fit(non_null_values)
                self.label_encoders[col] = le
            logger.info(f"Label encoders fitted for {len(self.label_encoders)} features")
        
        elif self.encoding_method == 'onehot':
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            # Impute first for one-hot encoding
            if self.categorical_imputer is not None:
                X_imputed = self.categorical_imputer.transform(X_categorical)
                X_imputed = pd.DataFrame(X_imputed, columns=X_categorical.columns)
            else:
                X_imputed = X_categorical
            self.onehot_encoder.fit(X_imputed)
            logger.info("One-hot encoder fitted")
    
    def _fit_scaler(self, X: Union[pd.DataFrame, np.ndarray]) -> None:
        """Fit scaler on features."""
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            return
        
        self.scaler.fit(X)
        logger.info(f"Scaler fitted with method: {self.scaling_method}")
    
    def _transform_features(self, X: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray]:
        """Transform features (imputation + encoding)."""
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        transformed_parts = []
        
        # Transform numerical features
        if numerical_features and self.numerical_imputer is not None:
            X_numerical = self.numerical_imputer.transform(X[numerical_features])
            transformed_parts.append(pd.DataFrame(
                X_numerical,
                columns=numerical_features,
                index=X.index
            ))
        
        # Transform categorical features
        if categorical_features:
            if self.categorical_imputer is not None:
                X_categorical = self.categorical_imputer.transform(X[categorical_features])
                X_categorical = pd.DataFrame(
                    X_categorical,
                    columns=categorical_features,
                    index=X.index
                )
            else:
                X_categorical = X[categorical_features].copy()
            
            # Encode
            if self.encoding_method == 'label':
                for col in categorical_features:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X_categorical[col] = X_categorical[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                transformed_parts.append(X_categorical)
            
            elif self.encoding_method == 'onehot' and self.onehot_encoder is not None:
                X_encoded = self.onehot_encoder.transform(X_categorical)
                feature_names = self.onehot_encoder.get_feature_names_out(categorical_features)
                transformed_parts.append(pd.DataFrame(
                    X_encoded,
                    columns=feature_names,
                    index=X.index
                ))
        
        # Combine all parts
        if len(transformed_parts) > 0:
            X_transformed = pd.concat(transformed_parts, axis=1)
        else:
            X_transformed = X.copy()
        
        return X_transformed
    
    def handle_class_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance in the dataset.
        
        Args:
            X: Feature array
            y: Target array
        
        Returns:
            Resampled X and y
        """
        if not self.handle_imbalance:
            return X, y
        
        logger.info(f"Handling class imbalance with method: {self.imbalance_method}")
        
        # Count classes before
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"Class distribution before: {dict(zip(unique, counts))}")
        
        # Apply resampling
        if self.imbalance_method == 'smote':
            resampler = SMOTE(random_state=Config.RANDOM_STATE)
        elif self.imbalance_method == 'oversample':
            resampler = RandomOverSampler(random_state=Config.RANDOM_STATE)
        elif self.imbalance_method == 'undersample':
            resampler = RandomUnderSampler(random_state=Config.RANDOM_STATE)
        else:
            logger.warning(f"Unknown imbalance method: {self.imbalance_method}")
            return X, y
        
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        
        # Count classes after
        unique, counts = np.unique(y_resampled, return_counts=True)
        logger.info(f"Class distribution after: {dict(zip(unique, counts))}")
        
        return X_resampled, y_resampled
    
    def save_pipeline(self, filepath: Union[str, Path]) -> None:
        """
        Save preprocessing pipeline to file.
        
        Args:
            filepath: Path to save pipeline
        """
        pipeline_data = {
            'imputation_strategy': self.imputation_strategy,
            'scaling_method': self.scaling_method,
            'encoding_method': self.encoding_method,
            'handle_imbalance': self.handle_imbalance,
            'imbalance_method': self.imbalance_method,
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'onehot_encoder': self.onehot_encoder,
            'feature_names': self.feature_names
        }
        save_pickle(pipeline_data, filepath)
        logger.info(f"Preprocessing pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: Union[str, Path]) -> 'DataPreprocessor':
        """
        Load preprocessing pipeline from file.
        
        Args:
            filepath: Path to pipeline file
        
        Returns:
            Self with loaded pipeline
        """
        pipeline_data = load_pickle(filepath)
        
        self.imputation_strategy = pipeline_data['imputation_strategy']
        self.scaling_method = pipeline_data['scaling_method']
        self.encoding_method = pipeline_data['encoding_method']
        self.handle_imbalance = pipeline_data['handle_imbalance']
        self.imbalance_method = pipeline_data['imbalance_method']
        self.numerical_imputer = pipeline_data['numerical_imputer']
        self.categorical_imputer = pipeline_data['categorical_imputer']
        self.scaler = pipeline_data['scaler']
        self.label_encoders = pipeline_data['label_encoders']
        self.onehot_encoder = pipeline_data['onehot_encoder']
        self.feature_names = pipeline_data['feature_names']
        
        logger.info(f"Preprocessing pipeline loaded from {filepath}")
        return self


def prepare_data(
    df: pd.DataFrame,
    target_column: str = Config.TARGET_COLUMN,
    test_size: float = Config.TEST_SIZE,
    random_state: int = Config.RANDOM_STATE,
    drop_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for modeling by splitting into train and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of test set
        random_state: Random seed
        drop_columns: Columns to drop (e.g., ID columns)
    
    Returns:
        X_train, X_test, y_train, y_test
    
    Example:
        >>> X_train, X_test, y_train, y_test = prepare_data(df)
    """
    logger.info("Preparing data for modeling...")
    
    # Drop specified columns
    if drop_columns is None:
        drop_columns = [Config.ID_COLUMN] if Config.ID_COLUMN in df.columns else []
    
    df = df.drop(columns=drop_columns, errors='ignore')
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target if it's categorical
    if y.dtype == 'object':
        logger.info("Encoding target variable...")
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=target_column)
        logger.info(f"Target classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
    logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test
