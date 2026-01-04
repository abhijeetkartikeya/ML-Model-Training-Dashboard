"""
Configuration Management Module

Centralized configuration for the Customer Churn Prediction system.
No hard-coded paths - all paths are dynamically resolved.

Author: Kartikeya
"""

from pathlib import Path
from typing import Dict, List
import os


class Config:
    """Main configuration class with dynamic path resolution."""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    MODELS_DIR = BASE_DIR / "models"
    SAVED_MODELS_DIR = MODELS_DIR / "saved_models"
    SCALERS_DIR = MODELS_DIR / "scalers"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Sample data
    SAMPLE_DATA_PATH = DATA_DIR / "sample_churn_data.csv"
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls) -> None:
        """Create all necessary directories."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.SAVED_MODELS_DIR,
            cls.SCALERS_DIR,
            cls.LOGS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # Feature definitions
    CATEGORICAL_FEATURES = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    NUMERICAL_FEATURES = [
        'tenure', 'MonthlyCharges', 'TotalCharges'
    ]
    
    TARGET_COLUMN = 'Churn'
    ID_COLUMN = 'customerID'
    
    # Data preprocessing
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    VALIDATION_SPLIT = 0.2


class ModelConfig:
    """Configuration for ML models and hyperparameters."""
    
    # Logistic Regression
    LOGISTIC_PARAMS = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
        'max_iter': [1000],
        'random_state': [Config.RANDOM_STATE]
    }
    
    # Random Forest
    RANDOM_FOREST_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [Config.RANDOM_STATE],
        'n_jobs': [-1]
    }
    
    # XGBoost
    XGBOOST_PARAMS = {
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'random_state': [Config.RANDOM_STATE]
    }
    
    # Neural Network
    NEURAL_NETWORK_PARAMS = {
        'hidden_layers': [128, 64, 32],
        'dropout_rates': [0.3, 0.2, 0.2],
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10,
        'reduce_lr_patience': 5,
        'validation_split': Config.VALIDATION_SPLIT
    }
    
    # Cross-validation
    CV_FOLDS = 5
    
    # Model names
    MODEL_NAMES = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'neural_network': 'Neural Network'
    }


class LoggingConfig:
    """Configuration for logging."""
    
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_LEVEL = 'INFO'
    LOG_FILE = Config.LOGS_DIR / 'churn_prediction.log'
    
    # Create logs directory
    Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)


class StreamlitConfig:
    """Configuration for Streamlit dashboard."""
    
    PAGE_TITLE = "Customer Churn Prediction System"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    INITIAL_SIDEBAR_STATE = "expanded"
    
    # Theme colors
    PRIMARY_COLOR = "#FF4B4B"
    BACKGROUND_COLOR = "#0E1117"
    SECONDARY_BACKGROUND_COLOR = "#262730"
    TEXT_COLOR = "#FAFAFA"
    
    # Chart settings
    CHART_HEIGHT = 400
    CHART_TEMPLATE = "plotly_dark"


# Initialize directories on import
Config.create_directories()
