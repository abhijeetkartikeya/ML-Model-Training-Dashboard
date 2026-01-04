"""
Base Model Class

Abstract base class for all ML models ensuring consistent interface.

Author: Kartikeya
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.helpers import save_pickle, load_pickle, get_timestamp

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Ensures consistent interface across different model types.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model: Optional[Any] = None
        self.is_trained = False
        self.training_history: Dict = {}
        self.feature_names: Optional[list] = None
        
        logger.info(f"Initialized {model_name}")
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted labels
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict on
        
        Returns:
            Predicted probabilities for each class
        """
        pass
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            logger.warning("Saving untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names
        }
        
        save_pickle(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Self with loaded model
        """
        model_data = load_pickle(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data['training_history']
        self.feature_names = model_data.get('feature_names')
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_params(self) -> Dict:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            return {}
        
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return {}
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available).
        
        Returns:
            DataFrame with feature importance or None
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            if self.feature_names is not None:
                return pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
            else:
                return pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importance))],
                    'importance': importance
                }).sort_values('importance', ascending=False)
        
        logger.info(f"Feature importance not available for {self.model_name}")
        return None
    
    def __repr__(self) -> str:
        """String representation of the model."""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name} ({status})"
