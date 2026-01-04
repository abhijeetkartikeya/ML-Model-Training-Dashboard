"""
Logistic Regression Model

Baseline model for binary classification.

Author: Kartikeya
"""

import numpy as np
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.models.base_model import BaseModel
from src.config import ModelConfig, Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogisticModel(BaseModel):
    """
    Logistic Regression classifier with hyperparameter tuning.
    
    Good baseline model that's fast to train and interpretable.
    """
    
    def __init__(self, tune_hyperparameters: bool = False):
        """
        Initialize Logistic Regression model.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        super().__init__("Logistic Regression")
        self.tune_hyperparameters = tune_hyperparameters
        self.best_params: Optional[Dict] = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for logistic regression)
            y_val: Validation labels (not used for logistic regression)
            **kwargs: Additional parameters
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_name}...")
        
        if self.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            
            # Grid search
            grid_search = GridSearchCV(
                LogisticRegression(),
                param_grid=ModelConfig.LOGISTIC_PARAMS,
                cv=ModelConfig.CV_FOLDS,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
            self.training_history = {
                'best_params': self.best_params,
                'best_cv_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
        
        else:
            # Train with default parameters
            self.model = LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000
            )
            self.model.fit(X_train, y_train)
            
            logger.info("Model trained with default parameters")
        
        self.is_trained = True
        
        # Training score
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'train_accuracy': train_score,
            'best_params': self.best_params
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get model coefficients.
        
        Returns:
            Coefficient array or None
        """
        if not self.is_trained:
            return None
        
        return self.model.coef_[0]
