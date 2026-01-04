"""
Random Forest Model

Ensemble tree-based model for classification.

Author: Kartikeya
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.models.base_model import BaseModel
from src.config import ModelConfig, Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier with hyperparameter tuning.
    
    Powerful ensemble model that handles non-linear relationships well.
    """
    
    def __init__(self, tune_hyperparameters: bool = False):
        """
        Initialize Random Forest model.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        super().__init__("Random Forest")
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
        Train Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used)
            y_val: Validation labels (not used)
            **kwargs: Additional parameters
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_name}...")
        
        if self.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning (this may take a while)...")
            
            # Randomized search (faster than grid search)
            random_search = RandomizedSearchCV(
                RandomForestClassifier(),
                param_distributions=ModelConfig.RANDOM_FOREST_PARAMS,
                n_iter=20,  # Number of parameter settings sampled
                cv=ModelConfig.CV_FOLDS,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1,
                random_state=Config.RANDOM_STATE
            )
            
            random_search.fit(X_train, y_train)
            
            self.model = random_search.best_estimator_
            self.best_params = random_search.best_params_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")
            
            self.training_history = {
                'best_params': self.best_params,
                'best_cv_score': random_search.best_score_,
                'cv_results': random_search.cv_results_
            }
        
        else:
            # Train with good default parameters
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1,
                verbose=0
            )
            self.model.fit(X_train, y_train)
            
            logger.info("Model trained with default parameters")
        
        self.is_trained = True
        
        # Training score
        train_score = self.model.score(X_train, y_train)
        
        # Out-of-bag score (if available)
        oob_score = None
        if hasattr(self.model, 'oob_score_'):
            oob_score = self.model.oob_score_
            logger.info(f"Out-of-bag score: {oob_score:.4f}")
        
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'train_accuracy': train_score,
            'oob_score': oob_score,
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
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from Random Forest.
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return None
        
        importance = self.model.feature_importances_
        
        if self.feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(importance))],
                'importance': importance
            }).sort_values('importance', ascending=False)
        
        return importance_df
