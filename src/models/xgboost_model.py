"""
XGBoost Model

Gradient boosting model for classification.

Author: Kartikeya
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.models.base_model import BaseModel
from src.config import ModelConfig, Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier with hyperparameter tuning.
    
    State-of-the-art gradient boosting algorithm.
    """
    
    def __init__(self, tune_hyperparameters: bool = False):
        """
        Initialize XGBoost model.
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        super().__init__("XGBoost")
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
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            **kwargs: Additional parameters
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_name}...")
        
        # Prepare evaluation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        if self.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning (this may take a while)...")
            
            # Randomized search
            random_search = RandomizedSearchCV(
                XGBClassifier(
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=Config.RANDOM_STATE
                ),
                param_distributions=ModelConfig.XGBOOST_PARAMS,
                n_iter=20,
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
            self.model = XGBClassifier(
                learning_rate=0.1,
                max_depth=5,
                n_estimators=200,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=Config.RANDOM_STATE,
                n_jobs=-1
            )
            
            # Fit with early stopping if validation set provided
            if eval_set is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=10,
                    verbose=False
                )
                logger.info(f"Training stopped at iteration: {self.model.best_iteration}")
            else:
                self.model.fit(X_train, y_train)
            
            logger.info("Model trained with default parameters")
        
        self.is_trained = True
        
        # Training score
        train_score = self.model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        return {
            'train_accuracy': train_score,
            'best_params': self.best_params,
            'best_iteration': getattr(self.model, 'best_iteration', None)
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
        Get feature importance from XGBoost.
        
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
