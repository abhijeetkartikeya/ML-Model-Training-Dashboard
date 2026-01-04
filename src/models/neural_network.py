"""
Neural Network Model

Deep learning model using Keras/TensorFlow.

Author: Kartikeya
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from src.models.base_model import BaseModel
from src.config import ModelConfig, Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


class NeuralNetworkModel(BaseModel):
    """
    Neural Network classifier using Keras/TensorFlow.
    
    Deep learning model with multiple hidden layers, dropout, and batch normalization.
    """
    
    def __init__(self):
        """Initialize Neural Network model."""
        super().__init__("Neural Network")
        self.history: Optional[keras.callbacks.History] = None
        self.input_dim: Optional[int] = None
    
    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network architecture.
        
        Args:
            input_dim: Number of input features
        
        Returns:
            Compiled Keras model
        """
        params = ModelConfig.NEURAL_NETWORK_PARAMS
        hidden_layers = params['hidden_layers']
        dropout_rates = params['dropout_rates']
        learning_rate = params['learning_rate']
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_dim,)),
            
            # First hidden layer
            layers.Dense(hidden_layers[0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rates[0]),
            
            # Second hidden layer
            layers.Dense(hidden_layers[1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rates[1]),
            
            # Third hidden layer
            layers.Dense(hidden_layers[2], activation='relu'),
            layers.Dropout(dropout_rates[2]),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        logger.info("Neural network architecture:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Train Neural Network model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            **kwargs: Additional parameters (epochs, batch_size, etc.)
        
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_name}...")
        
        # Get parameters
        params = ModelConfig.NEURAL_NETWORK_PARAMS
        epochs = kwargs.get('epochs', params['epochs'])
        batch_size = kwargs.get('batch_size', params['batch_size'])
        
        # Build model
        self.input_dim = X_train.shape[1]
        self.model = self._build_model(self.input_dim)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            # Use validation split from training data
            validation_split = params['validation_split']
            logger.info(f"Using {validation_split*100}% of training data for validation")
        
        # Callbacks
        callback_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=params['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=params['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info(f"Training for up to {epochs} epochs with batch size {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=params['validation_split'] if validation_data is None else 0,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.is_trained = True
        
        # Store training history
        self.training_history = {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'auc': self.history.history['auc'],
            'val_loss': self.history.history.get('val_loss', []),
            'val_accuracy': self.history.history.get('val_accuracy', []),
            'val_auc': self.history.history.get('val_auc', []),
            'epochs_trained': len(self.history.history['loss'])
        }
        
        # Final metrics
        final_loss = self.history.history['loss'][-1]
        final_accuracy = self.history.history['accuracy'][-1]
        final_auc = self.history.history['auc'][-1]
        
        logger.info(f"Training completed after {len(self.history.history['loss'])} epochs")
        logger.info(f"Final training - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}, AUC: {final_auc:.4f}")
        
        if 'val_loss' in self.history.history:
            val_loss = self.history.history['val_loss'][-1]
            val_accuracy = self.history.history['val_accuracy'][-1]
            val_auc = self.history.history['val_auc'][-1]
            logger.info(f"Final validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        
        return {
            'train_loss': final_loss,
            'train_accuracy': final_accuracy,
            'train_auc': final_auc,
            'epochs_trained': len(self.history.history['loss']),
            'history': self.training_history
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
        
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Predicted probabilities for both classes
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get probability for class 1
        prob_class_1 = self.model.predict(X, verbose=0).flatten()
        
        # Create probability matrix for both classes
        prob_class_0 = 1 - prob_class_1
        probabilities = np.column_stack([prob_class_0, prob_class_1])
        
        return probabilities
    
    def save_model(self, filepath: Path) -> None:
        """
        Save Keras model to file.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            logger.warning("Saving untrained model")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = filepath.with_suffix('.h5')
        self.model.save(model_path)
        
        # Save metadata
        import json
        metadata = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'input_dim': self.input_dim,
            'feature_names': self.feature_names
        }
        
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, filepath: Path) -> 'NeuralNetworkModel':
        """
        Load Keras model from file.
        
        Args:
            filepath: Path to model file
        
        Returns:
            Self with loaded model
        """
        filepath = Path(filepath)
        
        # Load Keras model
        model_path = filepath.with_suffix('.h5')
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        import json
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.model_name = metadata['model_name']
        self.is_trained = metadata['is_trained']
        self.training_history = metadata['training_history']
        self.input_dim = metadata['input_dim']
        self.feature_names = metadata.get('feature_names')
        
        logger.info(f"Model loaded from {model_path}")
        return self
