"""
Model Evaluator for Churn Prediction

Calculates metrics and creates visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    """
    Calculate model performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    print(f"📊 Accuracy: {accuracy:.4f}")
    
    return metrics


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    return fig


def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained model (must have feature_importances_)
        feature_names: List of feature names
        
    Returns:
        Dictionary of feature importances or None
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names:
            importance_dict = dict(zip(feature_names, importances))
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True))
            return importance_dict
        else:
            return {'feature_' + str(i): imp for i, imp in enumerate(importances)}
    
    return None


def plot_feature_importance(importance_dict, top_n=10, title="Feature Importance"):
    """
    Plot feature importance.
    
    Args:
        importance_dict: Dictionary of feature importances
        top_n: Number of top features to show
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if not importance_dict:
        return None
    
    # Get top N features
    items = list(importance_dict.items())[:top_n]
    features, importances = zip(*items)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(range(len(features)), importances, color='steelblue')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig
