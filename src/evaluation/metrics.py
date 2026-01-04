"""
Model Evaluation Metrics Module

Comprehensive evaluation metrics for classification models.

Author: Kartikeya
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Calculate all classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
    
    Returns:
        Dictionary containing all metrics
    
    Example:
        >>> metrics = calculate_all_metrics(y_test, y_pred, y_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    logger.info("Calculating evaluation metrics...")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    # ROC-AUC (requires probabilities)
    if y_proba is not None:
        # If y_proba is 2D, take probability of positive class
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # True Negatives, False Positives, False Negatives, True Positives
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    logger.info(f"Metrics calculated - Accuracy: {metrics['accuracy']:.4f}, "
               f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
               f"F1: {metrics['f1_score']:.4f}")
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Get detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes
    
    Returns:
        Classification report as string
    """
    if target_names is None:
        target_names = ['No Churn', 'Churn']
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    return report


def calculate_roc_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
    
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    # If y_proba is 2D, take probability of positive class
    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return fpr, tpr, thresholds


def calculate_precision_recall_curve_data(
    y_true: np.ndarray,
    y_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Precision-Recall curve data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
    
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    # If y_proba is 2D, take probability of positive class
    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return precision, recall, thresholds


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Args:
        results: Dictionary mapping model names to their metrics
    
    Returns:
        DataFrame with model comparison
    
    Example:
        >>> results = {
        ...     'Logistic Regression': metrics1,
        ...     'Random Forest': metrics2
        ... }
        >>> comparison_df = compare_models(results)
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'ROC-AUC': metrics.get('roc_auc', 0)
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('ROC-AUC', ascending=False)
    
    return df


def print_metrics_summary(metrics: Dict, model_name: str = "Model") -> None:
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - Performance Metrics".center(60))
    print(f"{'='*60}\n")
    
    print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    if 'roc_auc' in metrics:
        print(f"ROC-AUC:     {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    
    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        if 'true_positives' in metrics:
            print(f"\nTP: {metrics['true_positives']}, "
                  f"TN: {metrics['true_negatives']}, "
                  f"FP: {metrics['false_positives']}, "
                  f"FN: {metrics['false_negatives']}")
    
    print(f"\n{'='*60}\n")
