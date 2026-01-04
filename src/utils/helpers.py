"""
Helper Functions Module

Utility functions used across the application.

Author: Kartikeya
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    
    Example:
        >>> save_pickle(model, 'models/my_model.pkl')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle to {filepath}: {e}")
        raise


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded object
    
    Example:
        >>> model = load_pickle('models/my_model.pkl')
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from {filepath}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle from {filepath}: {e}")
        raise


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded dictionary
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        raise


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    
    Example:
        >>> timestamp = get_timestamp()
        >>> print(timestamp)  # '20240103_143022'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_model_filename(model_name: str, extension: str = 'pkl') -> str:
    """
    Create a timestamped filename for model saving.
    
    Args:
        model_name: Name of the model
        extension: File extension (default: 'pkl')
    
    Returns:
        Filename with timestamp
    
    Example:
        >>> filename = create_model_filename('random_forest')
        >>> print(filename)  # 'random_forest_20240103_143022.pkl'
    """
    timestamp = get_timestamp()
    return f"{model_name}_{timestamp}.{extension}"


def convert_to_serializable(obj: Any) -> Any:
    """
    Convert numpy/pandas objects to JSON-serializable types.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    else:
        return obj


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage string.
    
    Args:
        value: Decimal value (e.g., 0.856)
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string
    
    Example:
        >>> format_percentage(0.856)
        '85.60%'
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_class_weights(y: Union[pd.Series, np.ndarray]) -> Dict[int, float]:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target variable
    
    Returns:
        Dictionary mapping class labels to weights
    
    Example:
        >>> weights = calculate_class_weights(y_train)
        >>> print(weights)  # {0: 0.52, 1: 1.48}
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    if isinstance(y, pd.Series):
        y = y.values
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def print_section_header(title: str, width: int = 80) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Total width of header
    
    Example:
        >>> print_section_header("Data Loading")
        ================================================================================
                                    Data Loading
        ================================================================================
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width + "\n")
