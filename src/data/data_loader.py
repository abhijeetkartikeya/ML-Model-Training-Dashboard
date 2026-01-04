"""
Data Loading Module

Handles loading, validation, and initial inspection of datasets.

Author: Kartikeya
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from src.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loading and validation class.
    
    Supports CSV, Excel, and JSON formats with comprehensive validation.
    """
    
    def __init__(self):
        """Initialize DataLoader."""
        self.data: Optional[pd.DataFrame] = None
        self.filepath: Optional[Path] = None
    
    def load_data(
        self,
        filepath: Union[str, Path],
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load data from file with validation.
        
        Args:
            filepath: Path to data file
            validate: Whether to validate the data
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is unsupported or data is invalid
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_data('data/churn.csv')
        """
        filepath = Path(filepath)
        self.filepath = filepath
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        # Load based on file extension
        try:
            if filepath.suffix == '.csv':
                self.data = pd.read_csv(filepath)
            elif filepath.suffix in ['.xlsx', '.xls']:
                self.data = pd.read_excel(filepath)
            elif filepath.suffix == '.json':
                self.data = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            logger.info(f"Data loaded successfully: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
            
            if validate:
                self._validate_data()
            
            return self.data
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self) -> None:
        """
        Validate loaded data.
        
        Checks for:
        - Empty DataFrame
        - Required columns
        - Data types
        """
        if self.data is None or self.data.empty:
            raise ValueError("DataFrame is empty")
        
        logger.info("Validating data...")
        
        # Check for completely empty columns
        empty_cols = self.data.columns[self.data.isnull().all()].tolist()
        if empty_cols:
            logger.warning(f"Completely empty columns found: {empty_cols}")
        
        # Check for duplicate rows
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate rows")
        
        logger.info("Data validation complete")
    
    def validate_churn_schema(self) -> bool:
        """
        Validate that data matches expected churn prediction schema.
        
        Returns:
            True if schema is valid
        
        Raises:
            ValueError: If required columns are missing
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        required_columns = (
            Config.CATEGORICAL_FEATURES +
            Config.NUMERICAL_FEATURES +
            [Config.TARGET_COLUMN]
        )
        
        missing_columns = set(required_columns) - set(self.data.columns)
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info("Schema validation passed")
        return True
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive summary of the dataset.
        
        Returns:
            Dictionary containing data summary statistics
        
        Example:
            >>> summary = loader.get_data_summary()
            >>> print(summary['num_rows'])
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        summary = {
            'num_rows': len(self.data),
            'num_columns': len(self.data.columns),
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.astype(str).to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'duplicates': self.data.duplicated().sum(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Numerical columns summary
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary['numerical_summary'] = self.data[numerical_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['categorical_summary'] = {
                col: self.data[col].value_counts().to_dict()
                for col in categorical_cols
            }
        
        return summary
    
    def remove_duplicates(self, inplace: bool = True) -> pd.DataFrame:
        """
        Remove duplicate rows from data.
        
        Args:
            inplace: Whether to modify data in place
        
        Returns:
            DataFrame with duplicates removed
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        initial_rows = len(self.data)
        
        if inplace:
            self.data = self.data.drop_duplicates()
            final_rows = len(self.data)
        else:
            data = self.data.drop_duplicates()
            final_rows = len(data)
        
        removed = initial_rows - final_rows
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return self.data if inplace else data
    
    def get_column_info(self, column: str) -> Dict:
        """
        Get detailed information about a specific column.
        
        Args:
            column: Column name
        
        Returns:
            Dictionary with column information
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        
        col_data = self.data[column]
        
        info = {
            'name': column,
            'dtype': str(col_data.dtype),
            'count': col_data.count(),
            'missing': col_data.isnull().sum(),
            'missing_percentage': col_data.isnull().sum() / len(col_data) * 100,
            'unique_values': col_data.nunique()
        }
        
        # Add statistics based on data type
        if pd.api.types.is_numeric_dtype(col_data):
            info.update({
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75)
            })
        else:
            info['top_values'] = col_data.value_counts().head(10).to_dict()
        
        return info


def load_sample_data() -> pd.DataFrame:
    """
    Load the sample churn dataset.
    
    Returns:
        Sample churn DataFrame
    
    Example:
        >>> df = load_sample_data()
    """
    loader = DataLoader()
    return loader.load_data(Config.SAMPLE_DATA_PATH, validate=False)
