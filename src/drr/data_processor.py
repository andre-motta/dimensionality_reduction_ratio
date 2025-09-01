"""
Data Processing Module

This module handles preprocessing of datasets for intrinsic dimension estimation,
including categorical encoding, goal variable removal, and data validation.
"""

import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles preprocessing of datasets for intrinsic dimension estimation.

    This processor:
    1. Identifies and removes goal variables (columns ending with +, -, !)
    2. Handles categorical data with label encoding
    3. Converts all data to numeric format
    4. Validates data quality and handles missing values
    5. Applies sampling for very large datasets

    Attributes:
        max_rows_for_processing (int): Maximum rows to process before sampling
        random_seed (int): Random seed for reproducible sampling
    """

    def __init__(self, max_rows_for_processing: int = 5000, random_seed: int = 42):
        """
        Initialize the data processor.

        Args:
            max_rows_for_processing: Maximum number of rows to process before sampling
            random_seed: Random seed for reproducible results
        """
        self.max_rows_for_processing = max_rows_for_processing
        self.random_seed = random_seed

        logger.info(f"Initialized DataProcessor with max_rows={max_rows_for_processing}, " f"seed={random_seed}")

    def process_dataset(self, data: Union[pd.DataFrame, np.ndarray, str]) -> Tuple[np.ndarray, dict]:
        """
        Process a dataset for intrinsic dimension estimation.

        Args:
            data: Input data - can be DataFrame, numpy array, or path to CSV file

        Returns:
            Tuple of (processed_data, metadata) where:
            - processed_data: Clean numeric numpy array ready for analysis
            - metadata: Dictionary with processing information

        Raises:
            ValueError: If data is invalid or cannot be processed
        """
        logger.info("Starting dataset processing")

        # Load data if path provided
        if isinstance(data, str):
            logger.info(f"Loading data from file: {data}")
            data = pd.read_csv(data)
        elif isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError(f"Unsupported data type: {type(data)}")

        original_shape = data.shape
        logger.info(f"Original dataset shape: {original_shape}")

        metadata = {
            "original_shape": original_shape,
            "original_columns": list(data.columns),
            "sampling_applied": False,
            "goal_variables_removed": [],
            "categorical_columns": [],
            "numeric_columns": [],
        }

        # Apply early sampling for very large datasets
        if data.shape[0] > self.max_rows_for_processing:
            logger.warning(f"Dataset is very large ({data.shape[0]} rows). " f"Sampling {self.max_rows_for_processing} rows")
            data = self._sample_data(data)
            metadata["sampling_applied"] = True
            metadata["sampled_shape"] = data.shape

        # Remove goal variables
        data, goal_vars = self._remove_goal_variables(data)
        metadata["goal_variables_removed"] = goal_vars

        if data.shape[1] == 0:
            raise ValueError("No feature columns remaining after removing goal variables")

        # Process categorical and numeric data
        data = self._process_columns(data, metadata)

        # Final validation and conversion
        processed_array = self._finalize_processing(data, metadata)

        logger.info(f"Processing complete. Final shape: {processed_array.shape}")

        return processed_array, metadata

    def _sample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sample data for large datasets."""
        np.random.seed(self.random_seed)
        sample_indices = np.random.choice(data.shape[0], self.max_rows_for_processing, replace=False)
        sampled_data = data.iloc[sample_indices].reset_index(drop=True)
        logger.info(f"Sampled dataset shape: {sampled_data.shape}")
        return sampled_data

    def _remove_goal_variables(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove goal variables (columns ending with +, -, or !).

        Args:
            data: Input DataFrame

        Returns:
            Tuple of (data_without_goals, removed_columns)
        """
        goal_columns = []
        feature_columns = []

        for column in data.columns:
            if str(column).endswith(("+", "-", "!")):
                goal_columns.append(column)
            else:
                feature_columns.append(column)

        if goal_columns:
            logger.info(f"Removing {len(goal_columns)} goal variables: {goal_columns}")
            data_processed = data[feature_columns]
        else:
            logger.info("No goal variables found (columns ending with +, -, or !)")
            data_processed = data.copy()

        logger.info(f"Using {len(feature_columns)} feature columns for analysis")

        return data_processed, goal_columns

    def _process_columns(self, data: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Process each column to ensure numeric format.

        Args:
            data: DataFrame to process
            metadata: Metadata dictionary to update

        Returns:
            Processed DataFrame with all numeric columns
        """
        logger.info("Processing columns for numeric conversion")
        data_processed = data.copy()

        for column in data_processed.columns:
            col_data = data_processed[column]

            # Skip if already numeric
            if pd.api.types.is_numeric_dtype(col_data):
                logger.debug(f"Column '{column}': already numeric")
                metadata["numeric_columns"].append(column)
                continue

            logger.debug(f"Column '{column}': {col_data.dtype} -> converting...")

            # Try numeric conversion first
            if self._try_numeric_conversion(data_processed, column, col_data):
                metadata["numeric_columns"].append(column)
                continue

            # Handle as categorical
            if self._handle_categorical_column(data_processed, column, col_data):
                metadata["categorical_columns"].append(column)
                logger.debug(f"Column '{column}': encoded as categorical")
            else:
                # Last resort: force conversion
                logger.warning(f"Column '{column}': forcing conversion to numeric")
                data_processed[column] = pd.to_numeric(col_data, errors="coerce").fillna(0)
                metadata["numeric_columns"].append(column)

        logger.info(
            f"Processed {len(metadata['numeric_columns'])} numeric and "
            f"{len(metadata['categorical_columns'])} categorical columns"
        )

        return data_processed

    def _try_numeric_conversion(self, data: pd.DataFrame, column: str, col_data: pd.Series) -> bool:
        """
        Try to convert column to numeric.

        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            numeric_data = pd.to_numeric(col_data, errors="coerce")
            if not numeric_data.isna().all():
                # Fill NaN values with median for partially numeric columns
                data[column] = numeric_data.fillna(numeric_data.median())
                logger.debug(f"Column '{column}': converted to numeric " f"(filled {numeric_data.isna().sum()} NaN values)")
                return True
        except Exception as e:
            logger.debug(f"Column '{column}': numeric conversion failed: {e}")

        return False

    def _handle_categorical_column(self, data: pd.DataFrame, column: str, col_data: pd.Series) -> bool:
        """
        Handle categorical column with label encoding.

        Returns:
            True if categorical handling was successful, False otherwise
        """
        try:
            if col_data.dtype == "object" or str(col_data.dtype) == "category":
                unique_values = col_data.unique()
                logger.debug(f"Column '{column}': categorical with {len(unique_values)} unique values")

                # Create label encoding
                value_to_num = {val: idx for idx, val in enumerate(unique_values)}
                data[column] = col_data.map(value_to_num).fillna(0)
                logger.debug(f"Column '{column}': encoded as integers 0-{len(unique_values)-1}")
                return True
            else:
                # Try string conversion then encoding
                str_data = col_data.astype(str)
                unique_values = str_data.unique()
                value_to_num = {val: idx for idx, val in enumerate(unique_values)}
                data[column] = str_data.map(value_to_num)
                logger.debug(f"Column '{column}': string-encoded ({len(unique_values)} unique values)")
                return True
        except Exception as e:
            logger.debug(f"Column '{column}': categorical handling failed: {e}")

        return False

    def _finalize_processing(self, data: pd.DataFrame, metadata: dict) -> np.ndarray:
        """
        Finalize data processing and convert to numpy array.

        Args:
            data: Processed DataFrame
            metadata: Metadata dictionary to update

        Returns:
            Final numpy array ready for analysis
        """
        # Ensure all columns are numeric
        for column in data.columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                logger.debug(f"Column '{column}': forcing final conversion to float64")
                data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)

        # Convert to float64 for consistency
        data = data.astype(np.float64)

        # Final data quality checks
        nan_count = data.isna().sum().sum()
        inf_count = np.isinf(data.values).sum()

        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values, filling with 0")
            data = data.fillna(0)

        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values, replacing with finite values")
            data = data.replace([np.inf, -np.inf], 0)

        metadata["final_shape"] = data.shape
        metadata["data_quality"] = {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "min_value": float(data.min().min()),
            "max_value": float(data.max().max()),
        }

        logger.info(
            f"Final data quality check: shape={data.shape}, "
            f"range=[{metadata['data_quality']['min_value']:.6f}, "
            f"{metadata['data_quality']['max_value']:.6f}]"
        )

        return data.values

    def validate_processed_data(self, data: np.ndarray) -> bool:
        """
        Validate that processed data is suitable for intrinsic dimension estimation.

        Args:
            data: Processed data array

        Returns:
            True if data is valid, False otherwise
        """
        if data.ndim != 2:
            logger.error(f"Data must be 2D, got {data.ndim}D")
            return False

        if data.shape[0] < 2:
            logger.error(f"Need at least 2 samples, got {data.shape[0]}")
            return False

        if data.shape[1] < 1:
            logger.error(f"Need at least 1 feature, got {data.shape[1]}")
            return False

        if not np.all(np.isfinite(data)):
            logger.error("Data contains non-finite values")
            return False

        logger.info("Data validation passed")
        return True
