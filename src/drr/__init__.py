"""
DRR - Dimensionality Reduction Ratio Analysis

A professional Python toolkit for estimating the intrinsic dimensionality of datasets
and computing Dimensionality Reduction Ratio (DRR) metrics.

This package implements the Levina-Bickel correlation function method for intrinsic
dimension estimation with enhancements for large-scale dataset processing.

Classes:
    IntrinsicDimensionEstimator: Core algorithm for intrinsic dimension estimation
    DataProcessor: Data preprocessing and validation
    BatchProcessor: Batch processing of multiple datasets

Functions:
    estimate_intrinsic_dimension: Convenience function for single dataset analysis
"""

__version__ = "1.0.0"
__author__ = "Andre Lustosa"
__email__ = "dexmotta6@gmail.com"

from .batch_processor import BatchProcessor
from .data_processor import DataProcessor

# Import main classes for easy access
from .intrinsic_dimension_estimator import IntrinsicDimensionEstimator


# Convenience function for easy usage
def estimate_intrinsic_dimension(data, max_samples=2000, distance_metric="l1"):
    """
    Convenience function to estimate intrinsic dimension of a dataset.

    Args:
        data: NumPy array or pandas DataFrame with numerical data
        max_samples: Maximum samples for large datasets (default: 2000)
        distance_metric: Distance metric to use (default: 'l1')

    Returns:
        tuple: (original_dimensions, intrinsic_dimension, drr)
    """
    import numpy as np
    import pandas as pd

    # Convert to numpy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        data = np.array(data)

    # Initialize estimator and compute
    estimator = IntrinsicDimensionEstimator(max_samples=max_samples, distance_metric=distance_metric)

    return estimator.estimate(data)


# Package metadata
__all__ = [
    "IntrinsicDimensionEstimator",
    "DataProcessor",
    "BatchProcessor",
    "estimate_intrinsic_dimension",
    "__version__",
]
