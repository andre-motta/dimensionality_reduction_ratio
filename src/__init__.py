"""
Intrinsic Dimension Estimation Library

A Python library for estimating the intrinsic dimensionality of datasets using
correlation function analysis and computing the Dimensionality Reduction Ratio (DRR).

Authors: Andre Motta
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Andre Motta"

from .batch_processor import BatchProcessor
from .data_processor import DataProcessor
from .intrinsic_dimension import IntrinsicDimensionEstimator

__all__ = ["IntrinsicDimensionEstimator", "DataProcessor", "BatchProcessor"]
