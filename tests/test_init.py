"""Tests for __init__ module."""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import drr
from drr import (
    IntrinsicDimensionEstimator,
    DataProcessor,
    BatchProcessor,
    estimate_intrinsic_dimension,
    __version__
)


def test_package_metadata():
    """Test package metadata is properly defined."""
    assert hasattr(drr, '__version__')
    assert hasattr(drr, '__author__')
    assert hasattr(drr, '__email__')
    
    assert isinstance(drr.__version__, str)
    assert len(drr.__version__) > 0
    
    assert isinstance(drr.__author__, str)
    assert len(drr.__author__) > 0
    
    assert isinstance(drr.__email__, str)
    assert '@' in drr.__email__


def test_version_import():
    """Test version can be imported directly."""
    assert __version__ == drr.__version__
    assert __version__ == "1.0.0"


def test_class_imports():
    """Test that main classes can be imported from package."""
    # Test classes are available
    assert IntrinsicDimensionEstimator is not None
    assert DataProcessor is not None
    assert BatchProcessor is not None
    
    # Test they can be instantiated
    estimator = IntrinsicDimensionEstimator()
    processor = DataProcessor()
    batch = BatchProcessor()
    
    assert estimator is not None
    assert processor is not None
    assert batch is not None


def test_convenience_function_exists():
    """Test convenience function is available."""
    assert estimate_intrinsic_dimension is not None
    assert callable(estimate_intrinsic_dimension)


def test_convenience_function_with_numpy_array():
    """Test convenience function with numpy array input."""
    # Create simple test data
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    
    result = estimate_intrinsic_dimension(data)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    original_dims, intrinsic_dim, drr = result
    assert isinstance(original_dims, int)
    assert isinstance(intrinsic_dim, int)
    assert isinstance(drr, float)
    
    assert original_dims == 3
    assert intrinsic_dim >= 1
    assert 0 <= drr <= 1


def test_convenience_function_with_pandas_dataframe():
    """Test convenience function with pandas DataFrame input."""
    # Create test DataFrame
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 3, 4, 5, 6],
        'feature3': [3, 4, 5, 6, 7]
    })
    
    result = estimate_intrinsic_dimension(df)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    original_dims, intrinsic_dim, drr = result
    assert original_dims == 3
    assert intrinsic_dim >= 1
    assert 0 <= drr <= 1


def test_convenience_function_with_list():
    """Test convenience function with list input."""
    # Create test list
    data = [[1, 2], [3, 4], [5, 6], [7, 8]]
    
    result = estimate_intrinsic_dimension(data)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    original_dims, intrinsic_dim, drr = result
    assert original_dims == 2
    assert intrinsic_dim >= 1
    assert 0 <= drr <= 1


def test_convenience_function_with_custom_parameters():
    """Test convenience function with custom parameters."""
    data = np.random.randn(10, 4)
    
    result = estimate_intrinsic_dimension(
        data, 
        max_samples=100, 
        distance_metric='euclidean'
    )
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    original_dims, intrinsic_dim, drr = result
    assert original_dims == 4
    assert intrinsic_dim >= 1
    assert 0 <= drr <= 1


def test_convenience_function_different_metrics():
    """Test convenience function with different distance metrics."""
    data = np.random.randn(8, 3)
    
    metrics = ['l1', 'l2', 'euclidean', 'manhattan', 'cosine']
    
    for metric in metrics:
        result = estimate_intrinsic_dimension(data, distance_metric=metric)
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        
        original_dims, intrinsic_dim, drr = result
        assert original_dims == 3
        assert intrinsic_dim >= 1
        assert 0 <= drr <= 1


def test_convenience_function_max_samples_parameter():
    """Test convenience function with different max_samples values."""
    data = np.random.randn(20, 3)
    
    max_samples_values = [10, 50, 100]
    
    for max_samples in max_samples_values:
        result = estimate_intrinsic_dimension(data, max_samples=max_samples)
        
        assert isinstance(result, tuple)
        assert len(result) == 3


def test_all_exports():
    """Test that __all__ contains expected exports."""
    expected_exports = [
        'IntrinsicDimensionEstimator',
        'DataProcessor', 
        'BatchProcessor',
        'estimate_intrinsic_dimension',
        '__version__',
    ]
    
    assert hasattr(drr, '__all__')
    assert isinstance(drr.__all__, list)
    
    for export in expected_exports:
        assert export in drr.__all__


def test_imports_from_package():
    """Test that classes can be imported from package namespace."""
    # Test that these don't raise ImportError
    from drr import IntrinsicDimensionEstimator
    from drr import DataProcessor
    from drr import BatchProcessor
    from drr import estimate_intrinsic_dimension
    
    # Test functionality
    data = np.array([[1, 2], [3, 4], [5, 6]])
    
    estimator = IntrinsicDimensionEstimator()
    result = estimator.estimate(data)
    
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_convenience_function_error_handling():
    """Test convenience function handles errors gracefully."""
    # Test with invalid data types - should convert or handle gracefully
    try:
        # Empty data
        result = estimate_intrinsic_dimension([])
        # Should either succeed or raise appropriate error
    except (ValueError, IndexError):
        # Expected for empty data
        pass
    
    try:
        # Single point
        result = estimate_intrinsic_dimension([[1, 2]])
        # Should either succeed or raise appropriate error  
    except ValueError:
        # Expected for insufficient samples
        pass


def test_package_structure():
    """Test that package structure is correct."""
    # Test that drr module has expected attributes
    assert hasattr(drr, 'IntrinsicDimensionEstimator')
    assert hasattr(drr, 'DataProcessor')
    assert hasattr(drr, 'BatchProcessor')
    assert hasattr(drr, 'estimate_intrinsic_dimension')
    
    # Test that these are the correct types
    assert callable(drr.IntrinsicDimensionEstimator)
    assert callable(drr.DataProcessor)
    assert callable(drr.BatchProcessor)
    assert callable(drr.estimate_intrinsic_dimension)


def test_pandas_import_handling():
    """Test that pandas import is handled correctly in convenience function."""
    # Test with actual pandas DataFrame
    df = pd.DataFrame(np.random.randn(5, 3), columns=['A', 'B', 'C'])
    
    result = estimate_intrinsic_dimension(df)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    original_dims, intrinsic_dim, drr = result
    assert original_dims == 3


def test_numpy_import_handling():
    """Test that numpy import is handled correctly in convenience function."""
    # Test with numpy array
    arr = np.random.randn(5, 3)
    
    result = estimate_intrinsic_dimension(arr)
    
    assert isinstance(result, tuple)
    assert len(result) == 3
    
    original_dims, intrinsic_dim, drr = result
    assert original_dims == 3
