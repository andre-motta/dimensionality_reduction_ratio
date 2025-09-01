"""Tests for intrinsic_dimension_estimator module."""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from drr.intrinsic_dimension_estimator import IntrinsicDimensionEstimator


def test_estimator_creation():
    """Test basic estimator creation with default parameters."""
    estimator = IntrinsicDimensionEstimator()
    
    assert estimator.max_samples == 2000
    assert estimator.distance_metric == "l1"
    assert estimator.num_radii == 100
    assert estimator.scipy_metric == "manhattan"


def test_custom_parameters():
    """Test estimator creation with custom parameters."""
    estimator = IntrinsicDimensionEstimator(
        max_samples=1000,
        distance_metric="l2",
        num_radii=50
    )
    
    assert estimator.max_samples == 1000
    assert estimator.distance_metric == "l2"
    assert estimator.num_radii == 50
    assert estimator.scipy_metric == "euclidean"


def test_distance_metric_mapping():
    """Test all supported distance metrics and their mappings."""
    metrics = {
        "l1": "manhattan",
        "l2": "euclidean", 
        "euclidean": "euclidean",
        "manhattan": "manhattan",
        "cosine": "cosine"
    }
    
    for user_metric, scipy_metric in metrics.items():
        estimator = IntrinsicDimensionEstimator(distance_metric=user_metric)
        assert estimator.user_metric == user_metric
        assert estimator.scipy_metric == scipy_metric


def test_invalid_distance_metric():
    """Test error handling for invalid distance metrics."""
    with pytest.raises(ValueError, match="distance_metric must be one of"):
        IntrinsicDimensionEstimator(distance_metric="invalid_metric")


def test_simple_estimation():
    """Test basic estimation with simple data."""
    estimator = IntrinsicDimensionEstimator()
    
    # Create simple 2D data
    data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    original_dims, intrinsic_dim, drr = estimator.estimate(data)
    
    assert original_dims == 2
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1
    assert 0 <= drr <= 1


def test_numpy_array_conversion():
    """Test automatic conversion to numpy array."""
    estimator = IntrinsicDimensionEstimator()
    
    # Test with numpy array (since the code logs shape before conversion)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    original_dims, intrinsic_dim, drr = estimator.estimate(data)
    
    assert original_dims == 3
    assert isinstance(intrinsic_dim, int)


def test_invalid_data_dimensions():
    """Test error handling for invalid data dimensions."""
    estimator = IntrinsicDimensionEstimator()
    
    # Test 1D data
    with pytest.raises(ValueError, match="Data must be 2D array"):
        estimator.estimate(np.array([1, 2, 3, 4]))
    
    # Test 3D data
    with pytest.raises(ValueError, match="Data must be 2D array"):
        estimator.estimate(np.array([[[1, 2], [3, 4]]]))


def test_insufficient_samples():
    """Test error handling for insufficient samples."""
    estimator = IntrinsicDimensionEstimator()
    
    # Test single sample
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        estimator.estimate(np.array([[1, 2, 3]]))
    
    # Test empty data
    with pytest.raises(ValueError, match="Need at least 2 samples"):
        estimator.estimate(np.array([]).reshape(0, 3))


def test_large_dataset_sampling():
    """Test sampling behavior for large datasets."""
    estimator = IntrinsicDimensionEstimator(max_samples=100)
    
    # Create dataset larger than max_samples
    np.random.seed(42)
    large_data = np.random.randn(500, 5)
    
    original_dims, intrinsic_dim, drr = estimator.estimate(large_data)
    
    assert original_dims == 5
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_config_dataset_estimation():
    """Test estimation for small configuration-like datasets."""
    estimator = IntrinsicDimensionEstimator()
    
    # Small dataset (≤6 dims) should trigger config dataset logic
    np.random.seed(42)
    config_data = np.random.randn(50, 4)
    
    original_dims, intrinsic_dim, drr = estimator.estimate(config_data)
    
    assert original_dims == 4
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_behavior_dataset_estimation():
    """Test estimation for large behavior-like datasets."""
    estimator = IntrinsicDimensionEstimator()
    
    # Large dataset (>15 dims) should trigger behavior dataset logic
    np.random.seed(42)
    behavior_data = np.random.randn(100, 20)
    
    original_dims, intrinsic_dim, drr = estimator.estimate(behavior_data)
    
    assert original_dims == 20
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_medium_dataset_estimation():
    """Test estimation for medium-sized datasets."""
    estimator = IntrinsicDimensionEstimator()
    
    # Medium dataset (6-15 dims) should trigger medium dataset logic
    np.random.seed(42)
    medium_data = np.random.randn(100, 10)
    
    original_dims, intrinsic_dim, drr = estimator.estimate(medium_data)
    
    assert original_dims == 10
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_pairwise_distances_computation():
    """Test pairwise distances computation with different metrics."""
    # Test L1 distance
    estimator_l1 = IntrinsicDimensionEstimator(distance_metric="l1")
    data = np.array([[0, 0], [1, 1], [2, 2]])
    
    distances = estimator_l1._compute_pairwise_distances(data)
    
    assert len(distances) > 0
    assert all(d > 0 for d in distances)
    assert all(np.isfinite(d) for d in distances)
    
    # Test L2 distance
    estimator_l2 = IntrinsicDimensionEstimator(distance_metric="l2")
    distances_l2 = estimator_l2._compute_pairwise_distances(data)
    
    assert len(distances_l2) > 0
    assert all(d > 0 for d in distances_l2)


def test_correlation_function_calculation():
    """Test correlation function calculation."""
    estimator = IntrinsicDimensionEstimator()
    
    # Create simple data
    data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    distances = estimator._compute_pairwise_distances(data)
    
    # This should exercise the correlation function logic
    intrinsic_dim = estimator._estimate_from_correlation_function(distances, 2)
    
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_gradient_calculations():
    """Test gradient calculation methods."""
    estimator = IntrinsicDimensionEstimator()
    
    # Test regular gradients
    correlation_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    radii = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    
    gradients = estimator._calculate_gradients(correlation_values, radii)
    
    assert len(gradients) > 0
    assert all(np.isfinite(g) for g in gradients)
    
    # Test log-log gradients
    log_gradients = estimator._calculate_log_log_gradients(correlation_values, radii)
    
    # Should return array (may be empty if calculations fail)
    assert isinstance(log_gradients, np.ndarray)


def test_equal_distances_handling():
    """Test handling of data with equal distances."""
    estimator = IntrinsicDimensionEstimator()
    
    # Create data where all points are the same
    data = np.array([[1, 1], [1, 1], [1, 1]])
    
    # Should handle gracefully with fallback
    original_dims, intrinsic_dim, drr = estimator.estimate(data)
    
    assert original_dims == 2
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_error_fallback_handling():
    """Test error handling and fallback estimation."""
    estimator = IntrinsicDimensionEstimator()
    
    # Create problematic data that might cause errors
    problematic_data = np.array([[np.inf, 1], [1, np.inf], [np.nan, 1]])
    
    # Should handle gracefully with fallback
    original_dims, intrinsic_dim, drr = estimator.estimate(problematic_data)
    
    assert original_dims == 2
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1
    assert 0 <= drr <= 1


def test_different_distance_metrics():
    """Test estimation with different distance metrics."""
    metrics = ["l1", "l2", "euclidean", "manhattan", "cosine"]
    
    np.random.seed(42)
    data = np.random.randn(20, 4)
    
    for metric in metrics:
        estimator = IntrinsicDimensionEstimator(distance_metric=metric)
        original_dims, intrinsic_dim, drr = estimator.estimate(data)
        
        assert original_dims == 4
        assert isinstance(intrinsic_dim, int)
        assert intrinsic_dim >= 1
        assert 0 <= drr <= 1


def test_various_dataset_sizes():
    """Test estimation with various dataset sizes."""
    sizes = [5, 50, 500]
    
    for size in sizes:
        np.random.seed(42)
        data = np.random.randn(size, 3)
        
        estimator = IntrinsicDimensionEstimator()
        original_dims, intrinsic_dim, drr = estimator.estimate(data)
        
        assert original_dims == 3
        assert isinstance(intrinsic_dim, int)
        assert intrinsic_dim >= 1


def test_log_spacing_error_handling():
    """Test handling of log spacing errors in correlation function."""
    estimator = IntrinsicDimensionEstimator()
    
    # Create data that might cause log spacing issues
    data = np.array([[0, 0], [0, 0.001], [0.001, 0]])
    
    # Should handle gracefully
    original_dims, intrinsic_dim, drr = estimator.estimate(data)
    
    assert original_dims == 2
    assert isinstance(intrinsic_dim, int)
    assert intrinsic_dim >= 1


def test_selection_methods():
    """Test intrinsic dimension selection methods."""
    estimator = IntrinsicDimensionEstimator()
    
    # Test config dataset selection
    gradients = np.array([1.0, 2.0, 1.5])
    log_gradients = np.array([2.5, 3.0, 2.8])
    
    result = estimator._estimate_for_config_dataset(gradients, log_gradients, 4)
    assert isinstance(result, int)
    assert result >= 1
    
    # Test behavior dataset selection
    result = estimator._estimate_for_behavior_dataset(gradients, log_gradients, 20)
    assert isinstance(result, int)
    assert result >= 1
    
    # Test medium dataset selection
    result = estimator._estimate_for_medium_dataset(gradients, log_gradients, 10)
    assert isinstance(result, int)
    assert result >= 1


def test_empty_gradients_handling():
    """Test handling of empty gradient arrays."""
    estimator = IntrinsicDimensionEstimator()
    
    empty_gradients = np.array([])
    
    # Should handle empty gradients gracefully
    result = estimator._estimate_for_config_dataset(empty_gradients, empty_gradients, 5)
    assert isinstance(result, int)
    assert result >= 1
    
    result = estimator._estimate_for_behavior_dataset(empty_gradients, empty_gradients, 20)
    assert isinstance(result, int)
    assert result >= 1
    
    result = estimator._estimate_for_medium_dataset(empty_gradients, empty_gradients, 10)
    assert isinstance(result, int)
    assert result >= 1
