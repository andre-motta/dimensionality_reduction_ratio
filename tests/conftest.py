"""
Test configuration and fixtures for the DRR package.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drr import IntrinsicDimensionEstimator, DataProcessor, BatchProcessor


@pytest.fixture
def sample_data_2d():
    """Generate 2D sample data with known intrinsic dimension."""
    np.random.seed(42)
    # Create data that lies approximately on a 1D manifold (line)
    t = np.linspace(0, 2*np.pi, 200)
    data = np.column_stack([
        t + 0.05 * np.random.randn(200),  # x = t + noise
        2*t + 0.05 * np.random.randn(200)  # y = 2t + noise
    ])
    return data


@pytest.fixture
def sample_data_3d():
    """Generate 3D sample data with known intrinsic dimension."""
    np.random.seed(42)
    # Create data on a 2D manifold (circle in 3D space)
    t = np.linspace(0, 2*np.pi, 300)
    s = np.linspace(0, 1, 300)
    data = np.column_stack([
        np.cos(t) + 0.05 * np.random.randn(300),
        np.sin(t) + 0.05 * np.random.randn(300),
        s + 0.05 * np.random.randn(300)
    ])
    return data


@pytest.fixture
def sample_csv_data():
    """Create a temporary CSV file with test data."""
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100), 
        'feature3': np.random.randn(100),
        'target+': np.random.randn(100),  # Goal variable (to be removed)
        'score-': np.random.randn(100)     # Goal variable (to be removed)
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def batch_config_file():
    """Create a temporary batch configuration file."""
    config_content = """# Test configuration
test_folder
    subfolder
        dataset1
        dataset2
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(config_content)
        yield f.name
    
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def estimator():
    """Create a default IntrinsicDimensionEstimator."""
    return IntrinsicDimensionEstimator(max_samples=500, distance_metric='l1')


@pytest.fixture
def data_processor():
    """Create a default DataProcessor."""
    return DataProcessor(max_rows=1000, seed=42)


@pytest.fixture
def batch_processor(temp_results_dir):
    """Create a BatchProcessor with temporary output files."""
    return BatchProcessor(
        results_file=os.path.join(temp_results_dir, "test_results.csv"),
        error_log_file=os.path.join(temp_results_dir, "test_errors.log"),
        max_samples=500,
        distance_metric='l1'
    )


# Helper functions for tests
def create_test_dataset(shape, intrinsic_dim=None, seed=42):
    """Create a test dataset with specified properties."""
    np.random.seed(seed)
    
    if intrinsic_dim is None:
        # Random data
        return np.random.randn(*shape)
    
    # Create data with specific intrinsic dimension
    if intrinsic_dim == 1:
        # 1D manifold
        t = np.linspace(0, 4*np.pi, shape[0])
        data = np.zeros(shape)
        data[:, 0] = t
        for i in range(1, shape[1]):
            data[:, i] = np.sin(i * t) + 0.1 * np.random.randn(shape[0])
        return data
    
    elif intrinsic_dim == 2:
        # 2D manifold
        u = np.random.uniform(0, 2*np.pi, shape[0])
        v = np.random.uniform(0, 1, shape[0])
        data = np.zeros(shape)
        data[:, 0] = np.cos(u) * v
        data[:, 1] = np.sin(u) * v
        for i in range(2, shape[1]):
            data[:, i] = 0.1 * np.random.randn(shape[0])
        return data
    
    else:
        # Default to random
        return np.random.randn(*shape)
