"""Simple tests for DataProcessor module."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Add src to Python path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from drr.data_processor import DataProcessor


def test_processor_creation():
    """Test creating a DataProcessor."""
    processor = DataProcessor()
    assert processor is not None
    assert processor.max_rows_for_processing == 5000


def test_simple_dataframe_processing():
    """Test processing a simple DataFrame."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    
    result, metadata = processor.process_dataset(data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3
    assert 'original_shape' in metadata


def test_numpy_array_input():
    """Test processing numpy array input."""
    processor = DataProcessor()
    
    data = np.array([[1, 2], [3, 4], [5, 6]])
    
    result, metadata = processor.process_dataset(data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)


def test_goal_variable_removal():
    """Test removal of goal variables."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'feature': [1, 2, 3],
        'target+': [4, 5, 6]  # This should be removed
    })
    
    result, metadata = processor.process_dataset(data)
    
    assert result.shape[1] == 1  # Only feature column should remain
    assert 'target+' in metadata['goal_variables_removed']


def test_custom_parameters():
    """Test custom initialization parameters."""
    processor = DataProcessor(max_rows_for_processing=100, random_seed=999)
    
    assert processor.max_rows_for_processing == 100
    assert processor.random_seed == 999


def test_csv_file_processing():
    """Test processing CSV file input - hits line 70-71."""
    import tempfile
    processor = DataProcessor()
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("col1,col2\n1,2\n3,4\n5,6\n")
        temp_file = f.name
    
    try:
        result, metadata = processor.process_dataset(temp_file)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)
    finally:
        os.unlink(temp_file)


def test_unsupported_data_type():
    """Test unsupported data type - hits line 75."""
    processor = DataProcessor()
    
    # Use an unsupported type (not string, not DataFrame, not numpy array)
    with pytest.raises(ValueError, match="Unsupported data type"):
        processor.process_dataset(123)  # Integer is not supported


def test_no_features_after_goal_removal():
    """Test when no features remain after goal removal - hits line 104."""
    processor = DataProcessor()
    
    # DataFrame with only goal variables
    data = pd.DataFrame({
        'target+': [1, 2, 3],
        'minimize-': [4, 5, 6],
        'class!': [7, 8, 9]
    })
    
    with pytest.raises(ValueError, match="No feature columns remaining"):
        processor.process_dataset(data)


def test_large_dataset_sampling():
    """Test sampling for large datasets - hits lines 91-97."""
    processor = DataProcessor(max_rows_for_processing=10)
    
    # Create dataset larger than max_rows_for_processing
    large_data = pd.DataFrame({
        'feature1': range(50),
        'feature2': range(50, 100)
    })
    
    result, metadata = processor.process_dataset(large_data)
    
    assert metadata['sampling_applied'] is True
    assert 'sampled_shape' in metadata
    assert result.shape[0] <= 10


def test_mixed_data_types():
    """Test processing mixed data types - hits lines 120-126."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'numeric': [1.5, 2.5, 3.5],
        'categorical': ['A', 'B', 'C'],
        'boolean': [True, False, True]
    })
    
    result, metadata = processor.process_dataset(data)
    
    assert isinstance(result, np.ndarray)
    assert len(metadata['categorical_columns']) > 0
    assert len(metadata['numeric_columns']) > 0


def test_problematic_categorical_data():
    """Test categorical data that needs special handling - hits lines 187-188, 196-200."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'mixed_types': ['text', 123, 'more_text', None],  # Mixed types
        'weird_strings': ['', '  ', 'normal', 'data']     # Empty/whitespace strings
    })
    
    result, metadata = processor.process_dataset(data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 4


def test_data_with_missing_values():
    """Test data with missing values - hits lines 222-229."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'feature1': [1.0, np.nan, 3.0, 4.0],
        'feature2': [np.nan, 2.0, np.nan, 4.0],
        'feature3': [1.0, 2.0, 3.0, np.nan]
    })
    
    result, metadata = processor.process_dataset(data)
    
    # Should handle missing values
    assert isinstance(result, np.ndarray)
    assert not np.isnan(result).all()  # Should not be all NaN


def test_empty_dataframe():
    """Test processing empty DataFrame - hits lines 258-269."""
    processor = DataProcessor()
    
    data = pd.DataFrame()
    
    with pytest.raises((ValueError, IndexError)):
        processor.process_dataset(data)


def test_single_row_data():
    """Test processing single row of data - hits lines 285-286."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'feature1': [42],
        'feature2': [99]
    })
    
    result, metadata = processor.process_dataset(data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 2)


def test_complex_goal_variables():
    """Test complex goal variable patterns - hits lines 296-297, 300-303."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'cost-': [4, 5, 6],      # - goal variable (ends with -)
        'profit+': [7, 8, 9],    # + goal variable (ends with +)
        'class!': [10, 11, 12],  # ! goal variable (ends with !)
        'feature2': [13, 14, 15]
    })
    
    result, metadata = processor.process_dataset(data)
    
    expected_goals = ['cost-', 'profit+', 'class!']
    assert len(set(metadata['goal_variables_removed']).intersection(set(expected_goals))) > 0
    assert result.shape[1] == 2  # Only 2 feature columns should remain


def test_force_numeric_conversion():
    """Test data that requires force conversion - hits lines 196-200."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'weird_data': ['123abc', '456def', '789ghi'],  # Strings with numbers
        'numbers': [1, 2, 3]
    })
    
    result, metadata = processor.process_dataset(data)
    
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3


def test_data_quality_checks():
    """Test data quality checks with NaN and inf values - hits lines 285-286, 331-348."""
    processor = DataProcessor()
    
    data = pd.DataFrame({
        'feature1': [1.0, np.nan, np.inf],
        'feature2': [np.nan, -np.inf, 3.0]
    })
    
    result, metadata = processor.process_dataset(data)
    
    # Should handle NaN and inf values
    assert isinstance(result, np.ndarray)
    assert not np.isnan(result).any()
    assert not np.isinf(result).any()
    assert 'data_quality' in metadata
    assert 'final_shape' in metadata
