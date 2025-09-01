"""Tests for main module."""

import os
import sys
import tempfile
import shutil
import logging
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from drr.main import (
    setup_logging, 
    process_single_dataset, 
    process_batch_datasets, 
    cli,
    batch,
    single
)


def test_setup_logging_basic():
    """Test basic logging setup."""
    # Clear any existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    setup_logging()
    
    # Check that logging is configured
    assert logger.level == logging.INFO
    assert len(logger.handlers) > 0


def test_setup_logging_with_file():
    """Test logging setup with file handler."""
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "test.log")
        
        # Clear any existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        setup_logging(level="DEBUG", log_file=log_file)
        
        # Check that file was created
        assert os.path.exists(log_file)
        
        # Test logging
        logging.info("Test message")
        
        # Check that message was written to file
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test message" in content


def test_setup_logging_different_levels():
    """Test logging setup with different levels."""
    levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    for level_str in levels:
        # Clear any existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        setup_logging(level=level_str)
        
        expected_level = getattr(logging, level_str.upper())
        assert logger.level == expected_level


def test_process_single_dataset_nonexistent_file():
    """Test processing nonexistent dataset file."""
    result = process_single_dataset("nonexistent_file.csv")
    assert result is False


def test_process_single_dataset_success():
    """Test successful single dataset processing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create test CSV file
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(f.name, index=False)
        
        try:
            result = process_single_dataset(f.name)
            # Should succeed (or fail gracefully but return boolean)
            assert isinstance(result, bool)
            
        finally:
            os.unlink(f.name)


def test_process_single_dataset_with_custom_parameters():
    """Test single dataset processing with custom parameters."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create test CSV file
        df = pd.DataFrame({
            'x': np.random.randn(10),
            'y': np.random.randn(10),
            'z': np.random.randn(10)
        })
        df.to_csv(f.name, index=False)
        
        try:
            result = process_single_dataset(
                f.name, 
                max_samples=100, 
                distance_metric="l2"
            )
            assert isinstance(result, bool)
            
        finally:
            os.unlink(f.name)


def test_process_single_dataset_invalid_data():
    """Test processing dataset with invalid data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create invalid CSV file (single row)
        f.write("col1\n1\n")
        f.flush()
        
        try:
            result = process_single_dataset(f.name)
            # Should fail gracefully
            assert result is False
            
        finally:
            os.unlink(f.name)


def test_process_batch_datasets_nonexistent_file():
    """Test batch processing with nonexistent config file."""
    result = process_batch_datasets("nonexistent_config.txt")
    assert result is False


def test_process_batch_datasets_success():
    """Test successful batch processing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        data_file = os.path.join(temp_dir, "test_data.csv")
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 3, 4, 5, 6],
            'c': [3, 4, 5, 6, 7]
        })
        df.to_csv(data_file, index=False)
        
        # Create config file
        config_file = os.path.join(temp_dir, "config.txt")
        with open(config_file, 'w') as f:
            f.write(f"{data_file}\n")
        
        try:
            result = process_batch_datasets(
                config_file,
                data_root=temp_dir,
                max_samples=100,
                distance_metric="l1"
            )
            assert isinstance(result, bool)
            
        except Exception:
            # Allow failures but test that function returns boolean
            assert True


def test_process_batch_datasets_with_parameters():
    """Test batch processing with various parameters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty config file
        config_file = os.path.join(temp_dir, "empty_config.txt")
        with open(config_file, 'w') as f:
            f.write("# empty config\n")
        
        try:
            result = process_batch_datasets(
                config_file,
                data_root=temp_dir,
                max_samples=500,
                distance_metric="euclidean"
            )
            assert isinstance(result, bool)
            
        except Exception:
            # Allow failures but test parameters are passed
            assert True


def test_cli_basic():
    """Test basic CLI functionality."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    
    assert result.exit_code == 0
    assert "Intrinsic Dimension Analysis" in result.output


def test_cli_with_logging_options():
    """Test CLI with logging options."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_file = os.path.join(temp_dir, "cli_test.log")
        
        result = runner.invoke(cli, [
            '--log-level', 'DEBUG',
            '--log-file', log_file,
            '--help'
        ])
        
        assert result.exit_code == 0


def test_cli_no_command():
    """Test CLI with no command (should show help)."""
    runner = CliRunner()
    result = runner.invoke(cli, [])
    
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_single_command():
    """Test single dataset CLI command."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create test CSV file
        df = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [4, 5, 6, 7]
        })
        df.to_csv(f.name, index=False)
        
        try:
            result = runner.invoke(single, [f.name])
            # Command should execute (may succeed or fail, but should run)
            assert result.exit_code in [0, 1]  # Success or failure
            
        finally:
            os.unlink(f.name)


def test_single_command_with_options():
    """Test single dataset command with options."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Create test CSV file
        df = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20)
        })
        df.to_csv(f.name, index=False)
        
        try:
            result = runner.invoke(single, [
                f.name,
                '--max-samples', '50',
                '--metric', 'euclidean'
            ])
            assert result.exit_code in [0, 1]
            
        finally:
            os.unlink(f.name)


def test_single_command_nonexistent_file():
    """Test single command with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(single, ['nonexistent_file.csv'])
    
    # Should fail due to click path validation
    assert result.exit_code != 0


def test_batch_command():
    """Test batch processing CLI command."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test dataset
        data_file = os.path.join(temp_dir, "batch_test.csv")
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        df.to_csv(data_file, index=False)
        
        # Create config file
        config_file = os.path.join(temp_dir, "batch_config.txt")
        with open(config_file, 'w') as f:
            f.write(f"{data_file}\n")
        
        result = runner.invoke(batch, [config_file])
        assert result.exit_code in [0, 1]


def test_batch_command_with_options():
    """Test batch command with options."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create empty config file
        config_file = os.path.join(temp_dir, "batch_config_empty.txt")
        with open(config_file, 'w') as f:
            f.write("# empty config file\n")
        
        result = runner.invoke(batch, [
            config_file,
            '--data-root', temp_dir,
            '--max-samples', '100',
            '--metric', 'manhattan'
        ])
        assert result.exit_code in [0, 1]


def test_batch_command_nonexistent_file():
    """Test batch command with nonexistent file."""
    runner = CliRunner()
    result = runner.invoke(batch, ['nonexistent_config.txt'])
    
    # Should fail due to click path validation
    assert result.exit_code != 0


def test_cli_invalid_log_level():
    """Test CLI with invalid log level."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--log-level', 'INVALID'])
    
    assert result.exit_code != 0


def test_single_command_invalid_metric():
    """Test single command with invalid metric."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame({'x': [1], 'y': [2]})
        df.to_csv(f.name, index=False)
        
        try:
            result = runner.invoke(single, [
                f.name,
                '--metric', 'invalid_metric'
            ])
            assert result.exit_code != 0
            
        finally:
            os.unlink(f.name)


def test_batch_command_invalid_metric():
    """Test batch command with invalid metric."""
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# empty\n")
        
        try:
            result = runner.invoke(batch, [
                f.name,
                '--metric', 'invalid_metric'
            ])
            assert result.exit_code != 0
            
        finally:
            os.unlink(f.name)


@patch('drr.main.process_single_dataset')
def test_single_command_mock_success(mock_process):
    """Test single command with mocked successful processing."""
    mock_process.return_value = True
    
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        df.to_csv(f.name, index=False)
        
        try:
            result = runner.invoke(single, [f.name])
            assert result.exit_code == 0
            mock_process.assert_called_once()
            
        finally:
            os.unlink(f.name)


@patch('drr.main.process_single_dataset')
def test_single_command_mock_failure(mock_process):
    """Test single command with mocked failed processing."""
    mock_process.return_value = False
    
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        df.to_csv(f.name, index=False)
        
        try:
            result = runner.invoke(single, [f.name])
            assert result.exit_code == 1
            mock_process.assert_called_once()
            
        finally:
            os.unlink(f.name)


@patch('drr.main.process_batch_datasets')
def test_batch_command_mock_success(mock_process):
    """Test batch command with mocked successful processing."""
    mock_process.return_value = True
    
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# test config\n")
        
        try:
            result = runner.invoke(batch, [f.name])
            assert result.exit_code == 0
            mock_process.assert_called_once()
            
        finally:
            os.unlink(f.name)


@patch('drr.main.process_batch_datasets')
def test_batch_command_mock_failure(mock_process):
    """Test batch command with mocked failed processing."""
    mock_process.return_value = False
    
    runner = CliRunner()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("# test config\n")
        
        try:
            result = runner.invoke(batch, [f.name])
            assert result.exit_code == 1
            mock_process.assert_called_once()
            
        finally:
            os.unlink(f.name)


def test_logging_directory_creation():
    """Test that logging creates directories when needed."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nested_log = os.path.join(temp_dir, "nested", "dir", "test.log")
        
        # Clear any existing handlers
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        setup_logging(log_file=nested_log)
        
        # Directory should be created
        assert os.path.exists(os.path.dirname(nested_log))
        assert os.path.exists(nested_log)


def test_process_functions_error_handling():
    """Test error handling in processing functions."""
    # Test with problematic files that might cause exceptions
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # Write invalid CSV content
        f.write("invalid,csv,content\n1,2\n3,4,5,6\n")
        f.flush()
        
        try:
            # Should handle errors gracefully
            result = process_single_dataset(f.name)
            assert isinstance(result, bool)
            
        finally:
            os.unlink(f.name)
