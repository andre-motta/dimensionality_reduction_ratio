#!/usr/bin/env python3
"""
Example Usage of Intrinsic Dimension Analysis Toolkit

This script demonstrates various ways to use the intrinsic dimension
analysis toolkit for different scenarios.
"""

import sys
import os
import logging
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.intrinsic_dimension import IntrinsicDimensionEstimator
from src.data_processor import DataProcessor
from src.batch_processor import BatchProcessor


def setup_logging():
    """Setup basic logging for examples."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def example_single_dataset():
    """Example 1: Process a single dataset."""
    print("="*60)
    print("EXAMPLE 1: Single Dataset Processing")
    print("="*60)
    
    # Initialize components
    processor = DataProcessor()
    estimator = IntrinsicDimensionEstimator(max_samples=2000, distance_metric='l1')
    
    # Process a sample dataset
    dataset_path = "../data/optimize/config/SS-A.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure the dataset exists or modify the path.")
        return
    
    try:
        # Process the dataset
        print(f"Processing: {dataset_path}")
        processed_data, metadata = processor.process_dataset(dataset_path)
        
        print(f"Data shape after processing: {processed_data.shape}")
        print(f"Metadata: {metadata}")
        
        # Estimate intrinsic dimension
        original_dims, intrinsic_dim, drr = estimator.estimate(processed_data)
        
        # Display results
        print(f"\nResults:")
        print(f"  Raw Dimensions (R): {original_dims}")
        print(f"  Intrinsic Dimension (I): {intrinsic_dim}")
        print(f"  DRR (1 - I/R): {drr:.3f}")
        print(f"  Dimensionality Reduction: {drr:.1%}")
        
        # Interpretation
        if drr > 0.5:
            print(f"  → High dimensionality reduction potential!")
        elif drr > 0.3:
            print(f"  → Moderate dimensionality reduction potential.")
        else:
            print(f"  → Low dimensionality reduction potential.")
            
    except Exception as e:
        print(f"Error processing dataset: {e}")


def example_custom_parameters():
    """Example 2: Using custom parameters."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Parameters")
    print("="*60)
    
    # Initialize with custom parameters
    processor = DataProcessor()  # Use default parameters
    estimator = IntrinsicDimensionEstimator(
        max_samples=5000,           # Use more samples
        distance_metric='euclidean' # Different distance metric
    )
    
    dataset_path = "../data/optimize/config/SS-C.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    try:
        print(f"Processing with custom parameters: {dataset_path}")
        print(f"  Max samples for estimation: {estimator.max_samples}")
        print(f"  Distance metric: {estimator.distance_metric}")
        
        # Process
        processed_data, metadata = processor.process_dataset(dataset_path)
        original_dims, intrinsic_dim, drr = estimator.estimate(processed_data)
        
        print(f"\nResults with custom parameters:")
        print(f"  R={original_dims}, I={intrinsic_dim}, DRR={drr:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example 3: Batch processing demonstration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing")
    print("="*60)
    
    # Create a small test configuration
    test_config = """# Test configuration for examples
optimize
    config
        SS-A
        SS-C
"""
    
    # Write test configuration
    config_file = "test_datasets.txt"
    with open(config_file, 'w') as f:
        f.write(test_config)
    
    print(f"Created test configuration: {config_file}")
    print("Configuration contents:")
    print(test_config)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            results_file="example_results.csv",
            error_log_file="example_errors.log",
            max_samples=2000,
            distance_metric='l1'
        )
        
        # Process datasets
        print("Starting batch processing...")
        results = processor.process_datasets_from_file(config_file, data_root="../data")
        
        print(f"\nBatch Processing Results:")
        print(f"  Total datasets: {results['total_datasets']}")
        print(f"  Successfully processed: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Results saved to: {results['results_file']}")
        
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
            print(f"\nCleaned up: {config_file}")
            
    except Exception as e:
        print(f"Error in batch processing: {e}")


def example_error_handling():
    """Example 4: Error handling and validation."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Error Handling")
    print("="*60)
    
    processor = DataProcessor()
    estimator = IntrinsicDimensionEstimator()
    
    # Test with non-existent file
    print("Testing with non-existent file...")
    try:
        processed_data, metadata = processor.process_dataset("nonexistent.csv")
    except FileNotFoundError as e:
        print(f"  ✓ Caught expected error: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test with invalid data (if we had some)
    print("\nTesting data validation...")
    import numpy as np
    
    # Empty data
    empty_data = np.array([]).reshape(0, 0)
    if not processor.validate_processed_data(empty_data):
        print("  ✓ Empty data validation failed as expected")
    
    # Valid data
    valid_data = np.random.rand(100, 5)
    if processor.validate_processed_data(valid_data):
        print("  ✓ Valid data validation passed")
    
    # Test estimation with minimal data
    print("\nTesting estimation with minimal data...")
    try:
        minimal_data = np.random.rand(10, 3)  # Very small dataset
        r, i, drr = estimator.estimate(minimal_data)
        print(f"  ✓ Minimal data processed: R={r}, I={i}, DRR={drr:.3f}")
    except Exception as e:
        print(f"  ✗ Error with minimal data: {e}")


def example_different_distance_metrics():
    """Example 5: Comparing different distance metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Distance Metric Comparison")
    print("="*60)
    
    dataset_path = "../data/optimize/config/SS-A.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    # Process data once
    processor = DataProcessor()
    processed_data, metadata = processor.process_dataset(dataset_path)
    
    print(f"Comparing distance metrics on: {os.path.basename(dataset_path)}")
    print(f"Data shape: {processed_data.shape}")
    print()
    
    # Test different metrics
    metrics = ['l1', 'l2', 'euclidean', 'manhattan']
    results = {}
    
    for metric in metrics:
        try:
            estimator = IntrinsicDimensionEstimator(
                max_samples=2000,
                distance_metric=metric
            )
            
            r, i, drr = estimator.estimate(processed_data)
            results[metric] = {'R': r, 'I': i, 'DRR': drr}
            
            print(f"  {metric:>10}: R={r:2d}, I={i:2d}, DRR={drr:.3f}")
            
        except Exception as e:
            print(f"  {metric:>10}: Error - {e}")
    
    # Summary
    if results:
        print(f"\nSummary:")
        drr_values = [results[m]['DRR'] for m in results]
        print(f"  DRR range: {min(drr_values):.3f} - {max(drr_values):.3f}")
        print(f"  DRR std: {np.std(drr_values):.3f}")


def main():
    """Run all examples."""
    setup_logging()
    
    print("Intrinsic Dimension Analysis - Usage Examples")
    print("=" * 60)
    
    # Change to examples directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        example_single_dataset()
        example_custom_parameters()
        example_batch_processing()
        example_error_handling()
        example_different_distance_metrics()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)
        print("\nNext steps:")
        print("1. Try processing your own datasets")
        print("2. Modify the configuration files")
        print("3. Experiment with different parameters")
        print("4. Check the results and logs")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
