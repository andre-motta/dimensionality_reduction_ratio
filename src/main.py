#!/usr/bin/env python3
"""
Main entry point for Intrinsic Dimension Analysis

This script provides a command-line interface for:
1. Processing individual datasets
2. Batch processing multiple datasets from configuration files
3. Analyzing intrinsic dimensionality with DRR metrics

Usage:
    cd src
    python main.py --batch ../config/datasets.txt
    python main.py --single ../data/optimize/config/SS-A.csv
    python main.py --help
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

from batch_processor import BatchProcessor
from intrinsic_dimension import IntrinsicDimensionEstimator
from data_processor import DataProcessor


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)


def process_single_dataset(dataset_path: str, max_samples: int = 2000,
                          distance_metric: str = 'l1'):
    """Process a single dataset file."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False
    
    logger.info(f"Processing single dataset: {dataset_path}")
    
    try:
        # Initialize processors
        data_processor = DataProcessor()
        estimator = IntrinsicDimensionEstimator(
            max_samples=max_samples,
            distance_metric=distance_metric
        )
        
        # Process the dataset
        processed_data, metadata = data_processor.process_dataset(dataset_path)
        
        if not data_processor.validate_processed_data(processed_data):
            logger.error("Data validation failed")
            return False
        
        # Estimate intrinsic dimension
        original_dims, intrinsic_dim, drr = estimator.estimate(processed_data)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"RESULTS FOR: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")
        print(f"Original Dimensions (R): {original_dims}")
        print(f"Intrinsic Dimension (I): {intrinsic_dim}")
        print(f"DRR (1 - I/R): {drr:.3f}")
        print(f"Data Quality: {drr:.1%} dimensionality reduction")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return False


def process_batch_datasets(datasets_file: str, data_root: str = "../data",
                          max_samples: int = 2000, distance_metric: str = 'l1'):
    """Process multiple datasets from configuration file."""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(datasets_file):
        logger.error(f"Datasets configuration file not found: {datasets_file}")
        return False
    
    logger.info(f"Starting batch processing from: {datasets_file}")
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(
            results_file="results/dataset_results.csv",
            error_log_file="logs/batch_errors.log",
            max_samples=max_samples,
            distance_metric=distance_metric
        )
        
        # Process all datasets
        results = processor.process_datasets_from_file(datasets_file, data_root)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total datasets: {results['total_datasets']}")
        print(f"Already processed: {results['already_processed']}")
        print(f"Newly processed: {results['newly_processed']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Results file: {results['results_file']}")
        print(f"{'='*60}")
        
        return results['failed'] == 0
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intrinsic Dimension Analysis with DRR Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets from configuration file
  python main.py --batch ../config/datasets.txt
  
  # Process a single dataset
  python main.py --single ../data/optimize/config/SS-A.csv
  
  # Use custom parameters
  python main.py --batch ../config/datasets.txt --max-samples 5000 --metric euclidean
  
  # Enable debug logging
  python main.py --batch ../config/datasets.txt --log-level DEBUG --log-file ../logs/debug.log
        """
    )
    
    # Main operation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--batch', '-b',
        type=str,
        help='Process multiple datasets from configuration file (e.g., datasets.txt)'
    )
    group.add_argument(
        '--single', '-s',
        type=str,
        help='Process a single dataset file'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--max-samples',
        type=int,
        default=2000,
        help='Maximum number of samples for large datasets (default: 2000)'
    )
    parser.add_argument(
        '--metric', '--distance-metric',
        type=str,
        default='l1',
        choices=['l1', 'l2', 'euclidean', 'manhattan', 'cosine'],
        help='Distance metric for analysis (default: l1)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='../data',
        help='Root directory for dataset files (default: ../data)'
    )
    
    # Logging options
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (optional)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Intrinsic Dimension Analysis")
    logger.info(f"Arguments: {vars(args)}")
    
    # Process based on mode
    success = False
    
    if args.batch:
        success = process_batch_datasets(
            datasets_file=args.batch,
            data_root=args.data_root,
            max_samples=args.max_samples,
            distance_metric=args.metric
        )
    elif args.single:
        success = process_single_dataset(
            dataset_path=args.single,
            max_samples=args.max_samples,
            distance_metric=args.metric
        )
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    logger.info(f"Analysis complete. Exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
