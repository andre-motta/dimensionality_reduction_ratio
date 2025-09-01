#!/usr/bin/env python3
"""
Main entry point for Intrinsic Dimension Analysis

This script provides a command-line interface for:
1. Processing individual datasets
2. Batch processing multiple datasets from configuration files
3. Analyzing intrinsic dimensionality with DRR metrics

Usage:
    cd src
    python main.py batch ../config/datasets.txt
    python main.py single ../data/optimize/config/SS-A.csv
    python main.py --help
"""

import logging
import os
import sys
from typing import Optional

import click

from .batch_processor import BatchProcessor
from .data_processor import DataProcessor
from .intrinsic_dimension_estimator import IntrinsicDimensionEstimator


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper())

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(file_handler)


def process_single_dataset(dataset_path: str, max_samples: int = 2000, distance_metric: str = "l1"):
    """Process a single dataset file."""
    logger = logging.getLogger(__name__)

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return False

    logger.info(f"Processing single dataset: {dataset_path}")

    try:
        # Initialize processors
        data_processor = DataProcessor()
        estimator = IntrinsicDimensionEstimator(max_samples=max_samples, distance_metric=distance_metric)

        # Process the dataset
        processed_data, metadata = data_processor.process_dataset(dataset_path)

        if not data_processor.validate_processed_data(processed_data):
            logger.error("Data validation failed")
            return False

        # Estimate intrinsic dimension
        original_dims, intrinsic_dim, drr = estimator.estimate(processed_data)

        # Print results
        click.echo(f"\n{'='*60}")
        click.echo(f"RESULTS FOR: {os.path.basename(dataset_path)}")
        click.echo(f"{'='*60}")
        click.echo(f"Original Dimensions (R): {original_dims}")
        click.echo(f"Intrinsic Dimension (I): {intrinsic_dim}")
        click.echo(f"DRR (1 - I/R): {drr:.3f}")
        click.echo(f"Data Quality: {drr:.1%} dimensionality reduction")
        click.echo(f"{'='*60}")

        return True

    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        return False


def process_batch_datasets(
    datasets_file: str,
    data_root: str = "../data",
    max_samples: int = 2000,
    distance_metric: str = "l1",
):
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
            distance_metric=distance_metric,
        )

        # Process all datasets
        results = processor.process_datasets_from_file(datasets_file, data_root)

        # Print summary
        click.echo(f"\n{'='*60}")
        click.echo("BATCH PROCESSING SUMMARY")
        click.echo(f"{'='*60}")
        click.echo(f"Total datasets: {results['total_datasets']}")
        click.echo(f"Already processed: {results['already_processed']}")
        click.echo(f"Newly processed: {results['newly_processed']}")
        click.echo(f"Successful: {results['successful']}")
        click.echo(f"Failed: {results['failed']}")
        click.echo(f"Results file: {results['results_file']}")
        click.echo(f"{'='*60}")

        return results["failed"] == 0

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return False


@click.group(invoke_without_command=True)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.option("--log-file", type=click.Path(), help="Log file path (optional)")
@click.pass_context
def cli(ctx, log_level, log_file):
    """
    Intrinsic Dimension Analysis with DRR Metrics

    A professional toolkit for estimating intrinsic dimensionality and
    computing Dimensionality Reduction Ratio (DRR) metrics.

    Use 'python main.py COMMAND --help' for command-specific help.
    """
    # Setup logging
    setup_logging(log_level, log_file)

    # Store common options in context
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["log_file"] = log_file

    logger = logging.getLogger(__name__)
    logger.info("Starting Intrinsic Dimension Analysis")

    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("datasets_file", type=click.Path(exists=True))
@click.option("--data-root", default="../data", help="Root directory for dataset files")
@click.option("--max-samples", default=2000, help="Maximum number of samples for large datasets")
@click.option(
    "--metric",
    default="l1",
    type=click.Choice(["l1", "l2", "euclidean", "manhattan", "cosine"]),
    help="Distance metric for analysis",
)
def batch(datasets_file, data_root, max_samples, metric):
    """
    Process multiple datasets from a configuration file.

    DATASETS_FILE: Path to the configuration file listing datasets to process

    Example:
        python main.py batch ../config/datasets.txt --max-samples 5000
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Batch processing: {datasets_file}")

    success = process_batch_datasets(
        datasets_file=datasets_file,
        data_root=data_root,
        max_samples=max_samples,
        distance_metric=metric,
    )

    exit_code = 0 if success else 1
    logger.info(f"Batch processing complete. Exit code: {exit_code}")
    sys.exit(exit_code)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--max-samples", default=2000, help="Maximum number of samples for large datasets")
@click.option(
    "--metric",
    default="l1",
    type=click.Choice(["l1", "l2", "euclidean", "manhattan", "cosine"]),
    help="Distance metric for analysis",
)
def single(dataset_path, max_samples, metric):
    """
    Process a single dataset file.

    DATASET_PATH: Path to the dataset file to process

    Example:
        python main.py single ../data/optimize/config/SS-A.csv --metric euclidean
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Single dataset processing: {dataset_path}")

    success = process_single_dataset(dataset_path=dataset_path, max_samples=max_samples, distance_metric=metric)

    exit_code = 0 if success else 1
    logger.info(f"Single processing complete. Exit code: {exit_code}")
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
