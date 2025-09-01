"""
Batch Processing Module

This module handles batch processing of multiple datasets using a configuration file,
with support for resuming interrupted jobs and comprehensive logging.
"""

import csv
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .data_processor import DataProcessor
from .intrinsic_dimension_estimator import IntrinsicDimensionEstimator

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Processes multiple datasets in batch using configuration files.

    This processor:
    1. Reads dataset configurations from text files
    2. Processes datasets individually with error handling
    3. Saves results incrementally to CSV
    4. Supports resuming interrupted processing
    5. Logs errors and progress comprehensively

    Attributes:
        data_processor (DataProcessor): Data preprocessing handler
        estimator (IntrinsicDimensionEstimator): Intrinsic dimension estimator
        results_file (str): Path to results CSV file
        error_log_file (str): Path to error log file
    """

    def __init__(
        self,
        results_file: str = "results/dataset_results.csv",
        error_log_file: str = "logs/batch_errors.log",
        max_samples: int = 2000,
        distance_metric: str = "l1",
    ):
        """
        Initialize the batch processor.

        Args:
            results_file: Path to save results CSV
            error_log_file: Path to save error logs
            max_samples: Maximum samples for large datasets
            distance_metric: Distance metric for estimation
        """
        self.data_processor = DataProcessor()
        self.estimator = IntrinsicDimensionEstimator(max_samples=max_samples, distance_metric=distance_metric)

        self.results_file = results_file
        self.error_log_file = error_log_file

        # Ensure directories exist
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        os.makedirs(os.path.dirname(error_log_file), exist_ok=True)

        logger.info(f"Initialized BatchProcessor: results={results_file}, " f"errors={error_log_file}")

    def process_datasets_from_file(self, datasets_file: str, data_root: str = "data") -> Dict[str, Any]:
        """
        Process all datasets listed in a configuration file.

        Args:
            datasets_file: Path to file containing dataset configurations
            data_root: Root directory for dataset files

        Returns:
            Dictionary with processing statistics and results
        """
        logger.info(f"Starting batch processing from {datasets_file}")

        # Parse dataset configurations
        dataset_configs = self._parse_datasets_file(datasets_file)
        total_datasets = len(dataset_configs)

        logger.info(f"Found {total_datasets} datasets to process")

        # Check for existing results to resume processing
        already_processed = self._get_processed_datasets()
        remaining_datasets = [config for config in dataset_configs if config["name"] not in already_processed]

        logger.info(f"Already processed: {len(already_processed)} datasets")
        logger.info(f"Remaining to process: {len(remaining_datasets)} datasets")

        if len(remaining_datasets) == 0:
            logger.info("All datasets already processed!")
            return {
                "total_datasets": total_datasets,
                "already_processed": len(already_processed),
                "newly_processed": 0,
                "successful": 0,
                "failed": 0,
                "results_file": self.results_file,
            }

        # Initialize results file if it doesn't exist
        self._initialize_results_file()

        # Process remaining datasets
        successful = 0
        failed = 0

        for i, config in enumerate(remaining_datasets, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset {i}/{len(remaining_datasets)}: {config['name']}")
            logger.info(f"Path: {config['path']}")
            logger.info(f"{'='*60}")

            try:
                success = self._process_single_dataset(config, data_root)
                if success:
                    successful += 1
                    logger.info(f"✓ Successfully processed: {config['name']}")
                else:
                    failed += 1
                    logger.error(f"✗ Failed to process: {config['name']}")

            except Exception as e:
                failed += 1
                error_msg = f"Unexpected error processing {config['name']}: {str(e)}"
                logger.error(error_msg)
                self._log_error(config["name"], error_msg, traceback.format_exc())

        # Final summary
        logger.info(f"\n{'='*60}")
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total datasets: {total_datasets}")
        logger.info(f"Already processed: {len(already_processed)}")
        logger.info(f"Newly processed: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Results saved to: {self.results_file}")
        if failed > 0:
            logger.info(f"Errors logged to: {self.error_log_file}")
        logger.info(f"{'='*60}")

        return {
            "total_datasets": total_datasets,
            "already_processed": len(already_processed),
            "newly_processed": successful,
            "successful": successful,
            "failed": failed,
            "results_file": self.results_file,
        }

    def _parse_datasets_file(self, datasets_file: str) -> List[Dict[str, str]]:
        """Parse the datasets configuration file."""
        logger.info(f"Parsing dataset configuration from {datasets_file}")

        if not os.path.exists(datasets_file):
            raise FileNotFoundError(f"Datasets file not found: {datasets_file}")

        datasets = []
        current_section = None
        current_subsection = None

        with open(datasets_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                original_line = line.rstrip("\n\r")  # Remove only newlines, keep spaces
                line = original_line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Check indentation to determine hierarchy
                indent_level = len(original_line) - len(original_line.lstrip())

                if indent_level == 0:
                    # Top-level section (e.g., "optimize")
                    current_section = line
                    current_subsection = None
                    logger.debug(f"Found root section: {current_section}")
                elif indent_level == 4:  # First level of indentation (4 spaces)
                    # Subsection (e.g., "behavior_data", "config")
                    current_subsection = line
                    logger.debug(f"Found subsection: {current_subsection}")
                elif indent_level == 8 and current_section and current_subsection:  # Second level (8 spaces)
                    # Actual dataset entry
                    dataset_name = line
                    dataset_path = f"{current_section}/{current_subsection}/{dataset_name}"

                    datasets.append(
                        {
                            "name": dataset_name,
                            "path": dataset_path,
                            "section": current_section,
                            "subsection": current_subsection,
                            "line_number": line_num,
                        }
                    )

                    logger.debug(f"Added dataset: {dataset_name} -> {dataset_path}")

        logger.info(f"Parsed {len(datasets)} datasets from {datasets_file}")
        return datasets

    def _get_processed_datasets(self) -> List[str]:
        """Get list of datasets that have already been processed."""
        if not os.path.exists(self.results_file):
            return []

        try:
            df = pd.read_csv(self.results_file)
            processed = df["Dataset"].tolist()
            logger.debug(f"Found {len(processed)} already processed datasets")
            return processed
        except Exception as e:
            logger.warning(f"Error reading existing results file: {e}")
            return []

    def _initialize_results_file(self):
        """Initialize the results CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.results_file):
            headers = ["Dataset", "R", "I", "DRR"]

            with open(self.results_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

            logger.info(f"Initialized results file: {self.results_file}")

    def _process_single_dataset(self, config: Dict[str, str], data_root: str) -> bool:
        """Process a single dataset and save results."""
        dataset_name = config["name"]
        dataset_path = config["path"]

        try:
            # Construct full file path
            csv_file = os.path.join(data_root, f"{dataset_path}.csv")

            if not os.path.exists(csv_file):
                error_msg = f"Dataset file not found: {csv_file}"
                logger.error(error_msg)
                self._log_error(dataset_name, error_msg)
                return False

            logger.info(f"Processing file: {csv_file}")

            # Process the dataset
            processed_data, metadata = self.data_processor.process_dataset(csv_file)

            if not self.data_processor.validate_processed_data(processed_data):
                error_msg = "Data validation failed"
                logger.error(error_msg)
                self._log_error(dataset_name, error_msg)
                return False

            # Estimate intrinsic dimension
            original_dims, intrinsic_dim, drr = self.estimator.estimate(processed_data)

            logger.info(f"Results: R={original_dims}, I={intrinsic_dim}, DRR={drr:.3f}")

            # Save results to CSV
            self._save_result_to_csv(dataset_name, original_dims, intrinsic_dim, drr)

            return True

        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg)
            self._log_error(dataset_name, error_msg, traceback.format_exc())
            return False

    def _save_result_to_csv(self, dataset_name: str, original_dims: int, intrinsic_dim: int, drr: float):
        """Save processing result to CSV file."""
        result_row = [dataset_name, original_dims, intrinsic_dim, f"{drr:.3f}"]

        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(result_row)

        logger.debug(f"Saved result for {dataset_name} to {self.results_file}")

    def _log_error(self, dataset_name: str, error_message: str, traceback_str: Optional[str] = None):
        """Log error to error log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.error_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Error: {error_message}\n")
            if traceback_str:
                f.write(f"Traceback:\n{traceback_str}\n")
            f.write(f"{'='*60}\n")

        logger.debug(f"Logged error for {dataset_name} to {self.error_log_file}")
