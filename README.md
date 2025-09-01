# Intrinsic Dimension Analysis with DRR Metrics

[![CI](https://github.com/USER/REPO/workflows/CI/badge.svg)](https://github.com/USER/REPO/actions)
[![codecov](https://codecov.io/gh/USER/REPO/branch/main/graph/badge.svg)](https://codecov.io/gh/USER/REPO)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A professional Python toolkit for estimating the intrinsic dimensionality of datasets and computing Dimensionality Reduction Ratio (DRR) metrics. This implementation is based on the correlation function approach from Levina & Bickel (2005) with enhancements for large-scale dataset processing.

## 🚀 Quick Start

```bash
# Install the package
pip install drr

# Process all datasets from configuration file
drr batch datasets.txt

# Process a single dataset
drr single data/config/Apache_AllMeasurements.csv

# Use custom parameters with debug logging
drr --log-level DEBUG batch datasets.txt --max-samples 5000 --metric euclidean
```

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Configuration](#dataset-configuration)
- [Algorithm Details](#algorithm-details)
- [DRR Metrics](#drr-metrics)
- [API Reference](#api-reference)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This toolkit implements the **Levina-Bickel correlation function method** for intrinsic dimension estimation, enhanced with:

- **DRR (Dimensionality Reduction Ratio)** metric: `DRR = 1 - (I/R)`
- **Large dataset handling** with intelligent sampling strategies
- **Batch processing** capabilities for multiple datasets
- **Professional logging** and error handling
- **Resume functionality** for interrupted processing jobs

### What is Intrinsic Dimension?

The **intrinsic dimension** of a dataset is the minimum number of parameters needed to represent the data without significant information loss. While a dataset might exist in a high-dimensional space (raw dimension R), its true complexity might be much lower (intrinsic dimension I).

### What is DRR?

**Dimensionality Reduction Ratio (DRR)** quantifies how much dimensionality reduction is possible:
- `DRR = 1 - (I/R)`
- **High DRR (>0.5)**: Significant dimensionality reduction possible
- **Low DRR (<0.3)**: Dataset complexity is close to its raw dimensionality

## ✨ Features

### Core Capabilities
- 🔬 **Intrinsic dimension estimation** using correlation function analysis
- 📊 **DRR metric computation** for dataset complexity analysis
- 🗂️ **Batch processing** of multiple datasets from configuration files
- 📈 **Large dataset optimization** with multi-level sampling
- 🔧 **Resume functionality** for interrupted processing jobs

### Technical Features
- 🏗️ **Professional architecture** with modular design
- 📝 **Comprehensive logging** with configurable levels
- 🛡️ **Robust error handling** and validation
- 🔄 **Progress tracking** and status reporting
- 📊 **CSV results export** with detailed metrics

### Data Processing
- 🧹 **Automatic preprocessing** (categorical encoding, missing value handling)
- 🎯 **Goal variable detection** and removal
- 📏 **Distance metric selection** (L1, L2, Euclidean, Manhattan, Cosine)
- 🔀 **Intelligent sampling** for datasets >50K rows

## 🛠️ Installation

## 🛠️ Installation

### From PyPI (Recommended)
```bash
# Install the latest stable version
pip install drr

# Install with development dependencies
pip install drr[dev]

# Install with all optional dependencies
pip install drr[all]
```

### From Source
```bash
# Clone the repository
git clone https://github.com/andre-motta/dimensionality_reduction_ratio.git
cd dimensionality_reduction_ratio

# Install in development mode
pip install -e .

# Or install with development dependencies
pip install -e .[dev]
```

### Prerequisites
- Python 3.11+
- pip (Python package installer)

### Verify Installation
```bash
# Test the command-line interface
drr --help

# Or if installed from source
cd src
python -m drr --help
```

### Dependencies
This project uses the following key libraries:
- **Click**: Modern command-line interface framework
- **NumPy**: Numerical computing library
- **Pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing library
- **Matplotlib**: Plotting library

## 📖 Usage

### Command Line Interface

#### Batch Processing
Process multiple datasets from a configuration file:
```bash
drr batch datasets.txt
```

With custom parameters:
```bash
drr --log-level DEBUG batch datasets.txt \
    --max-samples 5000 \
    --metric euclidean \
    --data-root data
```

#### Single Dataset Processing
Process an individual dataset:
```bash
drr single data/config/Apache_AllMeasurements.csv
```

With custom parameters:
```bash
drr single data/config/Apache_AllMeasurements.csv \
    --max-samples 3000 \
    --metric manhattan
```

#### Global Options
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--log-file`: Optional log file path

#### Batch Command Options
- `datasets_file`: Path to configuration file listing datasets to process
- `--data-root`: Root directory for dataset files (default: `../data`)
- `--max-samples`: Maximum samples for large datasets (default: 2000)
- `--metric`: Distance metric (`l1`, `l2`, `euclidean`, `manhattan`, `cosine`)

#### Single Command Options  
- `dataset_path`: Path to the dataset file to process
- `--max-samples`: Maximum samples for large datasets (default: 2000)
- `--metric`: Distance metric (`l1`, `l2`, `euclidean`, `manhattan`, `cosine`)

### Python API

#### Single Dataset Analysis
```python
import drr

# Simple usage with convenience function
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Your dataset
original_dims, intrinsic_dim, drr_value = drr.estimate_intrinsic_dimension(data)

print(f"Raw dimensions: {original_dims}")
print(f"Intrinsic dimension: {intrinsic_dim}")
print(f"DRR: {drr_value:.3f}")

# Advanced usage with classes
estimator = drr.IntrinsicDimensionEstimator(max_samples=2000, distance_metric='euclidean')
processor = drr.DataProcessor()

# Process dataset from file
data, metadata = processor.process_dataset('data/config/Apache_AllMeasurements.csv')
original_dims, intrinsic_dim, drr_value = estimator.estimate(data)
```

#### Batch Processing
```python
import drr

# Initialize batch processor
processor = drr.BatchProcessor(
    results_file="results/my_results.csv",
    max_samples=2000,
    distance_metric='manhattan'
)

# Process all datasets
results = processor.process_datasets_from_file('datasets.txt')
print(f"Processed {results['successful']} datasets successfully")
```

## 📁 Dataset Configuration

The `datasets.txt` file defines which datasets to process using a hierarchical structure:

### Format
```
# Configuration section
config
    Apache_AllMeasurements
    HSMGP_num
    SQL_AllMeasurements

# Classification datasets  
classify
    breastcancer
    diabetes
    german

# Software measurement datasets
mvn
    training_set/mvn_training
    test_set/mvn_test
```

### Rules
1. **Section headers** have no indentation
2. **Dataset names** are indented (spaces or tabs)
3. **Comments** start with `#`
4. **File paths** are relative to `data_root` directory
5. **CSV extension** is automatically added

## 🔬 Algorithm Details

### Correlation Function Method

The algorithm estimates intrinsic dimension using the correlation function approach:

1. **Distance Computation**: Calculate pairwise distances between data points
2. **Correlation Function**: `C(r) = (2 * I) / (n * (n-1))` where I is the number of pairs with distance ≤ r
3. **Log-Log Analysis**: Fit linear regression to `log(C(r))` vs `log(r)`
4. **Dimension Estimation**: The slope approximates the intrinsic dimension

## 📊 DRR Metrics

### Understanding DRR Values

**DRR = 1 - (I/R)** where:
- **I**: Intrinsic dimension (estimated)
- **R**: Raw dimension (number of features)
- **DRR**: Dimensionality Reduction Ratio

### Interpretation Guidelines

| DRR Range | Interpretation | Example Dataset Type |
|-----------|----------------|---------------------|
| **0.0 - 0.2** | Low reduction potential | Behavior/performance data |
| **0.2 - 0.4** | Moderate reduction | Mixed datasets |
| **0.4 - 0.6** | Good reduction potential | Configuration data |
| **0.6 - 1.0** | High reduction potential | Highly correlated features |

## 📈 Results

### Sample Output

```
===============================================
RESULTS FOR: Apache_AllMeasurements.csv
===============================================
Original Dimensions (R): 43
Intrinsic Dimension (I): 12
DRR (1 - I/R): 0.721
Data Quality: 72.1% dimensionality reduction
===============================================
```

## 🗂️ Directory Structure

```
dimensionality_reduction_ratio/
├── src/                      # Source code modules
│   ├── main.py              # Command-line entry point
│   ├── intrinsic_dimension.py  # Core algorithm
│   ├── data_processor.py    # Data preprocessing
│   └── batch_processor.py   # Batch processing
├── config/                   # Configuration files
│   ├── datasets.txt         # Dataset configuration
│   └── test_datasets.txt    # Test configuration
├── data/                     # Dataset files
├── results/                  # Output files
├── logs/                     # Log files
├── examples/                 # Usage examples
│   └── example_usage.py     # API usage examples
└── README.md                # This documentation
```

## 🧪 Testing

### Validate Installation
```bash
# Test the command-line interface
drr --help
drr batch --help 
drr single --help

# Test with sample data
drr single data/optimize/config/SS-A.csv

# Test batch processing (small subset)
drr batch config/test_dataset.txt
```

---

## 🔗 Repository

**GitHub Repository**: https://github.com/andre-motta/dimensionality_reduction_ratio

For questions or support, please open an issue in the repository or contact the maintainers.