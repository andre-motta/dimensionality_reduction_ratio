# DRR: Dimensionality Reduction Ratio Toolkit

[![CI](https://github.com/andre-motta/dimensionality_reduction_ratio/workflows/CI/badge.svg)](https://github.com/andre-motta/dimensionality_reduction_ratio/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Unlicense](https://img.shields.io/badge/License-Unlicense-blue.svg)](http://unlicense.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage: 82%](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](https://github.com/andre-motta/dimensionality_reduction_ratio/actions)

> **"Less Noise, More Signal: DRR for Better Optimizations of SE Tasks"**  
> *A research-backed approach to predicting when lightweight algorithms suffice*

A professional Python toolkit for estimating intrinsic dimensionality and computing **Dimensionality Reduction Ratio (DRR)** metrics. This implementation is based on cutting-edge research from NC State University showing that **DRR can predict when simple algorithms outperform complex AI methods by orders of magnitude**.

## 🎯 Research Background

This toolkit implements the methodology from our research paper ["Less Noise, More Signal: DRR for Better Optimizations of SE Tasks"](https://arxiv.org/abs/2503.21086) which demonstrates that:

- **89% of Software Engineering datasets** satisfy the DRR threshold for simplified optimization
- **Simple methods can be 100x faster** than state-of-the-art optimizers when DRR > 1/3
- **SE data has lower intrinsic complexity** (median 3.1 dimensions) compared to general ML data (median 5 dimensions)

## 🔬 What is DRR?

The **Dimensionality Reduction Ratio (DRR)** is a metric that quantifies how much dimensionality reduction is possible in a dataset:

```
DRR = 1 - (I/R)
```

Where:
- **I** = Intrinsic dimension (estimated using correlation function analysis)
- **R** = Raw dimension (number of original features)

### Research Finding: The 1/3 Threshold

Our research shows that when **DRR > 1/3**, simple algorithms can achieve the same performance as complex state-of-the-art optimizers but run **two orders of magnitude faster**.

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

### Fractal-Based Intrinsic Dimension Estimation

This toolkit implements the **correlation function method** used in our research, which leverages fractal geometry concepts to estimate intrinsic dimensionality:

1. **Distance Analysis**: Calculate pairwise distances between data points
2. **Correlation Function**: For radius R, compute `C(r) = (2 * I) / (n * (n-1))` where I is the number of pairs with distance ≤ r
3. **Fractal Dimension**: Analyze how the number of points scales with distance radius
4. **Gradient Analysis**: The maximum gradient of log(C(r)) vs log(r) approximates the intrinsic dimension

### Key Advantages

- **Accuracy**: More precise than traditional PCA-based methods
- **Robustness**: Less sensitive to noise and outliers  
- **Scalability**: Efficient for large datasets through intelligent sampling
- **Research-Validated**: Proven effective on 24+ real-world datasets

### The Research Breakthrough

Our methodology **fixes critical errors** in previous approaches:
- Previous work suggested simple algorithms when I < 4
- Our research found **many counter-examples** to this threshold
- **New threshold: DRR > 1/3** provides much more accurate predictions

## 📊 DRR Metrics & Research Insights

### Understanding DRR Values

**DRR = 1 - (I/R)** where:
- **I**: Intrinsic dimension (estimated)
- **R**: Raw dimension (number of features)
- **DRR**: Dimensionality Reduction Ratio

### Research-Based Interpretation Guidelines

| DRR Range | Algorithm Recommendation | Performance Insight | SE Data Examples |
|-----------|--------------------------|---------------------|------------------|
| **> 0.67** | Use simple methods | 100x faster, same quality | Software configuration (SS-B, SS-D) |
| **0.33 - 0.67** | Simple methods often sufficient | 10-50x speedup possible | Most SE optimization tasks |
| **< 0.33** | Complex methods may be needed | Intrinsic complexity requires sophisticated algorithms | General ML datasets |

### Key Research Findings

📈 **SE vs General ML Data**:
- **Median SE intrinsic dimensionality**: 3.1 dimensions
- **Median general ML intrinsic dimensionality**: 5.0 dimensions  
- **Conclusion**: SE problems are inherently less complex

🚀 **Performance Implications**:
- **89% of SE datasets** satisfy DRR > 1/3 threshold
- **Simple algorithms** (30 samples) perform as well as complex ones (3000 samples)
- **Speedup**: 2 orders of magnitude (seconds vs 20 minutes)

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

## � Citation

If you use this toolkit in your research, please cite our paper:

```bibtex
@article{lustosa2025drr,
  title={Less Noise, More Signal: DRR for Better Optimizations of SE Tasks},
  author={Andre Lustosa and Tim Menzies},
  journal={arXiv preprint arXiv:2503.21086},
  year={2025},
  url={https://arxiv.org/abs/2503.21086}
}
```

## �🔗 Research Links

- **📄 Paper**: [arXiv:2503.21086](https://arxiv.org/abs/2503.21086)
- **💻 Research Code**: [GitHub Repository](https://github.com/andre-motta/dimensionality_reduction_ratio)
- **🏛️ Institution**: North Carolina State University, Department of Computer Science
- **👥 Authors**: Andre Lustosa, Tim Menzies (Fellow, IEEE)

## 🔗 Repository

**GitHub Repository**: https://github.com/andre-motta/dimensionality_reduction_ratio

For questions, issues, or contributions, please visit the repository or contact the maintainers.