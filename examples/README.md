# DRR Examples

This folder contains examples demonstrating how to use the DRR (Dimensionality Reduction Ratio) package for intrinsic dimensionality analysis.

## Files

### `simple_example.py` 
The simplest way to get started with the DRR package. Shows basic usage of the convenience function.

**Run with:**
```bash
python simple_example.py
```

### `example_usage.py`
Comprehensive examples showing all features of the DRR package including:

- Single dataset processing
- Custom parameters
- Batch processing 
- Error handling
- Distance metric comparison

**Run with:**
```bash
python example_usage.py
```

## Prerequisites

Make sure you have installed the DRR package:

```bash
# From the project root directory
pip install -e .
```

Or if you have the package installed from PyPI:
```bash
pip install drr
```

## Basic Usage Patterns

### Quick Analysis
```python
from drr import estimate_intrinsic_dimension
import numpy as np

# Your data
data = np.random.rand(1000, 10)

# Estimate intrinsic dimension
original_dims, intrinsic_dim, drr = estimate_intrinsic_dimension(data)
print(f"DRR: {drr:.3f} ({drr:.1%} reduction potential)")
```

### Advanced Usage
```python
from drr import IntrinsicDimensionEstimator, DataProcessor

# Initialize components
processor = DataProcessor()
estimator = IntrinsicDimensionEstimator(
    max_samples=5000,
    distance_metric='euclidean'
)

# Process CSV file
data, metadata = processor.process_dataset('your_file.csv')

# Estimate intrinsic dimension
r, i, drr = estimator.estimate(data)
```

### Batch Processing
```python
from drr import BatchProcessor

# Initialize batch processor
batch = BatchProcessor(
    results_file="results.csv",
    max_samples=2000
)

# Process multiple datasets
results = batch.process_datasets_from_file("datasets.txt")
```

## Expected Output

When you run the examples, you should see:

1. **Logging information** showing the processing steps
2. **Results** with original dimensions (R), intrinsic dimensions (I), and DRR values
3. **Interpretation** of the dimensionality reduction potential

## Troubleshooting

### Import Errors
If you see import errors in your IDE but the code runs correctly, this is normal. The package is properly installed and functional.

### Path Issues
The examples assume certain data files exist in the `../data/` directory. If you don't have these files, some examples may show "file not found" messages, but this won't affect the core functionality demonstration.

### Performance
The intrinsic dimension estimation can take time for large datasets. The examples use reasonable sample sizes (2000-5000 samples) for demonstration purposes.

## Next Steps

1. Try the examples with your own datasets
2. Experiment with different distance metrics (`l1`, `l2`, `euclidean`, `manhattan`, `cosine`)
3. Adjust the `max_samples` parameter based on your dataset size and performance needs
4. Check the comprehensive logging output to understand the processing steps
