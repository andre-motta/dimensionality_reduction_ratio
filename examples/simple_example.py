#!/usr/bin/env python3
"""
Simple Example - DRR Package Usage

This is the simplest way to use the DRR package for estimating
intrinsic dimensionality.
"""

import numpy as np

# Import the convenience function
from drr import estimate_intrinsic_dimension

def main():
    """Simple example showing basic usage."""
    print("DRR Package - Simple Example")
    print("=" * 40)
    
    # Create some sample data (3D data with intrinsic 1D structure)
    print("Creating sample data...")
    t = np.linspace(0, 4*np.pi, 500)
    data = np.column_stack([
        np.cos(t) + 0.1 * np.random.randn(500),
        np.sin(t) + 0.1 * np.random.randn(500), 
        t + 0.1 * np.random.randn(500)
    ])
    
    print(f"Sample data shape: {data.shape}")
    
    # Estimate intrinsic dimension
    print("\nEstimating intrinsic dimension...")
    original_dims, intrinsic_dim, drr = estimate_intrinsic_dimension(data)
    
    # Display results
    print(f"\nResults:")
    print(f"  Original Dimensions: {original_dims}")
    print(f"  Intrinsic Dimension: {intrinsic_dim}")
    print(f"  DRR: {drr:.3f} ({drr:.1%} reduction potential)")
    
    # Test with different parameters
    print(f"\nTesting with different distance metric...")
    r2, i2, drr2 = estimate_intrinsic_dimension(
        data, 
        distance_metric='euclidean',
        max_samples=1000
    )
    print(f"  Euclidean metric: R={r2}, I={i2}, DRR={drr2:.3f}")

if __name__ == '__main__':
    main()
