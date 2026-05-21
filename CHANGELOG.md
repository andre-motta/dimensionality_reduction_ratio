# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-05-21

### Fixed
- Distance metric bug — only L1 worked, all others silently fell back to euclidean
- Scipy metric mapping for scipy 1.17+ (manhattan → cityblock)
- Test fixture parameter mismatches hidden by error-swallowing patterns
- Division by zero risk when dataset has no features
- All-NaN column handling in data preprocessing
- Global random seed replaced with scoped RNG
- Path traversal validation in batch config parsing

### Changed
- Relaxed numpy upper bound for NumPy 2.x compatibility
- Extracted magic numbers to class constants

## [1.0.1] - 2025-05-21

### Changed
- Simplified CSV output format for results

## [1.0.0] - 2025-05-01

### Added
- Initial release of DRR toolkit
- Intrinsic dimension estimation via correlation function method
- DRR metric computation
- Batch processing with config files and resume support
- CLI interface with `drr` and `drr-toolkit` commands
- Support for L1, L2, Euclidean, Manhattan, and Cosine distance metrics
- Large dataset handling with intelligent sampling
- Comprehensive test suite (102 tests)
- GitHub Actions CI/CD with linting, security scanning, and PyPI publishing
