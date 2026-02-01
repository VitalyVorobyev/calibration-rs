# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Iterative intrinsics estimation** in `vision-calibration-linear`: New `IterativeIntrinsicsSolver` for jointly estimating camera intrinsics (K) and Brown-Conrady distortion without requiring ground truth distortion preprocessing
- **Distortion fitting** in `vision-calibration-linear`: New `DistortionSolver` for closed-form estimation of radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients from homography residuals
- **Shared test utilities** in `vision-calibration-core`: New `test_utils` module providing common calibration test data structures (`CalibrationView`, `ViewDetections`, `CornerInfo`) and helper functions
- **Realistic calibration tests**: New integration tests demonstrating full calibration pipeline without ground truth distortion (in `stereo_linear.rs` and `planar_intrinsics_real_data.rs`)
- Comprehensive documentation for all new modules with usage examples and algorithm descriptions
- Linear solver additions in `vision-calibration-linear`: camera matrix DLT + RQ decomposition, linear triangulation, 7-point fundamental, 5-point essential + decomposition, P3P, and EPnP
- New `vision-calibration-linear` README with algorithm overview and usage notes

### Changed
- Updated `vision-calibration-linear` lib.rs to export new `distortion_fit` and `iterative_intrinsics` modules
- Refactored test files to use shared utilities from `vision-calibration-core::test_utils`, eliminating code duplication
- Updated CLAUDE.md with detailed documentation of iterative intrinsics feature and typical workflow
- Expanded rustdoc across `vision-calibration-linear` algorithms and updated top-level README to reflect new solver coverage

### Deprecated

### Removed

### Fixed

### Security
