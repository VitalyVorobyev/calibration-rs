# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Manual parameter initialization workflow** (M10, ADR 0011): expert alternative to auto-initialization for all six problem types.
  - `step_set_init(session, XxxManualInit, opts)` per problem type — seeds provided fields, auto-initializes the rest.
  - `step_init` now delegates to `step_set_init` with all-`None` defaults; `step_optimize` is unchanged.
  - `ManualInit` structs: `PlanarManualInit`, `ScheimpflugManualInit`, `IntrinsicsManualInit`, `HandeyeManualInit`, `PerCamIntrinsicsManualInit`, `RigExtrinsicsManualInit`, `LaserlineManualInit`.
  - Partial init: any subset of parameter groups can be seeded; unspecified groups are auto-initialized.
  - Intrinsics-pose coupling preserved: when manual intrinsics but auto poses, poses are recovered using the manual intrinsics.
  - All `ManualInit` types derive `serde::Serialize/Deserialize` for Python interop.
- **Facade re-exports**: all `ManualInit` types and `step_set_*` functions exposed through `vision-calibration/src/lib.rs`.
- **Integration test** `manual_init_integration.rs`: exact seeds → reproj < 1e-3 px; perturbed seeds (~10%) → convergence; default equivalence regression guard.
- **Python bindings** for manual init (PlanarIntrinsics and SingleCamHandeye):
  - `PlanarManualInit` and `SingleCamHandeyeIntrinsicsManualInit` dataclasses in `vision_calibration.models`.
  - `run_planar_intrinsics_with_init` and `run_single_cam_handeye_with_init` in `vision_calibration`.

## [0.2.0] - 2026-03-07

### Added
- New high-level Scheimpflug intrinsics workflow:
  - Rust API: `vision_calibration::scheimpflug_intrinsics::run_calibration`
  - Python API: `vision_calibration.run_scheimpflug_intrinsics`
- Typed Python camera/result payload models for high-level workflows (planar, hand-eye, rig, laserline, Scheimpflug)
- Synthetic integration tests for Scheimpflug calibration in `vision-calibration`
- Facade API compile-surface integration tests to catch accidental public API regressions
- Session schema metadata validation tests for JSON session compatibility checks

### Changed
- Enforced workspace layering by moving Scheimpflug solver implementation to `vision-calibration-pipeline` and keeping `vision-calibration` as facade re-export
- Hardened facade/API contracts:
  - `#[non_exhaustive]` added to public config/export/error structs and enums
  - `vision-calibration` now enforces `#![deny(missing_docs)]`
  - Session import now rejects schema metadata mismatches with explicit errors
- Expanded rustdoc/book/readme coverage with updated workflow usage snippets
- CI and release hardening:
  - `cargo clippy` now runs with `--all-features`
  - `cargo test` now runs with `--all-features`
  - Python extension build + runtime tests are part of CI/release checks
- Python high-level bindings are now typed-first end-to-end:
  - top-level `run_*` APIs require typed dataset/config dataclasses
  - dict/list compatibility paths moved to explicit low-level raw helpers in `vision_calibration._api`
  - `vision_calibration.types` reduced to low-level compatibility surface

### Breaking
- Minor release bump to `0.2.0` due public API contract hardening and Python binding migration
- Python high-level runners are typed-only and no longer accept raw mapping/list payloads
- Dict-based high-level result access patterns (`camera`/`cameras`/`estimate`/`stats`/`raw` as mappings) were replaced by typed model fields
- Several low-level type aliases are no longer re-exported at the top-level Python package; import from `vision_calibration.types` only when using low-level compatibility APIs

## [0.1.2]

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
