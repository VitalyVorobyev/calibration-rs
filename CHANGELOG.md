# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (breaking, pre-1.0)
- **Rig family sensor-axis refactor (ADR 0013).** The pinhole and Scheimpflug
  rig modules collapse into a single workflow per problem family. Five rig
  sibling modules become three (~−2,300 LoC across PRs #36, #37, #38).
  - `vision_calibration_pipeline::rig_scheimpflug_extrinsics` — module
    deleted. Migrate to `rig_extrinsics::RigExtrinsicsProblem` with
    `RigExtrinsicsConfig::sensor = SensorMode::Scheimpflug { … }`.
  - `vision_calibration_pipeline::rig_scheimpflug_handeye` — module
    deleted. Migrate to `rig_handeye::RigHandeyeProblem` with
    `RigHandeyeConfig::sensor = SensorMode::Scheimpflug { … }`.
  - `RigExtrinsicsProblem::Output` is now `RigExtrinsicsOutput::{Pinhole,
    Scheimpflug}`; `RigHandeyeProblem::Output` is now
    `RigHandeyeOutput::{Pinhole, Scheimpflug}`. Use accessor methods
    (`cam_to_rig()`, `cameras()`, `sensors()`, `mean_reproj_error()`, …)
    instead of `output.params.*`.
  - `RigExtrinsicsExport` and `RigHandeyeExport` gain
    `sensors: Option<Vec<ScheimpflugParams>>` (`None` for pinhole, `Some(_)`
    for Scheimpflug). `RigExtrinsicsExport` also gains
    `rig_se3_target: Vec<Iso3>` (always populated; pinhole exports
    previously omitted it).
  - `RigIntrinsicsManualInit` (rig_extrinsics) and
    `RigHandeyeIntrinsicsManualInit` gain
    `per_cam_sensors: Option<Vec<ScheimpflugParams>>` for Scheimpflug
    seeds.
  - `RigHandeyeBaConfig` gains `refine_scheimpflug_in_handeye_ba: bool`.
  - `SensorMode` lives in `crate::rig_family::SensorMode` and is
    re-exported by both `rig_extrinsics::SensorMode` and
    `rig_handeye::SensorMode` (single source of truth).
  - `RigHandeyeExport::to_upstream_calibration` (used by
    `rig_laserline_device`) now returns `Result<…>` and errors on pinhole
    rigs.
  - Facade `pixel_to_gripper_point` accepts `&RigHandeyeExport` and errors
    when sensors are absent.
  - Python: `run_rig_scheimpflug_extrinsics`, `run_rig_scheimpflug_handeye`,
    and their dataclasses are removed; `RigHandeyeResult` gains a
    `sensors: list[ScheimpflugSensor] | None` field plus a `to_payload()`
    method that round-trips the unified export. The Python
    `pixel_to_gripper_point` accepts `RigHandeyeResult`.
  - A handful of advanced Scheimpflug-only intrinsics knobs
    (`initial_cameras`, `initial_sensors`, `fallback_to_shared_init`,
    `fix_intrinsics_when_overridden`, `fix_intrinsics_in_percam_ba`,
    `fix_distortion_in_percam_ba`) are dropped. The most load-bearing
    default — `DistortionFixMask::radial_only()` for Scheimpflug
    per-camera intrinsics refinement — is preserved as a hard-coded
    constant.

### Added
- **ADR 0013** ([`docs/adrs/0013-rig-family-sensor-axis-refactor.md`](docs/adrs/0013-rig-family-sensor-axis-refactor.md))
  records the rig family sensor-axis refactor decision: composition over
  traits, single-axis collapse, alternatives considered.

## [0.4.0] - 2026-04-29

### Added
- Scheimpflug rig calibration family:
  - `optim::optimize_rig_extrinsics_scheimpflug` + `RigExtrinsicsScheimpflugParams/SolveOptions/Estimate`.
  - `optim::optimize_handeye_scheimpflug` + `HandEyeScheimpflugParams/SolveOptions/Estimate` (EyeInHand).
  - `optim::optimize_rig_laserline` + `RigLaserlineDataset/Upstream/SolveOptions/Estimate` — per-camera
    laser-plane calibration against a frozen upstream rig calibration, with plane output expressed in
    rig frame.
  - `LaserPlane::transform_by(&Iso3)` utility for frame-to-frame plane transforms.
- Three new session-API pipelines in `vision_calibration_pipeline`:
  - `rig_scheimpflug_extrinsics` (4 steps)
  - `rig_scheimpflug_handeye` (6 steps, EyeInHand)
  - `rig_laserline_device` (2 steps, consumes a frozen `RigScheimpflugHandeyeExport`)
- Facade helper `vision_calibration::pixel_to_gripper_point(cam_idx, pixel, rig_cal, laser_planes_rig)`
  maps a laser pixel in any camera to a 3D point in the robot gripper frame by chaining undistort →
  rig-frame ray → plane intersection → hand-eye transform.
- New IR factor kinds `ReprojPointPinhole4Dist5Scheimpflug2{TwoSE3,HandEye,HandEyeRobotDelta}` with
  matching TinySolver adapters and autodiff-ready residual generics.
- New private example crate `vision-calibration-examples-private` (publish = false) with
  `examples/puzzle_130x130_rig.rs` running the full pipeline on a sensor dataset.

## [0.3.0] - 2026-04-12

### Added
- Typed `Error` enum (using `thiserror`) exposed from every workspace crate:
  `vision-calibration-core`, `vision-calibration-linear`, `vision-calibration-optim`,
  and `vision-calibration-pipeline` now return structured, matchable error variants instead of `anyhow::Error`.
- `# Errors` sections added across fallible public APIs in `vision-calibration-linear`,
  `vision-calibration-optim`, and `vision-calibration-pipeline` rustdoc.
- Optional `tracing` feature on `vision-calibration-core` instruments `ransac_fit` with spans
  (off by default, no runtime cost when disabled).
- Boundary validation for Python inputs with `PyValueError` (high-level `run_*` APIs).
- CI job enforcing the declared MSRV and a typing-stub coverage check for the Python package.

### Changed
- **MSRV bumped to 1.88** (workspace-wide `rust-version = "1.88"`).
- `vision-calibration-core`: renamed `choose_multiple` → `sample` in the RANSAC sampling API.
- Replaced `Option<Vec<_>>` with `Vec<_>` for weights fields across public configs (empty = unweighted).
- Documented the rationale for the `RUSTSEC-2024-0436` audit ignore.
- Dropped redundant empty `[features]` blocks in `vision-calibration-optim`.

### Breaking
- Minor release bump to `0.3.0` for the migration from `anyhow::Error` to typed `Error`
  in `vision-calibration-{core,linear,optim,pipeline}` public signatures.
- `choose_multiple` → `sample` rename in `vision-calibration-core`.
- MSRV raised to 1.88.
- `weights: Option<Vec<_>>` replaced by `weights: Vec<_>` in public config structs.

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
