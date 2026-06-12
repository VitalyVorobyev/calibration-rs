# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING (`vision-calibration-optim`): camera model as data in the
  factor IR** ([ADR 0020](docs/adrs/0020-camera-model-as-data-factor-ir.md)).
  The 18 enumerated `FactorKind` variants
  (`ReprojPointPinhole4Dist5Scheimpflug2HandEyeRobotDelta`, …) are
  replaced by four families — `ReprojPoint`, `LaserPointToPlane`,
  `LaserLineDistance`, `Se3TangentPrior` — that carry a
  `CameraModelDesc` (projection × distortion × sensor) and a
  `ReprojChain`/`LaserChain` as data. Parameter layouts and validation
  are derived from the descriptors; the backend monomorphizes residual
  kernels once per factor through a single dispatch table. Numerics
  are bit-identical on every production path (pinned by golden-value
  tests). Adding a future camera model is one descriptor variant + one
  kernel + one dispatch row instead of new variants per chain. The
  unused `math::projection` helpers were removed.

### Added
- **Pinhole rig laserline support.**
  `RigHandeyeExport::to_upstream_calibration` and
  `pixel_to_gripper_point` accept pinhole rig hand-eye exports
  (`sensors == None`), substituting exact zero-tilt sensors.

### Fixed
- The `laserline_device` export path computes target residuals through
  the shared generic `compute_planar_target_residuals_views` helper
  instead of a stale inlined projection loop.

## [0.5.1] - 2026-05-23

### Fixed
- **Docs publishing (`publish-docs.yml`).** Repair 7 rustdoc broken
  intra-doc-links in `vision-calibration-dataset` and
  `vision-calibration-detect` (both new crates in `0.5.0`) that
  surfaced as hard errors under `RUSTDOCFLAGS=-D warnings`. The
  workflow ran cleanly under the older `0.4.x` workspace because the
  affected files did not exist. No public API change.
- **PyPI release pipeline (`release-pypi.yml`).** Re-sync
  `crates/vision-calibration-py/pyproject.toml` `project.version` with
  the workspace version. The mismatch (pyproject was stuck at `0.3.0`
  while the workspace shipped `0.4.0` and `0.5.0`) tripped the
  `Verify tag/version sync` gate, so the wheel/sdist build and PyPI
  upload were skipped for both prior tags. Wheels for `0.5.1` are the
  first PyPI upload since `0.3.0`.

## [0.5.0] - 2026-05-21

`0.5.0` bundles two pre-1.0 breaking efforts plus the first desktop
viewer scaffold: the rig-family sensor-axis refactor (ADR 0013) and a
batched public-API-surface revision applied before the library's
contract stabilizes. Pre-1.0, breaking changes are expected and are
collected here in a single minor bump rather than dribbled across
several `0.x.y` releases. The API revision draws three boundaries that
debugging and algorithm work had blurred: stable *results* (`*Export` +
typed `step_*` return values), opt-in *diagnostics* (`session.log()` /
`session.metadata()`), and implementation *internals* (now hidden or
sealed). The full audit and rationale are in
[`API_REVISION.md`](API_REVISION.md).

### Added
- **Track B / B0 — diagnose UI scaffold (ADR 0014).**
  - `vision_calibration_core::{ImageManifest, FrameRef, PixelRect}` — new
    viewer-facing image-data contract. Pose-major frame list with optional
    per-frame ROI for tiled multi-camera images. Re-exported from
    `vision_calibration::core`.
  - `PlanarIntrinsicsExport` gains an optional
    `image_manifest: Option<ImageManifest>` field. Serde-skipped when
    absent so existing exports remain byte-identical. Other `*Export`
    types extend in their own follow-up PRs (B0.5+).
  - New example
    `cargo run -p vision-calibration --example planar_synthetic_with_images`
    deterministically renders a 5-pose 9×6 checkerboard fixture
    (`target/fixtures/planar_synthetic_with_images/`) — `export.json`
    with manifest + 5 PNGs — and is the source of truth for the v0
    diagnose viewer's input contract.
  - New regression test
    `crates/vision-calibration/tests/planar_synthetic_with_images.rs`
    pins fixture residuals (mean < 0.5 px, max < 1.5 px) and verifies
    every manifest entry maps to a rendered PNG.
  - New top-level `app/` directory carrying the Tauri 2 + React +
    TypeScript desktop shell. The Rust backend (`app/src-tauri/`) is
    excluded from the workspace via `Cargo.toml`'s
    `exclude = ["app"]`. Two Tauri commands: `load_export` and
    `load_image`. One UI surface: file-open + (pose, camera) dropdown
    + canvas with per-feature residual arrows colored by error
    bucket. See `app/README.md`.
- **ADR 0014** ([`docs/adrs/0014-tauri-desktop-app.md`](docs/adrs/0014-tauri-desktop-app.md))
  records the Tauri 2 + React + TypeScript framework choice (vs
  rerun.io and egui), the Track B re-sequencing (diagnose-first vs
  the original B0 → B6 ordering), the v0 viewer-only scope, and the
  `ImageManifest` Export-side contract.
- **ADR 0013** ([`docs/adrs/0013-rig-family-sensor-axis-refactor.md`](docs/adrs/0013-rig-family-sensor-axis-refactor.md))
  records the rig family sensor-axis refactor decision: composition over
  traits, single-axis collapse, alternatives considered.
- **Typed `step_*` return values.** Every step function now returns a
  typed, non-`Option` result instead of `()` — e.g.
  `step_init -> PlanarInitResult`, `step_optimize -> PlanarOptimizeResult`,
  with analogous `*InitResult` / `*OptimizeResult` (and rig-stage
  variants) per problem type. Consumers read step outputs directly
  rather than fishing intermediate values out of `session.state`.
- **`linear::prelude`.** A curated re-export module covering the most
  common `vision-calibration-linear` items, available both on the
  `linear` crate and through `vision_calibration::linear::prelude`.
- **`CalibrationSession::log()` / `metadata()`.** Accessor methods
  returning immutable views of the session log and metadata — the
  documented introspection channel that replaces direct field access.
- **`vision_calibration_pipeline::common`.** New module holding the
  shared step-option structs, re-exported from the facade as
  `vision_calibration::common`.

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

#### Public-API surface revision

Every break and the concrete migration a consumer must apply:

| Change | Before | After |
|--------|--------|-------|
| `session.state.*` is no longer public — `CalibrationSession::state` is `pub(crate)`, and the seven `*State` structs (`PlanarState`, `SingleCamHandeyeState`, `RigExtrinsicsState`, `RigHandeyeState`, `ScheimpflugIntrinsicsState`, `LaserlineDeviceState`, `RigLaserlineDeviceState`) are `pub(crate)`. Step functions now return their own typed, non-`Option` result. | `step_init(&mut session, None)?; let k = session.state.initial_intrinsics.as_ref().unwrap();` | `let init = step_init(&mut session, None)?; let k = &init.intrinsics;` |
| Session introspection moved to accessors. `CalibrationSession::log` / `::metadata` fields are `pub(crate)`. | `&session.log`, `&session.metadata` | `session.log()`, `session.metadata()` |
| `step_set_*` manual-seed functions renamed to `step_*_with_seed`. | `step_set_init(&mut s, manual, None)?`, `step_set_intrinsics_init(...)`, `step_set_handeye_init(...)`, `step_set_rig_init(...)`, `step_set_intrinsics_init_all(...)` | `step_init_with_seed(...)`, `step_intrinsics_init_with_seed(...)`, `step_handeye_init_with_seed(...)`, `step_rig_init_with_seed(...)`, `step_intrinsics_init_all_with_seed(...)` |
| Facade `linear` no longer glob-re-exports `vision-calibration-linear`. | `use vision_calibration::linear::*;` | import the module (`vision_calibration::linear::homography`), a specific item, or `vision_calibration::linear::prelude::*` for the curated common set |
| Facade `optim` no longer glob-re-exports `vision-calibration-optim`. The curated set is `LaserPlane`, `HandEyeMode`, `RobustLoss`, `LaserlineMeta`, `LaserlineView`, `RobotPoseMeta`, plus the `compute_*_feature_residuals` helpers; typical consumers go through `pipeline`. | `use vision_calibration::optim::*;` | name the specific item (`vision_calibration::optim::LaserPlane`) or use the `pipeline` workflow |
| Facade `synthetic` no longer glob-re-exports `vision_calibration_core::synthetic`. | `use vision_calibration::synthetic::*;` | `use vision_calibration::synthetic::{planar, noise};` |
| `vision-calibration-linear` no longer flattens its modules into the crate root — items live only at their module path. | `linear::dlt_homography`, `linear::CameraMatrixDecomposition`, `linear::DistortionFitOptions` | `linear::homography::dlt_homography`, `linear::camera_matrix::CameraMatrixDecomposition`, `linear::distortion_fit::DistortionFitOptions` (or `linear::prelude` for common items) |
| `vision-calibration-optim` no longer re-exports `core` types. | `vision_calibration_optim::{RigDataset, RigViewObs, View}` | `vision_calibration_core::{RigDataset, RigViewObs, View}` |
| `pixel_to_gripper_point` moved into the `rig_laserline_device` module. | `vision_calibration::pixel_to_gripper_point(...)` | `vision_calibration::rig_laserline_device::pixel_to_gripper_point(...)` — the old crate-root path remains as a `#[deprecated]` alias for one release |
| `#[non_exhaustive]` added to growth-prone public types: all `*Export`, `*Config`, `*ManualInit`, and `*Result` structs/enums (including `PlanarRunResult`, `RigExtrinsicsOutput`, `RigHandeyeOutput`, `LogEntry`, `SessionMetadata`, the per-problem `*Options` structs, and the diagnostic types `ReprojectionStats`, `FeatureResidualHistogram`, `PerFeatureResiduals`, `TargetFeatureResidual`, `LaserFeatureResidual`, `FrameRef`, `ImageManifest`, `PixelRect`). | `Config { a, b }` (bare struct literal) | `Config { a, b, ..Default::default() }`, or a constructor / `Config::default()` then field assignment. Serde round-trips are unchanged. |
| `ProblemType` is now sealed. The seven problem types are a closed set (ADR 0013); a new `pub(crate)` `ProblemState` supertrait blocks downstream `impl`. | `impl ProblemType for MyProblem { ... }` | not supported — use one of the seven built-in problem types |
| `Detector` (`vision-calibration-detect`) is now sealed via a private supertrait. | `impl Detector for MyDetector { ... }` | not supported downstream — use the provided detectors |
| `vision_calibration_core::test_utils` is no longer public API. The module is `#[doc(hidden)]` and gated behind a non-default `test-utils` feature. | `use vision_calibration_core::test_utils::*;` | enable `features = ["test-utils"]` on the dev-dependency, or migrate to `vision_calibration_core::synthetic` helpers |
| RANSAC scaffolding (`Estimator`, `RansacOptions`, `RansacResult`, `ransac_fit`) is `#[doc(hidden)]`. Still `pub` for cross-crate use, but no longer part of the documented surface. | (documented API) | treat as internal; do not rely on it |
| Shared step-option structs hoisted to a new `vision_calibration_pipeline::common` module. `IntrinsicsInitOptions`, `IntrinsicsOptimizeOptions`, `HandeyeInitOptions`, `HandeyeOptimizeOptions` are now single types; they remain re-exported from each problem module, so existing per-module paths still resolve. | `planar_intrinsics::IntrinsicsInitOptions` (a distinct per-problem copy) | `vision_calibration_pipeline::common::IntrinsicsInitOptions` (canonical); per-module re-exports unchanged |

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
