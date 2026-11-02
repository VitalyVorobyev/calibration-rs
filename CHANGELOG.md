# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.12.0] - 2026-11-02

Minor release: Kannala-Brandt fisheye cameras calibrate in every workflow
and agree with OpenCV's `fisheye` module on its own test set, and stated
uncertainties match real-image scatter. Breaking: the camera-model enums
are `#[non_exhaustive]`, seeds and the planar init result take the camera's
own model, and Python has one `Camera` class. Calibration estimates of
existing models are unchanged; their reported uncertainties change.

### Migration

- **Python: one `Camera` class.** `PinholeCamera` and
  `PinholeScheimpflugCamera` are replaced by `Camera(intrinsics, distortion,
  projection=PinholeProjection(), sensor=IdentitySensor())`, which carries
  any model per slot. Results report every camera as a `Camera`; check
  `camera.sensor` (`IdentitySensor` or `ScheimpflugSensor`) where code
  tested the class.

- **Planar intrinsics: `IntrinsicsInitResult` is `{ camera, poses }`**, as
  in the other workflows: `camera` is the initial `CameraParams` in the
  configured model, replacing the `intrinsics`, Brown-Conrady `distortion`
  and `sensor` fields (`init.camera.intrinsics.fx` for
  `init.intrinsics.fx`). Python's `planar_intrinsics.IntrinsicsInitResult`
  likewise carries a `Camera`.

- **The camera-model enums are `#[non_exhaustive]`**: `ProjectionKind`,
  `DistortionKind`, `SensorKind`, `ProjectionParams`, `DistortionParams`
  and `SensorParams`. New models can then arrive in a minor release; a
  `match` on one of them needs a wildcard arm.

- **Distortion seeds are values in the camera's own model.**
  `IntrinsicsManualInit::distortion`, `DeviceManualInit::distortion` and the
  rig seeds' `per_cam_distortion` hold `DistortionParams` (they were
  `BrownConrady5`), and a seed's kind must equal the camera's configured
  `model.distortion`: another kind is an `InvalidInput` naming the field,
  raised before any work. A seed is used as given, no longer embedded into
  another model, so a Kannala-Brandt seed now works on a fisheye camera.
  Rust: `seed.distortion = Some(DistortionParams::BrownConrady5 { params })`.
  Python: the seed fields take any `Distortion` dataclass
  (`vc.BrownConradyDistortion(...)`, `vc.KannalaBrandt4Distortion(...)`);
  JSON seeds carry the tag, `{"type": "brown_conrady5", "k1": ...}`.

### Added

- `vision_calibration::optim::LaserlineResidualType`, the type of the laser
  workflows' `laser_residual_type` config field (a fisheye camera takes
  `PointToPlane`).
- **Fisheye camera model.** `ProjectionKind::Equidistant` (the normalized
  radius is the incidence angle, so rays beyond 90° project) and
  `DistortionKind::KannalaBrandt4` (`r_d = r (1 + k1 r² + … + k4 r⁸)`).
  Together they are the Kannala-Brandt model of OpenCV's `fisheye` module.
  Python: `EquidistantProjection`, `KannalaBrandt4Distortion`;
  `CameraModel` projects and back-projects such cameras.
  - Every workflow that calibrates a camera calibrates an equidistant one
    from scratch, a rig per camera. Its linear stage is a new
    bearing-based initializer instead of Zhang's method: no focal length
    or image size is needed. On OpenCV's own fisheye test set it lands
    within 0.3 % of OpenCV's focal length and about 1 px of its principal
    point before refinement.
  - `vision-calibration-linear` has the pieces:
    `equidistant_init::estimate_equidistant_intrinsics` (focal length,
    principal point, `k1`, `k2` and poses from planar views) and
    `planar_pose::estimate_planar_pose_from_bearings` (a board pose from
    any central camera's bearings, including points beyond 90°).
  - The analysis tools handle rays at and beyond 90° from the axis:
    projection uncertainty, `view_information`, next-view candidates and
    ranking, and the per-view pose of the reprojection report (now seeded
    from the camera's bearings instead of a pinhole homography).
  - Configuration errors: an equidistant camera with a Scheimpflug
    sensor, and the `LineDistNormalized` laser
    residual on a projection that bends lines (use `PointToPlane`).
- Each camera-model slot converts to and from its values in
  `param_names()` order: `ProjectionParams`, `DistortionParams` and
  `SensorParams` have `kind()`, `values()` and `from_values(kind, &values)`,
  and `DistortionParams::with_values` replaces the coefficients but keeps
  the model's settings. `CameraParams::blocks()` gives all four slots'
  values (intrinsics, projection, distortion, sensor). `ProjectionKind::ALL`,
  `DistortionKind::ALL` and `SensorKind::ALL` list the supported models.
- **Fisheye comparison with OpenCV.** The regression suite has
  `opencv_fisheye_left`, `opencv_fisheye_right` and `opencv_fisheye_stereo`,
  OpenCV's own fisheye stereo test set (34 pairs, 1280×800), frozen from the
  corners OpenCV publishes with it: a suite entry can name a publisher's
  `cv::FileStorage` corner files under `corners`, fetched and checksummed
  like images. calibration-rs and `cv::fisheye::calibrate` /
  `stereoCalibrate` agree within 0.01σ. The book walks through the
  calibration, and the `fisheye_opencv` example runs it. The solver
  benchmark's scene matrix has a fisheye camera for every problem type,
  laser workflows included, tested against its ground truth on both
  solver backends.

### Changed

- **`view_scatter` is measured by leaving out one view at a time, and set
  only when it is significant.** Estimates are unchanged; reported
  uncertainties change.
  - The old measure was biased low when a few views carry most of the
    information. On the public real-image cameras, calibrations from
    disjoint halves of the views still differed by 1.4–3.4 times the
    stated variance; now they differ by 0.7–1.5 times, 1.0 on average.
  - On real images `view_scatter`, and with it the covariance, grows by
    about 10–40 %.
  - On data that fit the noise model, `view_scatter` is now `None` and the
    covariance unscaled. Before, it was scaled whenever the noisy measure
    exceeded 1. It is `None` whenever the excess is within the measure's
    own sampling noise.

### Fixed

- The factrs backend panicked on a Kannala-Brandt camera, whose parameter
  block has a size it had no variable for. It now covers every camera
  model, and an unsupported block size is an error, not a panic.

## [0.11.0] - 2026-11-01

Minor release: the public API is cut down to what a calibration needs before
it is frozen for 1.0. Every change below is breaking; the numbers a
calibration produces are unchanged. Exports put residuals and image
manifests in a `diagnostics` group, detectors take typed configs and 8-bit
image views, manifests load in one call, synthetic data needs a feature,
and the low-level solve API is no longer documented API.

### Migration

- **Internal helpers leave the public API.** Nothing in the calibration
  workflows used them.
  - `vision_calibration_linear::handeye::{build_all_pairs, MotionPair}` are
    removed; the hand-eye initializers form their motion pairs internally.
  - `LaserPlane::{to_dvec, from_dvec, normal_to_dvec, distance_to_dvec,
    from_split_dvec}` are removed from the public API; build a plane with
    `LaserPlane::new` and read `normal` / `distance`.
  - `vision_mvg::cheirality` (and `mvg::cheirality` on the facade) is
    private: `recover_relative_pose` and `recover_relative_pose_robust` pick
    the pose. `degeneracy::{detect_pure_rotation, detect_planar_scene,
    detect_poor_baseline}` and `triangulation::triangulate_two_view_partial`
    are gone; `degeneracy::analyze_scene` reports the same verdicts.
  - The session types are reached only as `session::X`; the `calibsession`,
    `problem_type` and `types` submodules are private.
  - The detection cache types are reached only at the crate root of
    `vision-calibration-detect`; the `cache` module and `hash_image_bytes`
    are private (`CacheKey::from_inputs` hashes the image).
- **One import path per type.**
  - The shared step options (`IntrinsicsInitOptions`,
    `IntrinsicsOptimizeOptions`, `RigInitOptions`, `RigOptimizeOptions`,
    `HandeyeInitOptions`, `HandeyeOptimizeOptions`) are imported from
    `common` only; the problem modules no longer re-export them. Step
    results and manual seeds stay per problem.
  - Types shared by several workflows moved to `common`: `RobotPoseMeta`,
    `RigLaserlineView`, `HandeyeEstimate` and `HandeyeParams`. The facade's
    `optim` module no longer carries input types; `LaserlineMeta` and
    `LaserlineView` are in `laserline_device`.
  - `single_cam_handeye::HandeyeMeta` is removed: a single-camera hand-eye
    view is `View<RobotPoseMeta>`, the type the rig workflows use. The JSON
    is unchanged (`{"base_se3_gripper": …}`).
  - `prelude::run_planar_intrinsics` is removed; call
    `planar_intrinsics::run_calibration`.
  - The facade's `linear` and `geometry` modules are removed. The two-view
    solvers (`dlt_homography`, `essential_5point`, `fundamental_8point`,
    `decompose_essential`, `triangulate_point_linear`, `dlt_camera_matrix`,
    `decompose_camera_matrix`, `Mat34`, …) are at the root of `mvg`; the
    closed-form initializers are used through the `vision-calibration-linear`
    and `vision-geometry` crates directly.
  - `vision_calibration_core::coordinate_utils` is private; its four
    functions stay at the crate root.
- **Exports separate the result from its diagnostics (JSON).** Every
  `*Export` has a `diagnostics` group (`common::ExportDiagnostics`) holding
  `per_feature_residuals` and `image_manifest`; the two top-level fields are
  gone. Read `export.diagnostics.per_feature_residuals` and set
  `export.diagnostics.image_manifest`. `CalibratedCameras::feature_residuals()`
  and `images()` become `diagnostics()`. Python results keep their
  `per_feature_residuals` property.
- **`PlanarIntrinsicsExport` drops `report`** (Python: `final_cost`). The
  solve's cost and iteration count stay on the session output
  (`PlanarIntrinsicsEstimate::report`).
- **`LaserlineDeviceExport` matches the other exports.** `estimate` is
  replaced by `params` (camera, laser plane, target poses) and a top-level
  `uncertainty`; `export.estimate.params.plane` becomes `export.params.plane`.
  Python's `LaserlineDeviceExport` follows (`result.params`,
  `result.uncertainty`), and `LaserlineEstimate` is removed.
- **`PlanarIntrinsicsParams` loses its accessors.** `intrinsics()` and
  `poses()` duplicated the public fields, and `distortion()` returned
  `None` for any non-Brown-Conrady model (and zeros for no distortion). Read
  `params.camera.intrinsics`, `params.camera_se3_target`, and match
  `params.camera.distortion`.
- **Manifests load in one call, and the runner takes a detection mode.**
  `DatasetSpec::from_path` reads a TOML or JSON manifest (errors are
  `ManifestError`). The `build_*_input` converters take a `Detection` in
  place of `cache, force_redetect`: `Detection::Cached(&cache)` for
  `(&cache, false)`, `Detection::Refresh(&cache)` for `(&cache, true)`, and
  the new `Detection::Uncached` for a run without a cache. The laser
  converters take it before the extractor:
  `build_laserline_device_input(&spec, dir, Detection::Cached(&cache), &extractor)`.
- **Synthetic data is a feature.** `vision_calibration::synthetic` (and
  `vision_calibration_core::synthetic`) need the new `synthetic` feature:
  `vision-calibration = { version = "…", features = ["synthetic"] }`.
- **The low-level solve API is workspace-internal.** The
  `vision-calibration-optim` entry points (`optimize_*`,
  `evaluate_rig_handeye_laserline`), their option structs
  (`*SolveOptions`, `BackendSolveOptions`, `CameraBounds`,
  `ScheimpflugStagedInitOptions`) and helpers (`view_information`,
  `with_leading_radial`, `projection_covariance_at`, `apply_robot_delta`)
  stay callable but are hidden from the docs and outside the semver
  promise, as are core's `FixedParams` and `INTRINSICS_PARAM_NAMES`.
  `planar_intrinsics::PlanarIntrinsicsSolveOptions` is no longer
  re-exported, and the configs' `init_opts()`, `solve_opts()`,
  `backend_options()`, `iterative_opts()` and `distortion_fit_opts()` are
  private. Calibrate through the step functions; set the solver on the
  config (`config.solver.backend`, `config.solver.max_iters`).
- **Detectors are a typed enum.** `DetectorConfig` (one variant per
  detector, carrying `ChessboardConfig`, `CharucoConfig`,
  `PuzzleboardConfig` or `RinggridConfig`) replaces the sealed `Detector`
  trait and the `ChessboardDetector`, `CharucoDetector`,
  `PuzzleboardDetector` and `RinggridDetector` types:
  `ChessboardDetector.detect_json(&image, &json)` becomes
  `DetectorConfig::Chessboard(cfg).detect(view)`. Detectors read an 8-bit
  grayscale `GrayImageView` (`GrayImageView::new(width, height, &pixels)`)
  instead of an `image::DynamicImage`; so do
  `dataset_runner::detect_target` and `LaserPixelExtractor::extract`.
  `CacheKey::for_detector(bytes, &detector, &json)` becomes
  `CacheKey::for_detection(bytes, &config, roi)`, `CacheKey::from_inputs`
  takes any serializable config, and `CacheError::Serde` carries a message.
  `DetectError::Config` is gone (configs are typed). A manifest
  `detector.chess_corners` override on a PuzzleBoard or ring-grid target is
  now rejected up front. Cached detections are recomputed once. Neither
  `image` nor `serde_json` appears in the detector API any more.
- **Session bookkeeping is read-only.** `session::current_timestamp` is
  private, as are the constructors of `SessionMetadata`, `LogEntry` and
  `ExportRecord`; read them through `metadata()`, `log()` and `exports`.

### Added

- **`vision_calibration::nalgebra`** re-exports the linear-algebra crate the
  math types are built on, so callers can name its types in the matching
  version.

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

- [`0.10.0`](docs/changelog/0.10.0.md) - 2026-10-31
- [`0.9.0`](docs/changelog/0.9.0.md) - 2026-09-29
- [`0.8.3`](docs/changelog/0.8.3.md) - 2026-09-28
- [`0.8.2`](docs/changelog/0.8.2.md) - 2026-09-27
- [`0.8.1`](docs/changelog/0.8.1.md) - 2026-09-07
- [`0.8.0`](docs/changelog/0.8.0.md) - 2026-08-16
- [`0.7.0`](docs/changelog/0.7.0.md) - 2026-07-10
- [`0.6.0`](docs/changelog/0.6.0.md) - 2026-06-17
- [`0.5.1`](docs/changelog/0.5.1.md) - 2026-05-23
- [`0.5.0`](docs/changelog/0.5.0.md) - 2026-05-21
- [`0.4.0`](docs/changelog/0.4.0.md) - 2026-04-29
- [`0.3.0`](docs/changelog/0.3.0.md) - 2026-04-12
- [`0.2.0`](docs/changelog/0.2.0.md) - 2026-03-07
- [`0.1.2`](docs/changelog/0.1.2.md)
