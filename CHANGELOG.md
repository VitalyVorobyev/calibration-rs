# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration

- **`generate-manifest` is `vcal init`.** The dataset crate's
  `generate-manifest` binary and its `cli` feature are removed;
  `vcal init <folder>` writes the same draft to `<folder>/dataset.toml`
  (`-o -` prints it).

- **`RunError` has an `ImageSizeMismatch` variant**, and
  `calibrate_dataset` returns it, before any detection, when one camera's
  images differ in size. A `match` on `RunError` needs the new arm; Python's
  `RunError.kind` is `"image_size_mismatch"`.

- **`ResidualKind` has a `Custom(String)` variant** and is no longer
  `Copy`: a noise level of a residual family of your own (see
  `optim::problem` below). A `match` on `ResidualKind` needs the new arm,
  and code that copied a kind clones it. In JSON a custom family is
  `{"custom": "<name>"}`; the built-in kinds are unchanged.

- **Multiple-view geometry takes a calibration and pixels** (breaking, in
  `vision-mvg`, `vision-geometry` and the facade's `mvg`). Closed-form steps
  work on bearings, the unit rays of each camera's own unprojection, so
  every camera model works, including a fisheye beyond 90° from its axis:
  - `recover_relative_pose(&corrs)` is
    `recover_relative_pose(&rig, [camera0, camera1], &matches)`, and
    `Correspondence2D` holds a pair of **pixels**, not normalized
    coordinates. The same for `robust::estimate_essential` and
    `robust::recover_relative_pose_robust`, which take the rig, the two
    camera indices, the matches and the options (`[0, 0]` for one moving
    camera). `RelativePose` and `RobustRelativePose` return
    `second_se3_first: Iso3` (unit translation) instead of `r` and `t`;
    `RelativePose` also carries `scene: SceneDiagnostics`.
  - **The RANSAC threshold of `estimate_essential` and
    `recover_relative_pose_robust` is in radians**, the first-order angular
    epipolar error, no longer a Sampson distance on the normalized plane:
    a pixel threshold `t` is about `t / f` radians for a focal length of
    `f` px (`1e-3` is about 0.8 px at `f = 800`). `inlier_rms` is in
    radians too.
  - `triangulation::triangulate_nview`, `triangulate_two_view` and
    `triangulate_point_two_view` are `triangulation::triangulate(&rig,
    &rig_se3_world, &tracks)`, with `RigObservation { capture, camera,
    pixel }` tracks; for one stereo capture pass `&[Iso3::identity()]`.
    `TriangulatedPoint::reprojection_error` is an `Option` in pixels
    through each camera's model (`None` when the point does not project),
    `parallax_deg` is the largest angle between two observing rays and
    `in_front` means ahead along every bearing. The estimate is the linear
    triangulation; there is no Gauss-Newton polish.
  - `triangulation::triangulate_point_linear` takes poses and bearings
    (`&[Iso3]` world to camera, `&[Vec3]`) instead of projection matrices
    and image points. `triangulate_point` and both `refine_point`s are
    removed.
  - `essential_5point` and `essential_linear` take bearings (`&[Vec3]`,
    `b2ᵀ E b1 = 0`); for a pinhole, pass `(x, y, 1)` of the normalized
    coordinates.
  - `degeneracy::analyze_scene` is gone; `RelativePose::scene` carries its
    `SceneDiagnostics`, with depth measured along the first camera's
    bearing.
  Rectification, bundle adjustment and the homography and fundamental
  estimators are unchanged. A type that implements `CalibratedCameras`
  implements the new required `cam_se3_rig`.

### Added

- `core::CalibratedRig` is a calibration as a value: the cameras, each
  camera's `cam_se3_rig` and the joint parameter uncertainty. Every export
  with cameras produces one with `CalibratedCameras::calibrated_rig`, and
  `CalibratedCameras::cam_se3_rig` gives the extrinsics (the identity for a
  single camera). The multiple-view functions take it.
- `optim::problem` builds and solves a least-squares problem of your own
  (Rust): parameter blocks (vectors, SE(3) poses, unit vectors) with fixed
  components, bounds and labels; residuals you write once with a generic
  `eval` (the `Residual` trait), differentiated exactly by dual numbers;
  and the library's reprojection residual through a camera of any model
  (`add_camera`, `add_reprojection`). A reprojection reaches its camera
  through one pose or a rig chain (`PoseChain::Pose`, `PoseChain::Rig`:
  `cam_se3_rig · rig_se3_target`), projects a known point or a point being
  estimated (`TargetPoint::Known`, `TargetPoint::Block`), and its pixel's
  noise is estimated with the other reprojections' or a stated 2×2
  covariance (`PixelNoise::Estimated`, `PixelNoise::Stated`). Each residual
  of your own states its noise, known (`Noise::Known(σ)`, weighed by it) or
  estimated per named family. `solve`
  returns each block's value, the noise of each family and the parameter
  covariance, labelled as the workflows' results are, so
  `projection_uncertainty` reads a camera solved this way. Every input is
  checked as it is added. The book's "Your Own Residual" chapter adds a
  measured distance to a planar calibration.
- `optim::problem::Problem::add_calibrated_rig` adds a calibration's
  cameras and extrinsics (a `CalibratedRig`, from any result's
  `calibrated_rig()`) with its uncertainty. With `Use::Consider` they stay
  fixed and the solution's covariance adds the calibration's share: in the
  book's measurement of a part at half a metre, σ(depth) is 2.54 mm and
  matches the scatter of repeated calibrations and measurements (ratio
  0.91, as solving with the board images does), where a fixed camera
  states 0.91 mm. With `Use::Prior` they are free, held by a prior with
  the calibration's covariance, for bundle adjustment that refines the
  cameras. A calibration without a covariance is added fixed, and the
  solution's uncertainty is unavailable for the new reason
  `UncertaintyUnavailable::CalibrationWithoutCovariance` (JSON
  `"calibration_without_covariance"`; a `match` on the enum needs the new
  arm).
- `init` gathers the closed-form initializers on the facade: Zhang's
  intrinsics (with distortion), a board's pose from its homography, PnP,
  hand-eye (Tsai–Lenz) and a plane through points.
- `optim::SolveReport` is on the facade.

- `vcal`, the command line (`cargo install vision-calibration-cli`; the
  Python package installs it too): `vcal init <folder>` drafts the
  manifest, `vcal check` checks it and its images without detecting,
  `vcal run` detects, calibrates, prints each camera's parameters with
  their σ and the quality verdict, and writes `calibration.json` and
  `quality.json`, and `vcal sample list` / `vcal sample fetch <id>` give
  the sample datasets. `-c <file>` changes the workflow's default
  configuration (TOML or JSON). The exit status is 3 when a run finishes
  with a failing verdict.
- `dataset_runner::camera_image_sizes` reads each camera's image size from
  the image headers (or its region), the `image_sizes` that `assess` and
  the other image-wide analyses take; `DatasetCalibration::image_sizes`
  carries it. Python: `dataset_runner.camera_image_sizes(manifest)`.
- `RejectReason` implements `Display`.
- `analysis::html_report` renders a result as one self-contained HTML page:
  the verdict and every quality check, each camera's model and parameters
  with σ, the reprojection-error statistics, and four figures (the error
  histogram, the mean error by image region, the mean error per view and
  the projection-σ map). It loads nothing from elsewhere and is the same
  bytes for the same result. `vcal run --report` writes it as
  `report.html`, the desktop app writes it into every saved run, and Python
  has `vc.html_report(result, image_sizes=...)`.

- `dataset_runner::calibrate_dataset` runs a dataset manifest end to end in
  Rust, as `calibrate_dataset` already did in Python: the manifest's
  topology picks the workflow. `CalibrationConfig` carries the
  configuration of any workflow and `CalibrationExport` the export of any
  workflow; `CalibrationExport` implements `CalibratedCameras`, so `assess`
  and the other analyses take it directly.
- The `calibrate_folder` example calibrates a folder of your own images:
  it drafts the manifest, then detects, calibrates and prints each
  camera's parameters with their σ and the quality verdict.
- Sample datasets without a source checkout: `dataset::samples` lists real
  calibration datasets with a ready manifest (`opencv_left`,
  `opencv_fisheye_stereo`, `kuka_1`, `ds8`), each with its licence and
  citation, and `samples::fetch` (feature `fetch`) downloads one from its
  publisher at a pinned revision, verifies every file's SHA-256 and returns
  the manifest's path. Python: `vc.samples.fetch(id)`,
  `vc.samples.catalogue()`. The cache defaults to the platform's cache
  directory; `VISION_CALIBRATION_DATA` moves it.
- `analysis::predict_with_robot_poses` and `analysis::rank_robot_poses`
  plan the next robot pose of a rig hand-eye calibration: the covariance
  the calibration would have with views at candidate `base_se3_gripper`
  poses added, without a re-solve, over every estimated parameter (cameras,
  rig, hand-eye, target), with robot pose corrections marginalized under
  the result's robot pose error; candidates ranked by the reduction of
  `ln det Σ`. Python: `vc.predict_with_robot_poses`, `vc.rank_robot_poses`.
- `rig_handeye_laserline::run_calibration_with_seed` starts the joint rig
  laser solve from a `JointManualInit`: a rig hand-eye export of the same
  views (an earlier `rig_handeye` run, so a staged calibration need not
  solve the rig twice) and/or one laser plane per camera (a design value).
  A seed that does not fit the input or the config is an `InvalidInput`
  error raised before any solve; with an empty seed it is
  `run_calibration`. Python: `RigHandeyeLaserlineSession.run_calibration_with_seed`
  and `rig_handeye_laserline.JointManualInit`.

### Changed

- **A covariance with priors counts their degrees of freedom.** The
  estimated noise level of a residual family is scaled by
  `m_obs / (m − n)`, the observations' rows over the redundancy, instead of
  `m / (m − n)`: a prior holds the parameters it is over, so its rows do
  not add to the observations' degrees of freedom. Only solves with priors
  change: in the hand-eye workflows with robot-pose priors σ moves by −1.2 %
  to +0.5 % on the regression datasets (the robot-pose σ re-estimated from
  the covariance shifts with it), and the estimates by at most 0.02 σ.
- The real-image examples (`planar_real`, `handeye_session`, Rust and
  Python) fetch the `kuka_1` sample instead of reading the repository's Git
  LFS images.

### Fixed

- **`view_scatter` scales fewer covariances of data that fit the noise
  model.** Its significance threshold came from a scaled χ² fitted to two
  moments of the jackknife statistic, which is skewed when few views carry
  the information; the threshold now fits three (Pearson). On synthetic
  two-camera rigs at 6–24 views it fired on 3.0 % of clean solves (3.75 %
  at eight views or fewer) against the intended 2.5 %; it now fires on
  2.3 % (2.75 %). A false alarm at six views can scale a covariance by 12
  or more. Estimates are unchanged; the regression suite's results are
  unchanged.

### Documentation

- A "Compose Your Own Pipeline" chapter builds a measurement from the
  public pieces: closed-form starting values, one problem holding a
  calibration and the pose of a part, and the part's σ with the
  calibration's uncertainty in it. A camera held fixed at its calibrated
  values states σ about 2.5 times too small. The book's chapters on adding
  a problem type or a residual to the library are now under "Contributing
  to the Library".
- A "The Command Line" chapter walks through `vcal` on a sample dataset,
  and "The Calibration Report" describes the HTML report.
- A "Capturing Calibration Images" chapter says how to capture: how many
  views, how much tilt and in which directions, how large the board should
  be in the image and where to place it, with the uncertainty each choice
  buys, measured by the new `capture_planning` example.
- Getting Started has a real-image quickstart, in Rust and Python: draft
  the manifest from a folder, describe the board, calibrate, and read the
  σ and the verdict, or start from a fetched sample dataset.
- The book opens with a "Choose Your Workflow" chapter that maps a
  hardware setup to a workflow, a camera model and a capture plan, and the
  Introduction names the app, Python and Rust routes first.
- Every published crate's documentation opens with an example that runs
  as a test: the facade and pipeline calibrate synthetic views, and each
  problem type (`PlanarIntrinsicsProblem`, `SingleCamHandeyeProblem`,
  `LaserlineDeviceProblem`, `RigExtrinsicsProblem`, `RigHandeyeProblem`)
  carries a complete example that recovers its synthetic ground truth. The
  lower crates show one of their own tools each: Zhang's method, PnP, a
  homography, two-view relative pose, chessboard detection, the sample
  catalogue and the covariance labels. docs.rs builds the facade with its
  `synthetic`, `refine` and `fetch` features.
- Each crate's README says which crate a reader wants: the facade, unless
  they need one building block on its own.

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
