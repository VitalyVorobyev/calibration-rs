# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.10.0] - 2026-10-31

### Migration

- **Camera model and intrinsics.**
  - Intrinsics are exactly `fx, fy, cx, cy`; skew is gone. `FxFyCxCySkew`
    becomes `FxFyCxCy` (no `skew` field; `k_matrix()` has a zero in the skew
    slot). `IntrinsicsParams` and `AnyIntrinsics` are removed:
    `CameraParams::intrinsics` is the plain `FxFyCxCy<Real>`. Delete `skew:`
    initializers and `.skew` reads. `zero_skew` is removed from
    `IntrinsicsInitConfig` (also in Python), `IterativeIntrinsicsOptions` and
    `ScheimpflugIntrinsicsInitOptions`; Zhang's initialization always drops
    the skew. `pack_intrinsics` no longer returns a `Result`. JSON intrinsics
    lose `skew` and the `"type": "fx_fy_cx_cy_skew"` tag
    (`{"fx": …, "fy": …, "cx": …, "cy": …}`); Python `PinholeIntrinsics` has
    no `skew`.
  - `ProjectionModel` is now `in_domain` / `project` / `unproject` /
    `is_central` (was `project_dir` / `unproject_dir`).
    `Camera::backproject_pixel` returns `Ray { origin, direction }`; replace
    `ray.point` with `ray.point_at_unit_depth()` (an `Option`, always `Some`
    for a pinhole) or use `ray.direction`. The distortion, sensor and
    projection traits drop their `Copy` bound; `HomographySensor` is generic
    over the scalar; `ScheimpflugParams<T = f64>` is a sensor model.
  - The kinds live in `vision-calibration-core`: `ProjectionKind`,
    `DistortionKind`, `SensorKind`, `CameraModelSpec`. `CameraModelDesc` is
    renamed `CameraModelSpec`, its constants (`PINHOLE4_DIST5`, …) become
    `CameraModelSpec::pinhole(distortion, sensor)`, and `SensorKind::None` /
    `Scheimpflug2` become `Identity` / `Scheimpflug`. Facade
    `optim::DistortionKind` moves to `core::DistortionKind`.
  - The shared linear-algebra primitives live only at
    `vision_calibration_core::linalg`; `vision_calibration_linear::math` keeps
    just `ridge_lstsq` and `project_to_so3`, and `vision_geometry::math` is
    private.

- **Every workflow takes `camera: CameraConfig`.**
  `CameraConfig { model: CameraModelSpec, fix: CameraFix }` selects any
  distortion kind (`none`, `brown_conrady5`, `rational8`, `thin_prism9`,
  `division1`) and an identity or Scheimpflug sensor, with fix lists named
  per slot (`camera.fix.intrinsics`, `.distortion`, `.sensor`). A slot left
  out keeps the model's default (`k3` for Brown-Conrady; everything but `k1`,
  `k2` / `lambda` under a Scheimpflug sensor); an unknown name is a
  validation error. Exports and results carry `CameraParams` (JSON
  `{"projection", "distortion", "sensor", "intrinsics"}`, each slot tagged by
  `type`) in place of `PinholeCamera` or separate `sensors`.
  - `scheimpflug_intrinsics` is merged into `planar_intrinsics`
    (`camera.model.sensor = scheimpflug`, `kind: "planar_intrinsics"`).
    Renames: `ScheimpflugIntrinsicsProblem` / `Config` / `Result` / `Export` /
    `Params` / `…InitResult` / `…OptimizeResult` → the `PlanarIntrinsics*` /
    `PlanarInitResult` / `PlanarOptimizeResult` equivalents;
    `ScheimpflugManualInit` → `PlanarManualInit` (gains
    `sensor: Option<ScheimpflugParams>`); `distortion_model` →
    `camera.model.distortion`; `fix_camera.intrinsics` / `.distortion` →
    `camera.fix.intrinsics` / `.distortion` (under a Scheimpflug sensor
    `k4`–`k6`, `s1`–`s4` are named directly); `fix_scheimpflug` →
    `camera.fix.sensor`; `ExportKind::ScheimpflugIntrinsics` →
    `ExportKind::PlanarIntrinsics`; dataset `Topology::ScheimpflugIntrinsics`
    / `"scheimpflug_intrinsics"` → `"planar_intrinsics"`;
    `device_seed::scheimpflug_seed` returns `PlanarManualInit`;
    `scheimpflug_intrinsics::run_calibration(s, Some(config))` →
    `s.set_config(config)?; planar_intrinsics::run_calibration(s)`. Python
    `run_scheimpflug_intrinsics` → `run_planar_intrinsics` with
    `PlanarCalibrationConfig(camera=CameraConfig(model=CameraModelSpec(sensor="scheimpflug")))`;
    `PlanarCalibrationConfig.distortion_model` / `.fix_camera` →
    `.camera.model.distortion` / `.camera.fix`. Planar `solver.max_iters`
    defaults to 120 (was 50). A general homography sensor, a Scheimpflug
    sensor without distortion, and a sensor change between `step_init` and
    `step_optimize` are typed errors.
  - optim: `ScheimpflugIntrinsicsParams`, `…SolveOptions`, `…Estimate`,
    `optimize_scheimpflug_intrinsics` → `PlanarIntrinsicsParams`,
    `PlanarIntrinsicsSolveOptions { robust_loss, fix: CameraFix, fix_poses,
    bounds: Option<CameraBounds> }`, `PlanarIntrinsicsEstimate`,
    `optimize_planar_intrinsics`; `ScheimpflugBounds` → `CameraBounds`;
    `optimize_scheimpflug_intrinsics_staged` takes the planar types (a
    non-Scheimpflug camera is a typed error);
    `params::distortion::{fix_mask_indices, distortion_kind}` →
    `CameraFix::resolve`, `CameraParams::spec`; `PlanarIntrinsicsParams::
    pinhole_camera()` is removed (read `params.camera`).
  - Single-camera hand-eye and the laserline device gain `camera:
    CameraConfig` (default Brown-Conrady, identity sensor). Their init and
    optimize results carry `camera: CameraParams`.
    `SingleCamIntrinsicsManualInit` gains `sensor: Option<ScheimpflugParams>`
    (a non-zero tilt for an untilted camera is a typed error).
    `LaserlineDeviceConfig.sensor_init` → `camera.model.sensor =
    scheimpflug` plus `LaserlineDeviceManualInit.sensor`;
    `optimize.fix_camera` → `camera.fix.intrinsics` / `.distortion`;
    `optimize.fix_sensor` → `camera.fix.sensor` (unset holds the tilt at its
    seed; an explicit list estimates it). `LaserlineDeviceInitResult` is
    `{ camera: CameraParams, … }`.
  - Rig extrinsics and rig hand-eye (so also `RigHandeyeLaserlineConfig.
    handeye`) take `camera: CameraConfig` for all cameras and `camera_overrides:
    Vec<CameraOverride { camera, model, fix }>` for single cameras (an index
    out of range or listed twice is a validation error). `SensorMode`
    (`Pinhole` / `Scheimpflug`, JSON `"sensor": {"kind": …}`) is removed:
    use `camera.model.sensor` per camera or override.
    `distortion_mask_in_percam_ba` → `camera.fix.distortion`;
    `refine_scheimpflug_in_rig_ba` → `rig.refine_sensor`;
    `handeye_optimize.refine_scheimpflug` → `.refine_sensor`;
    `init_tilt_x` / `init_tilt_y` are removed (seed the tilt through manual
    init `per_cam_sensors`; a seed is used verbatim and skips the sweep;
    zero tilts need nothing). `RigExtrinsicsOutput::{Pinhole, Scheimpflug}`
    and `RigHandeyeOutput::{…}` → `RigExtrinsicsEstimate { params:
    RigExtrinsicsParams { cameras, cam_to_rig, rig_from_target } }` and
    `HandeyeEstimate { params: HandeyeParams { cameras, cam_to_rig, handeye,
    target_poses } }`. Exports' `cameras: Vec<PinholeCamera>` + `sensors` →
    `cameras: Vec<CameraParams>`; step results' `per_cam_intrinsics` +
    `per_cam_sensors` → `cameras: Vec<CameraParams>`;
    `RigUpstreamCalibration { cameras, cam_se3_rig, rig_se3_target }`.
    optim: `optimize_rig_extrinsics_scheimpflug`,
    `optimize_handeye_scheimpflug` and their `*Scheimpflug*` types are
    removed (`optimize_rig_extrinsics` / `optimize_handeye` take any model
    per camera); `RigExtrinsicsSolveOptions` / `HandeyeSolveOptions` use
    `fix: Vec<CameraFix>` (one per camera, empty = model defaults) instead of
    `default_fix`, `camera_overrides`, `*scheimpflug*`;
    `RigHandeyeLaserlineParams.sensors` is removed;
    `RigHandeyeLaserlineSolveOptions` uses `fix: Vec<CameraFix>`;
    `RigLaserlineUpstream { cameras, … }`;
    `compute_rig_laserline_feature_residuals(dataset, cameras:
    &[CameraParams], cam_se3_rig, rig_se3_target, planes)`;
    `evaluate_rig_handeye_laserline` returns a `Result`.
  - The rig laser workflows accept every distortion kind per camera and name
    the joint stage's fixed parameters per camera:
    `joint_ba.default_camera_fix` + `fix_scheimpflug` →
    `joint_optimize.fix: Vec<CameraFix>` (empty = intrinsics and `k1`, `k2`
    refined, higher-order distortion and tilt held, as before).
    optim `LaserlineParams { camera: CameraParams, poses, plane }` with
    `LaserlineParams::new(camera, poses, plane)`;
    `LaserlineSolveOptions { fix: CameraFix, … }`;
    `LaserlinePlaneSolver::from_views` accepts any
    `Camera<Real, P, D, Sm, K>`.
  - Removed fix masks: core `IntrinsicsFixMask`, `DistortionFixMask`,
    `CameraFixMask`; optim `ScheimpflugFixMask`. Python:
    `PinholeBrownConradyCamera` and `CameraFixMask` are removed;
    `SingleCamHandeyeResult.camera` is a `PinholeCamera` or
    `PinholeScheimpflugCamera`; `LaserlineEstimateParams.camera` replaces
    `.intrinsics` / `.distortion` / `.sensor`; rig configs take
    `camera=CameraConfig(…)` and `camera_overrides=[…]` instead of `sensor=`
    (the `SensorMode` classes are removed);
    `RigHandeyeLaserlineBaConfig.default_camera_fix` / `.fix_scheimpflug` →
    `.fix: list[CameraFix]`.
  - Behaviour. An unset `intrinsics.fix_tangential` / `init.fix_tangential`
    is now `Option<bool>` everywhere and follows the camera: `true` for a
    Scheimpflug camera (each rig camera too), `false` otherwise. Single-camera
    hand-eye with a Scheimpflug sensor and no tilt seed takes the planar tilt
    route (linear sweep, then staged multi-start). In the laserline device
    only an explicitly estimated tilt takes that route; a held tilt
    initializes as an untilted camera. A camera model changed after the
    camera's init step, a non-zero tilt seed for an untilted camera (it was
    ignored), and a Scheimpflug sensor without distortion are typed errors.
    `rig.refine_intrinsics_in_rig_ba` holds each camera's fix lists in the rig
    bundle adjustment (it held only `k3`). An untilted camera has no sensor
    block, so the fixed zero-tilt labels (`tilt_x`, `tilt_y`) leave its
    covariance. Default-configuration results are unchanged.

- **Config and API naming is uniform across the seven problem types**,
  identical in Rust, Python, the JSON schemas and the app. Each module
  exports `<Name>Problem`, `<Name>Input`, `<Name>Config`, its estimate and
  `<Name>Export`; step returns are `<Stage><Action>Result`; every step takes
  `Option<<Stage><Action>Options>`, its `*_with_seed` variant a
  `<Stage>ManualInit`; `run_calibration` takes only the session. `Handeye`
  is one word everywhere. Export JSON and numeric results are unchanged; a
  saved config needs the four field renames below.

  | Before | After |
  |---|---|
  | `rig.refine_intrinsics_in_rig_ba` | `rig.refine_intrinsics` |
  | `init.init_iterations` / `intrinsics.init_iterations` | `init.iterations` / `intrinsics.iterations` |
  | `RigHandeyeConfig.handeye_ba: HandeyeBaConfig` | `handeye_optimize: HandeyeOptimizeConfig` |
  | `RigHandeyeLaserlineConfig.joint_ba: RigHandeyeLaserlineBaConfig` | `joint_optimize: JointOptimizeConfig` |
  | `LaserlineDeviceOptimizeConfig` | `DeviceOptimizeConfig` |
  | `HandEyeMode`, `HandEyeParams`, `HandEyeEstimate`, `HandEyeDataset`, `HandEyeSolveOptions`, `linear::handeye::HandEyeInit` | `Handeye…` spellings |
  | `analysis::ReprojLevel::HandEye` (JSON `"hand_eye"`) | `ReprojLevel::Handeye` (`"handeye"`) |
  | `LaserlineDeviceOutput` | `LaserlineDeviceEstimate` |
  | `RigHandeyeLaserlineOutput { estimate, handeye_mode }` | `RigHandeyeLaserlineEstimate` itself (`output.estimate.x` → `output.x`); the mode is `config.handeye.handeye_init.handeye_mode` |
  | planar input `PlanarDataset` | `PlanarIntrinsicsInput` (alias of `PlanarDataset`) |
  | `PlanarRunResult.dataset`, `RigRunResult.dataset` | `.input` |
  | `PlanarInitResult` / `PlanarOptimizeResult`; `SingleCam…InitResult` / `…OptimizeResult`; `RigIntrinsicsInitAllResult` / `…OptimizeAllResult`; `RigHandeye…` equivalents | `IntrinsicsInitResult`, `IntrinsicsOptimizeResult`, `RigInitResult`, `RigOptimizeResult`, `HandeyeInitResult`, `HandeyeOptimizeResult`; laserline: `DeviceInitResult`, `DeviceOptimizeResult` |
  | `PlanarManualInit`, `SingleCam…ManualInit`, `RigIntrinsicsManualInit`, `RigExtrinsicsManualInit`, `RigHandeye…ManualInit`, `LaserlineDeviceManualInit`, `RigLaserlineDeviceManualInit` | `IntrinsicsManualInit`, `HandeyeManualInit`, `RigManualInit`, `DeviceManualInit`, `PlaneManualInit` |
  | `planar_intrinsics::FilterOptions`; `step_filter(s, opts)`, `run_calibration_with_filtering(s, opts)` | `IntrinsicsFilterOptions`; both take `Some(opts)` (`None` = defaults) |
  | rig `step_rig_init(s)`, `step_rig_init_with_seed(s, seed)` | `step_rig_init(s, None)`, `…_with_seed(s, seed, None)`; options `RigInitOptions` |
  | per-module `RigOptimizeOptions` | one type, also at `common::RigOptimizeOptions` |
  | `rig_laserline_device::StepOptions`; `step_init(s)` | `PlaneOptimizeOptions`; `step_init(s, None)`, options `PlaneInitOptions` |
  | `laserline_device::run_calibration(s, Some(config))` | `s.set_config(config)?; run_calibration(s)` |
  | now re-exported | `single_cam_handeye::{HandeyeEstimate, HandeyeParams}`, `RigLaserlineEstimate`, `RigHandeyeLaserlineEstimate` |
  | Python `*CalibrationConfig` (`PlanarCalibrationConfig`, …) | `PlanarIntrinsicsConfig`, `SingleCamHandeyeConfig`, `RigExtrinsicsConfig`, `RigHandeyeConfig`, `LaserlineDeviceConfig`, `RigLaserlineDeviceConfig`, `RigHandeyeLaserlineConfig` |
  | Python `*Result` (`PlanarCalibrationResult`, `SingleCamHandeyeResult`, …) | `…Export` (`PlanarIntrinsicsExport`, …) |
  | Python `*Dataset` (`PlanarDataset`, `LaserlineDataset`, …) | `…Input` (`PlanarIntrinsicsInput`, `LaserlineDeviceInput`, …) |
  | Python `HandeyeBaConfig`, `RigHandeyeLaserlineBaConfig` | `HandeyeOptimizeConfig`, `JointOptimizeConfig` |
  | Python `RigLaserlineUpstreamCalibration`, `LaserlineEstimateParams`, `LaserlinePlane` | `RigUpstreamCalibration`, `LaserlineParams`, `LaserPlane` |
  | Python `types.HandEyeMode`, `types.PlanarInput`, `types.CameraModel` (a `CameraParams` alias) | `types.HandeyeMode`, `types.PlanarIntrinsicsInput`, `types.CameraParams` (`CameraModel` is the camera object, as in Rust) |

- **Robot-pose corrections.** A view's correction is
  `base_se3_gripper · exp(δ)` (it was `exp(δ) · base_se3_gripper`), so
  `robot_poses.rot_sigma` / `trans_sigma` are the tool's orientation and
  position errors and `robot_deltas` entries are gripper-frame corrections.
  Every hand-eye result that refines the robot poses changes (on `kuka_1`
  the largest translation correction drops from 3.7 to 1.5 mm; on synthetic
  scenes with tool-frame error the hand-eye translation error halves).
  Re-run calibrations. Code that applies `robot_deltas` itself calls
  `apply_robot_delta`, which now multiplies on the right with the same SE(3)
  exponential as the solve.
  - The priors are weighed against the corner noise, which the solve
    estimates, and every view's pose is refined (view 0 was fixed): so
    `rot_sigma` and `trans_sigma` are the standard deviations of the robot's
    pose error, `robot_deltas[0]` is non-zero and the covariance has a block
    for it. Results change in `single_cam_handeye`, `rig_handeye`,
    `rig_handeye_laserline` and the matching optim options with
    `robot_poses.refine` (the default); the standard deviations become
    honest. The solve estimates the robot's error by default
    (`estimate_sigma`, below); `refine: false` takes the poses as exact.

- **Linear stage fits only what the camera estimates.**
  `IntrinsicsInitConfig` loses `fix_tangential` (an unknown field is now a
  configuration error; Python `IntrinsicsInitConfig(fix_tangential=...)` is
  removed). A term the camera's fix list holds (`camera.fix.distortion` or
  the model's default) stays zero from the start, as do tangential terms
  while a tilt is estimated, and cameras without tangential terms
  (`division1`, `none`) fit none. To hold `p1`, `p2`, list them in
  `camera.fix.distortion`, which holds them in every stage. `fix_k3` stays.
  `IntrinsicsInitConfig::iterative_opts` and `distortion_fit_opts` take the
  camera's `CameraModelSpec` instead of a `bool`. Results change only for
  configurations that fitted then froze such terms.

- **Tilted cameras reach their minimum.** For Scheimpflug cameras in
  `rig_extrinsics` and `rig_handeye`, and the per-camera tilt solve of
  `single_cam_handeye` and an unseeded `laserline_device`, the linear stage no
  longer fits `p1`, `p2` (they mimic a tilt and stayed frozen) and the
  camera's own solve frees every pose. Results change; on a synthetic
  Scheimpflug rig the reprojection error falls to the noise floor and tilt
  errors from degrees to tenths of a degree.

- **Defaults that change results.**
  - Rig results refine each camera's intrinsics jointly with the rig:
    `rig.refine_intrinsics` (rig extrinsics, rig hand-eye) and the new
    `handeye_optimize.refine_intrinsics` default to `true`, so rig
    covariances include the intrinsics and projection uncertainty works for
    rigs. Set both to `false` for the previous results.
  - The Scheimpflug intrinsics and laserline device `fix_poses` defaults
    (configs, solve options, Python configs) hold no pose fixed, so reported
    uncertainty is not conditional on view 0; pass `fix_poses=[0]` for the
    old behaviour.
  - The laser robust loss defaults to `Huber { scale: 1.0 }` (was `0.01`;
    the residual is in pixels, where 0.01 made the fit nearly L1) in
    `DeviceOptimizeConfig`, `LaserlineSolveOptions` and Python. Set
    `laser_loss: {"Huber": {"scale": 0.01}}` for the previous fit.
  - `SolveReport.final_cost` is the robust objective `½ Σ ρ(‖rᵢ‖²)` (`½‖r‖²`
    without a loss); for robust solves it is not comparable with earlier
    releases.

- **Per-point covariances replace weights.** `CorrespondenceView` loses
  `weights`, `new_with_weights`, `weight(idx)` and `iter_weighted`; use
  `new_with_covariances(points_3d, points_2d, cov_px)`, `covariance(idx)`,
  `covariances()` (JSON `cov_px`: `{"xx", "xy", "yy"}` in px² per point).
  A weight `w` is the covariance `PointCovariance::new(1.0 / w, 0.0, 1.0 / w)`;
  weight 1 is no covariance. Every view of an input states covariances or
  none does. The field is private: build views with
  `CorrespondenceView::new(..)` or `new_with_covariances(..)`. Python
  `Observation(weights=...)` → `Observation(cov_px=[PointCovariance(xx, xy,
  yy), ...])`. `ResidualNoise.sigma` becomes `ResidualNoise.level:
  NoiseLevel` (`Sigma(σ̂)` or `Sigma0(σ̂₀)`); JSON keeps `sigma` for the first
  and writes `sigma0` for the second; read `noise.level.value()` or match.
  Python keeps `ResidualNoise.sigma` and adds `.whitened`.

- **RANSAC thresholds are explicit.** `RansacOptions` has no `Default`: use
  `RansacOptions::new(thresh)` (`RansacOptions { max_iters: 500,
  ..RansacOptions::new(t) }` works). A non-finite or non-positive threshold
  is an `InvalidInput` error. `thresh` is a distance (never squared) in each
  estimator's unit: `dlt_ransac` and `dlt_homography_ransac` pixels;
  `fundamental_8point_ransac` Sampson distance in the input's unit;
  `mvg::robust::estimate_essential` and `recover_relative_pose_robust`
  Sampson distance on the normalized plane (`t / f` for a pixel threshold);
  `mvg::robust::estimate_homography` symmetric transfer error in the points'
  unit. `RansacOptions::default()` → `RansacOptions::new(2.0)` reproduces
  the old pixel behaviour (use e.g. `1.0e-3` for the essential matrix).

- **Solver backend.** `BackendSolveOptions` gains `backend`
  (`SolverBackend`, default `TinySolver`; end struct literals with
  `..BackendSolveOptions::default()`). The `*_with_backend` functions
  (`optimize_planar_intrinsics_with_backend`,
  `optimize_scheimpflug_intrinsics_with_backend`) are removed: the plain
  functions read `BackendSolveOptions::backend`.
  `PlanarIntrinsicsConfig::backend_opts()` and
  `LaserlineDeviceConfig::backend_opts()` are removed; use
  `config.solver.backend_options()`. `BackendSolveOptions::linear_solver` and
  `LinearSolverKind` are removed (the loop always uses sparse Cholesky);
  JSON that still carries `"linear_solver"` loads and is ignored.

- **Laser residual records.** `LaserFeatureResidual::projected_line_px` is
  removed: the projected lines are `PerFeatureResiduals::laser_lines`
  (`LaserLineTrace` per view and camera, a polyline in `points_px`), and
  `compute_laserline_feature_residuals` /
  `compute_rig_laserline_feature_residuals` return `LaserResidualRecords`
  (`features`, `lines`).

- **Chessboard partial views.** A `single_cam_handeye` or rig manifest with
  a plain chessboard whose views of one camera are all partial or turned by
  90° fails with `RunError::NoWholeChessboardViews` instead of calibrating a
  moved target frame. Capture the whole board or use a ChArUco or PuzzleBoard
  target; add an arm for the variant to exhaustive matches on `RunError`.
  `detector.min_features_per_view = rows × cols` is no longer needed.

- **Python.** `ScheimpflugSensor.from_payload` reads only `tilt_x` / `tilt_y`
  (`tau_x` / `tau_y` are not accepted). The package depends on `numpy`.
  Python payload types are generated (see Changed).

### Added

- **Uncertainty and quality.**
  - Parameter uncertainty: every `*Export` (and optimizer estimate) carries
    `uncertainty` (the laserline device in `estimate.uncertainty`): the
    covariance of the estimated parameters by label (`fx`, `k1`, `tilt_x`,
    `cam1.cam_se3_rig.tx`, `plane.nx` …) with `std_dev` and `correlation`,
    per-view blocks for board or rig poses and robot-pose corrections, the
    fixed parameters it is conditional on, and a robust noise estimate per
    residual kind. It reports `rank_deficient` parameters, parameters on a
    bound and solves that ran out of iterations; robust losses and
    robot-pose priors are accounted for. Python `result.uncertainty`
    (`ParameterUncertainty`).
  - View scatter: `ParameterCovariance::view_scatter` (Python
    `ParameterUncertainty.view_scatter`) says how many times the shared
    parameters vary more across views than the residual noise predicts,
    from the scatter of the views' contributions to the fit (errors a
    view's corners share: lens model, target, detector bias). From six
    views, the covariance is scaled by it when it is above 1. On the public
    cameras it is 3–11, and calibrations from disjoint halves of the views
    now differ by 1.1–1.8 times their stated σ instead of 2.3–5.
  - Robot-pose error estimate: with robot-pose refinement,
    `robot_poses.estimate_sigma` (default on) estimates the robot's
    orientation and position error from the corrections (variance
    components), starting from `rot_sigma` / `trans_sigma`, and exports it
    as `robot_pose_sigma` (`RobotPoseSigma`). A prior ten times too tight
    or too loose no longer misstates the hand-eye σ; off, the configured
    values are taken as the robot's accuracy.
  - Projection uncertainty: `CalibratedCameras::projection_uncertainty` and
    `optim::projection_uncertainty` give the major-axis σ in pixels of where
    a camera-frame ray lands, on a 16 × 12 grid with centre, largest corner
    and maximum. `CalibratedCameras::projection_covariance(camera, pixels)`
    and `optim::projection_covariance_at` return the full 2 × 2 covariance
    in px² per pixel (`None` where no ray reaches it; Rust only). Python
    `vision_calibration.projection_uncertainty(result, (width, height),
    camera=0)`; every result keeps its export as `result.export`.
  - Quality verdict: `analysis::assess(&export, image_sizes,
    &QualityThresholds)` returns a `QualityReport` with a `pass` / `warn` /
    `fail` verdict and typed checks (status, value, unit, threshold,
    message): observability (`rank_deficient`, no redundancy, parameter at a
    bound), convergence, a χ² test against a stated
    `expected_feature_noise_px` (no default), the largest projection σ per
    camera, image coverage (8 × 6 cells), views whose mean residual exceeds
    3 σ̂, and parameter correlation above 0.99. Checks whose inputs are
    missing are `not_applicable`. Python `vision_calibration.assess(result,
    thresholds=None, image_sizes=None)`; `calib-bench accept` fails an entry
    whose report fails (`accept.quality` in a registry entry).
    `CalibratedCameras` also gives feature residuals and the image manifest.
  - Per-point pixel covariance: an observation may state each point's 2 × 2
    covariance (`PointCovariance`, px²). Residuals are whitened by it in both
    backends; noise is reported as `σ̂₀` (JSON `sigma0`, about 1 when the
    covariances hold) and the parameter covariance is scaled by it. The
    variance-factor check tests `σ̂₀²` against 1, the view-outlier check uses
    the new `error_whitened`, and a robust loss's scale applies to the
    whitened residual. Mixing views with and without covariances is
    `Error::MixedCovariance`; without covariances nothing changes. Python:
    `PointCovariance`, `Observation.cov_px`, `ResidualNoise.whitened`,
    `TargetFeatureResiduals.error_whitened`.
  - Detector corner covariance: `[detector.chess_corners.covariance]`
    (`image_noise = "estimate"`, the default, or `{ known = σ }` in gray
    levels) makes the chessboard and ChArUco detectors report each corner's
    covariance and the runner attach it. Off by default, and not recommended:
    on real images weighting by it did not improve held-out accuracy.
    `Feature.cov_px` (`Features.cov_px` in Python), `ChessCornersConfig.
    covariance`; failures are
    `RunError::CornerCovariance` (missing or unusable covariance, typically
    JPEG input with estimated noise: give `known`) and
    `RunError::CovarianceUnsupported` (other targets).
  - Next-view planning: `analysis::predict_with_views(export, candidates,
    target_points, image_size)` predicts the covariance with candidate poses
    added, without a re-solve (`PredictedUncertainty`);
    `analysis::rank_candidates` ranks candidates by the largest projection σ
    left and the reduction of `ln det Σ` (`RankedCandidate`);
    `analysis::candidate_poses` builds a deterministic set from a
    `CandidateGrid`. Any camera model; optim `view_information`. Python
    equivalents and the schemas `candidate_grid`, `predicted_uncertainty`,
    `ranked_candidate`.
  - `common::calibrated_cameras` reads any export JSON by `kind`;
    `CalibratedCameras::camera_parameter` and `optim::camera_parameters`
    return a camera parameter by its covariance label.

- **Solver backends.** `factrs` is selectable in every workflow through
  `solver.backend` (Rust `SolverBackend`, JSON `"tiny_solver" | "factrs"`,
  Python `SolverConfig(backend="factrs")`, the app's forms); both run one
  shared Levenberg–Marquardt loop and reach the same minimizer, `tiny_solver`
  stays the default. `RobustLoss::rho` evaluates a loss at a squared
  residual norm.

- **Python reaches the facade.**
  - Sessions: one class per problem type (`PlanarIntrinsicsSession`,
    `RigHandeyeSession`, …) with `set_input`, `set_config`, `step_*` and
    `*_with_seed`, `run_calibration`, `export`, the operation log
    (`LogEntry`), `metadata` and JSON checkpoints (`to_json` / `from_json`).
    Step options are dataclasses at the package root; manual-init seeds and
    step results live in the problem's module
    (`vision_calibration.rig_extrinsics.RigManualInit`). Long steps release
    the GIL.
  - Manifest runs: `calibrate_dataset(manifest, config)`;
    `vision_calibration.dataset_runner` (`build_*_input` converters, run
    results); `sniff_folder`; `FsDetectionCache`, `CacheKey`. Runner
    failures raise `RunError` (Rust variant as `kind`); the ask-user case
    raises `AskUserError` with `field`, `prompt`, `suggestions`. A Python
    object with `name` and `extract(image, spec)` serves as laser-line
    extractor.
  - `detect_target(image, target, detector=None)` runs a manifest target's
    detector on a `(H, W)` `uint8` array and returns `Features`;
    `reject_ambiguous_detection`.
  - `CameraModel`: vectorized `project`, `unproject` and `undistort` over
    numpy arrays for any distortion and sensor.
  - `per_feature_residuals` on every result (`PerFeatureResiduals`, numpy
    tables, unset values NaN, laser lines, per-camera histograms).
  - Manifest types (`DatasetSpec`, `TargetSpec`, `DetectorSpec`, …) as
    `TypedDict`s in `vision_calibration.types`.

- **Rust API.** `dataset_runner::detect_target` detects a manifest's target
  in one in-memory image as a manifest run lowers it (the facade re-exports
  `dataset::DetectorSpec`). Step options and results of every problem type
  derive `Serialize` / `Deserialize`. `vision_calibration::mvg::RansacOptions`
  is re-exported. Per-camera `camera_overrides` and every distortion kind
  with either sensor are available in rig extrinsics and rig hand-eye.
  `synthetic::poses` (`make_iso`, `pose_error` / `PoseError`,
  `tilted_board_poses`, `centered_board_poses` / `BoardPoseSpec`,
  `robot_stations`, `rig_layout`) and `synthetic::laser`
  (`laser_stripe_pixels`, `BoardExtent`) are deterministic seeded generators,
  re-exported by the facade's `synthetic`. Dataset manifests accept the
  `counted4x4` robot-pose format (matrix count, then row-major 4×4
  transforms; the sniffer recognizes it).

- **Regression suite and `calib-bench`.**
  - `regression/` holds image-free observations (plain JSON, checksummed in
    `SHA256SUMS`) of `kuka_1`, the stereo pair and dataset 8 of Tabb and
    Ahmad Yousef's robot-world hand-eye data, with full reference results.
    `calib-bench suite check` recalibrates every entry and fails on drift
    (CI runs it); `--sigma k` compares in units of the reference σ;
    `--backend` reruns on the other backend (CI checks factrs within 1σ).
    Six synthetic entries (planar, Scheimpflug, hand-eye, rig, laserline
    device, joint rig hand-eye laserline) carry ground truth in
    `truth/<id>.json`; `suite check` fails one 5σ or more from it.
    `regression/schema/` holds the JSON Schemas of the file types (generated
    by `cargo xtask emit-schemas`; problem inputs derive `JsonSchema` under
    `schemars`).
  - `calib-bench suite fetch` downloads the public images (`kuka_1`, `ds8`)
    into `target/suite-images/<id>/` and verifies them against
    `regression/fetch/<id>.sha256`; entries needing a login print their
    download page. `suite freeze --images <dir>` re-extracts observations
    from fetched images (byte-identical on Apple-silicon macOS; x86-64 Linux
    corners differ by about 0.001 px), and a weekly macOS job checks it.
  - An OpenCV harness (`tools/compare/opencv`, pinned with `uv`) calibrates
    every entry OpenCV supports with the same parameter set, and
    `calib-bench suite compare` reports differences in units of our σ, both
    libraries' σ, reprojection error and, on synthetic entries, error
    against truth.
  - `calib-bench solver` is a deterministic synthetic benchmark of the
    non-linear solve across the eight problem types (JSON records, Markdown
    tables, `compare` subcommand); `--backend` (also on `run` and `accept`)
    selects the backend.

- **App.**
  - Project folders: `calibration-project.json`, `dataset.json` and
    `runs/run-NNN/` (manifest, config, export, diagnostics) keep a dataset
    and its runs together. The header's **Project** menu creates, opens,
    saves and reopens recent projects; each successful run is saved as the
    next `run-NNN` and listed in a navigator. A folder without
    `calibration-project.json` is reported as not a project.
  - Diagnose has an **Uncertainty** panel: value ± σ per parameter, fixed
    parameters, residual noise, a correlation heatmap and the projection σ
    map.

### Changed

- **Detectors and dependencies.** Detection runs on `calib-targets` 0.16.1
  (`chess-corners` 2.1) and `ringgrid` 0.13.2; the public suite's corners are
  unchanged. ChArUco finds more corners and views on small high-resolution
  boards and resolves ambiguous placements. Use 0.16.1, not 0.16.0 (which
  could label a dark, low-contrast ChArUco frame shifted by whole squares),
  and run `cargo update` on the whole `calib-targets` family. Cached
  detections are keyed on the detection crates' versions too, so upgrading
  recomputes them. The app moves to Tauri 2.12.1.
- **Solver.** The Levenberg–Marquardt loop builds the sparsity pattern and
  symbolic Cholesky analysis once per solve, with results bit-for-bit
  unchanged and roughly 5–9 % faster solves. Each camera model is
  implemented once in `vision-calibration-core` and the optimizer evaluates
  that code under autodiff; laser residuals undistort with the core's Newton
  solver (was a fixed 5- or 10-step iteration), so laser results can move in
  their last digits.
- **Errors.** A result without a covariance is
  `Error::CovarianceUnavailable { reason }` from
  `CalibratedCameras::projection_uncertainty` and the next-view functions
  (was `Error::InvalidInput`); Python still raises `ValueError`.
- **Python payload types are generated** from the Rust JSON Schemas:
  `vision_calibration.types` re-exports `TypedDict`s for every config, input
  and export, the quality report and thresholds, and projection
  uncertainty, under their Rust names. `cargo xtask emit-schemas` writes
  them (and the input, export and quality schemas to `schemas/`; config
  schemas stay in `app/src/schemas/`); `--check` fails when stale. The
  dataclass API is unchanged. Shape changes: a config `TypedDict` requires
  the keys the library requires; exports gain `kind`,
  `per_feature_residuals`, `image_manifest` (rig exports `rig_se3_target`);
  `LaserlineParams` carries `camera`; several fields admit `None`
  (`CameraFix` lists, `IntrinsicsInitConfig.fix_tangential`,
  `ResidualNoise.camera`, `QualityCheck.camera` / `measurement`);
  `RigHandeyeConfig` gains `manual_init`; inputs, cameras, poses and laser
  planes are typed instead of `dict[str, Any]`; an enum with data is a union
  of `<Enum>_<Variant>` `TypedDict`s; fixed-size arrays are `tuple | list`.
  The old spellings (`Transform`, `CameraModel`, `IntrinsicsPayload`,
  `DistortionPayload`, `DistortionParamsPayload`, `ScheimpflugPayload`,
  `LaserPlanePayload`, `DistortionModel`, `ProjectionModel`, `SensorModel`,
  `DistortionBrownConrady5`, `DistortionDivision`, `DistortionRational`,
  `DistortionThinPrism`, `RobustLossHuber`, `RobustLossCauchy`,
  `RobustLossArctan`, `UncertaintyUnavailableReason`) remain as aliases of
  `Iso3`, `CameraParams`, `FxFyCxCy`, `DistortionParams`,
  `ScheimpflugParams`, `LaserPlane`, `DistortionKind`, `ProjectionKind`,
  `SensorKind`, `DistortionParams_<Variant>`, `RobustLoss_<Variant>` and
  `UncertaintyUnavailable`.
- **`calib-bench` datasets.** `run`, `diagnose` and `feature-cov` use the
  regression suite's image entries through their manifests (`--images <dir>`
  reads the fetched cache); `accept` and `basin` read only the `--registry`
  files given. Run records count the expected corners of a chessboard or
  ChArUco target, so detection coverage is no longer zero.
- **Examples.** `planar_real` (Rust and Python) calibrates the shipped
  `data/kuka_1` images and prints each parameter's σ; `stereo_session` and
  `manual_init` (renamed from `manual_init_proof`) read
  `regression/observations/stereo_rig.json`. The
  `stereo_charuco_session`, `dense_stereo_real` and `viewer_fixtures`
  examples are removed (they needed images the repository does not ship).
  The regression suite credits each real-image entry's publisher: `kuka_1`
  from RWHE-Calib (Ali et al., *Sensors* 2019), the stereo pair from the
  Kaggle set *Chessboard Pictures for Stereocamera Calibration* (neither
  states a licence); DS8 is U.S. public domain.
- **Documentation.** The book is the one user guide: the former tutorials
  are merged in, with new chapters (Getting Started in Rust and Python,
  Calibration Targets and Detection, The Dataset Manifest with a field
  reference generated by `cargo xtask emit-schemas`, four Uncertainty
  chapters, Multiple-View Geometry chapters for pose recovery, bundle
  adjustment, rectification and dense stereo). READMEs, book and API docs
  state their rationale inline; the Python README gains an uncertainty
  section.
- **App.** The frame is the shared `@vitavision/workbench` shell (workspace
  rail with `⌘1`…`⌘5`, resizable navigator, status line, notifications) on
  the published `@vitavision/ui` design system with a system / light / dark
  toggle (`calib-theme`). Config forms are `SchemaValueForm` from
  `@vitavision/forms` (labelled groups, units, nullable-block switches,
  inline JSON errors); `cargo xtask emit-schemas` also writes
  `app/src/schemas/<name>.default.json`. Source images load from disk
  through Tauri's asset protocol, only from folders you opened, and the app
  runs under a Content Security Policy. An export whose image root is
  absolute shows its images, and the Run workspace keeps its setup when you
  switch workspaces. Quick-start presets are `kuka_1` and DS8; local presets
  go in an untracked `app/src/workspaces/RunWorkspace/presets.local.ts`.

### Fixed

- **Solver correctness.** SE(3) and quaternion parameters retract through
  tiny-solver's rotation exponential, now exact for small steps and
  sign-preserving beyond 2π (requires tiny-solver 0.18.7 or newer). The
  robot-pose exponential returns its series near zero rotation instead of
  the identity, so corrections have a rotation Jacobian from the first
  iteration (the solve used to stop after one); `apply_robot_delta` applies
  the full exponential including the `V(ω)` translation factor. The
  `SparseQR` linear solver, which solved for the negated step and returned
  the initial parameters, is removed. Robust solves judge each step by
  `Σρ(‖r‖²)` and stop once a step cannot change the parameters, so Huber,
  Cauchy and Arctan reach their minimum (tiny-solver 0.18.3 also corrects
  the Cauchy loss value). Variables have a fixed order, so solves are
  bit-for-bit reproducible, and the non-linear solve is 13–30 % faster.
  The `RobustLoss` docs and book state every loss as `ρ(s)` of the squared
  residual norm.
- **Scheimpflug.** The seeded route lets the tilt move ±0.35 rad from the
  seed (was ±0.10), so a seed off by more no longer pins the tilt on its
  bound with a worse fit and no covariance. A cold-start Scheimpflug camera,
  alone or in a rig, now reaches its principal point under a robust loss:
  the last staged solve runs under the caller's loss and is kept unless it
  raises that objective (it used to be rejected with outliers, leaving the
  seed). Results change for robust-loss cold starts and rig Scheimpflug
  calibrations; they are unchanged without a robust loss.
- **Laser.** The laserline device holds no pose fixed, so a robust loss can
  discount outlying corners and the fit reaches the noise floor.
  `LaserFeatureResidual::residual_px`, the per-camera laser histograms and
  the Diagnose overlay now use the laser solve's line-distance residual
  (they measured the distance to a straight segment 12 cm either side of
  the image centre, which misplaced the line by tens of pixels under strong
  distortion); the overlay is a polyline that follows distortion.
- **Rig hand-eye.** `rig_handeye::step_handeye_optimize` reports the mean
  per-point reprojection error in `HandeyeOptimizeResult::mean_reproj_error`
  (it returned the square root of the final solver cost).
- **Dataset runner.**
  - `detector.min_features_per_view` applies to every topology (rig,
    hand-eye and laser manifests drop views below it, as planar ones do).
  - Where views share one target frame (`single_cam_handeye`, every rig
    topology), a plain chessboard view is kept only when its corners reach
    all four board edges (`cols` across, `rows` down); a partial or
    90°-turned board used to move the target frame (one camera's mean error
    rose from 0.29 to 4.7 px). `planar_intrinsics` and `laserline_device`
    keep partial boards; ChArUco, PuzzleBoard and ring-grid targets are
    unaffected. Every `*RunResult` lists the dropped detections in
    `rejected` (`RejectedDetection`: camera, image,
    `RejectReason::{TooFewFeatures, PartialChessboard, TurnedChessboard}`;
    in Python `dataset_runner.RejectedDetection` with a snake-case reason),
    and a camera with no whole-board view fails with
    `RunError::NoWholeChessboardViews` (Python `RunError`, `kind`
    `"no_whole_chessboard_views"`). A board seen upside down, or a square
    board turned by 90°, cannot be told from upright: capture it the same
    way up in every view, or use a ChArUco or PuzzleBoard target.
- **Detection cache.** Entries are keyed on the `vision-calibration-detect`
  release as well as image, detector and config, so upgrades recompute
  detections (existing caches recompute once). `FsDetectionCache` parses
  floats correctly rounded (serde_json `float_roundtrip`, which applies to
  every JSON read in a build linking `vision-calibration-detect`), so a
  cache hit is bit-identical to the fresh detection.

## [0.9.0] - 2026-09-29

Minor release. The dataset runner's chessboard detector keeps the whole
board, chess-based detectors can use Radon corners, and the laser-rig
workflows are usable from the facade alone. Documentation — READMEs, the
book, tutorials and rustdoc — is brought up to date with the API, and
crate READMEs are now compiled as doctests.

### Migration

- `ChessCornersConfig` (detect) and `ChessCornersDetectorSpec` (dataset)
  gain a `strategy` field. Struct literals must add it or end with
  `..Default::default()`. Manifests and JSON are unaffected: the field is
  optional.

### Fixed

- **The laser-rig workflows are usable from the facade alone.**
  `rig_laserline_device` now re-exports `RigLaserlineDataset` and
  `RigLaserlineView`, and `rig_handeye_laserline` re-exports
  `RigHandeyeLaserlineView`, `RigLaserlineView` and `RobotPoseMeta`, so
  their `*Input` types can be built without depending on
  `vision-calibration-optim` directly.
- **The dataset runner's chessboard detector keeps the whole board.** It
  bounded `calib-targets`' labels by `u < rows` and `v < cols`, but those
  labels run `u` across the image and `v` down it, so a board seen as its
  manifest declares it (`cols` across, `rows` down) lost the corners past
  column `rows`. Both committed chessboard datasets were affected: `kuka_1`
  kept 289 of 476 corners per view, `stereo` 49 of 77. The bounds are now
  `u < cols`, `v < rows`, as the benchmark detector already had them; a
  board seen turned by 90° is still kept whole. Chessboard detection-cache
  keys carry a revision, so entries written before the fix are re-detected
  rather than served.

### Added

- **Radon corner detection for chessboard and ChArUco targets.**
  `ChessCornersDetectorSpec::strategy` (manifest `[detector.chess_corners]`,
  benchmark registry) and `ChessCornersConfig::strategy` select
  `chess-corners`' `"radon"` detector in place of the default `"chess"`.
  Under Radon, `threshold_value` is a fraction in `(0, 1]` of the frame's
  maximum response; the validator rejects values above 1.
- `Detector::output_revision` and `CacheKey::for_detector`: a detector's
  output revision is part of its detection-cache key, so a change to what
  a detector returns invalidates entries written before it.

### Security

- Desktop app: dependency refresh (Tauri 2.12) clears the `unic-*`,
  `anyhow` and yanked `chacha20` advisories. The two left, in the
  Linux-only GTK3 stack Tauri pins (`glib` 0.18, `proc-macro-error`), are
  not reachable from the app and are tracked in `.cargo/audit.toml`.

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

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
