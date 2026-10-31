# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration

- Breaking: robot-pose corrections act in the gripper frame. A view's
  correction is now `base_se3_gripper · exp(δ)` (it was
  `exp(δ) · base_se3_gripper`), so `robot_poses.rot_sigma` and
  `trans_sigma` are the tool's orientation and position errors, and each
  exported `robot_deltas` entry is a correction in the gripper frame. Before,
  a correction turned the gripper about the robot base origin: its
  translation carried the rotation's lever arm, and `trans_sigma` also bounded
  the rotation. Every hand-eye result that refines the robot poses changes. On
  `kuka_1` the largest translation correction drops from 3.7 to 1.5 mm, and
  hand-eye results from disjoint halves of the views agree 39 % better in
  translation (1.8 against 2.9 mm) and 10 % better in rotation; on `ds8`, 8 %
  and 11 % better. On synthetic scenes with the robot's error in its tool
  frame, the hand-eye translation error halves and the stated σ is honest
  when the priors match the robot. With the default 0.5° prior the hand-eye
  rotation's σ follows that prior and is pessimistic. Migration: re-run the
  calibrations. Code that applies `robot_deltas` itself should call
  `apply_robot_delta`, which now multiplies on the right. The rig laserline
  workflows apply an upstream rig's corrections with that function, so with
  the same SE(3) exponential as the solve (before, a rotation then a separate
  translation).

- Breaking: `IntrinsicsInitConfig` loses `fix_tangential`. The linear
  stage now fits only the distortion terms the camera's solves estimate: a
  term the camera's fix list holds (`camera.fix.distortion`, or the model's
  default list) stays at zero from the start, and so do the tangential terms
  while a sensor tilt is estimated. Before, a camera holding `p1`, `p2` with
  `fix_tangential` unset got them fitted in the linear stage and then frozen
  at that value by every later solve; the same held for a camera holding
  `k3` under `fix_k3: false`. Results change only for such configurations,
  and for cameras whose model has no tangential terms (`division1`, `none`),
  whose linear stage no longer fits terms it then drops. Migration: remove
  `fix_tangential` from `init` / `intrinsics` (an unknown field is now a
  configuration error); to hold `p1`, `p2`, list them in
  `camera.fix.distortion`, which now holds them in every stage. `fix_k3`
  stays: it holds `k3` in the linear stage even where the solves estimate
  it. In Python, `IntrinsicsInitConfig(fix_tangential=...)` is removed the
  same way.

- Breaking: `RansacOptions` has no `Default` and no default threshold. The
  old `thresh: 2.0` was a pixel-scale value that silently disabled outlier
  rejection for the estimators measuring on the normalized image plane.
  Build options with `RansacOptions::new(thresh)`; the other fields keep their
  values, so `RansacOptions { max_iters: 500, ..RansacOptions::new(t) }`
  still works. A threshold that is not finite and positive is an
  `InvalidInput` error from every estimator. `thresh` is a distance (never a
  squared one) in each estimator's unit: `dlt_ransac` (PnP) and
  `dlt_homography_ransac` pixels; `fundamental_8point_ransac` the Sampson
  distance in the unit of the input points; `mvg::robust::estimate_essential`
  and `recover_relative_pose_robust` the Sampson distance on the normalized
  image plane (a pixel threshold `t` is `t / f` for a focal length `f` in
  pixels); `mvg::robust::estimate_homography` the symmetric transfer error in
  the unit of the points. Migration: replace `RansacOptions::default()` with
  `RansacOptions::new(t)` using the unit above (`2.0` reproduces the old pixel
  behaviour; use e.g. `1.0e-3` for the essential matrix).

- Breaking: per-point weights are replaced by per-point covariances.
  `CorrespondenceView` loses `weights`, `new_with_weights`, `weight(idx)`
  and `iter_weighted`; a view states an optional covariance per point
  instead (`new_with_covariances(points_3d, points_2d, cov_px)`,
  `covariance(idx)`, `covariances()`; JSON `cov_px`, one
  `{"xx", "xy", "yy"}` in px² per point). A weight `w` on every point of
  the input is the covariance `I / w` (`PointCovariance::new(1.0 / w, 0.0,
  1.0 / w)`); a weight of 1, or no weights, is no covariance. Every view of
  an input states covariances or none does. The view's covariance field is
  private: build views with `CorrespondenceView::new(..)` or
  `new_with_covariances(..)` instead of a struct literal. In Python
  `Observation(weights=...)` becomes `Observation(cov_px=[PointCovariance(xx,
  xy, yy), ...])`.

- Breaking: `ResidualNoise.sigma` becomes `ResidualNoise.level:
  NoiseLevel`, `Sigma(σ̂)` in the residual's unit or `Sigma0(σ̂₀)` for
  residuals whitened by stated point covariances. JSON keeps the `sigma`
  key for the first and writes `sigma0` for the second, so results without
  stated covariances serialize as before. Read `noise.level.value()` where
  the unit is known, or match on the variant. Python keeps
  `ResidualNoise.sigma` and adds `ResidualNoise.whitened`.

- Rig results refine each camera's intrinsics jointly with the rig by
  default: `rig.refine_intrinsics` (rig extrinsics, rig hand-eye) and the new
  `handeye_optimize.refine_intrinsics` (rig hand-eye's final solve) default
  to `true`. The covariance of a rig result then includes the intrinsics and
  their correlation with the extrinsics, and projection uncertainty works for
  rigs. With the intrinsics held, the extrinsics' standard deviations were
  conditional on them: on the synthetic rig the held estimate lay 31σ from the
  truth in those units, the joint one lies within 2.1σ. Results change: on
  the public stereo rig camera 0's focal length moves by 12 px (5σ) and the
  baseline's depth component by 9 mm, and the mean reprojection error drops
  from 0.2544 to 0.2509 px; `ds8` from 0.2961 to 0.2900 px. Set
  `rig.refine_intrinsics` and `handeye_optimize.refine_intrinsics` to
  `false` for the previous results.

- Breaking: one naming scheme across the seven problem types, identical in
  Rust, Python, the JSON config schemas and the app. Each problem module
  exports `<Name>Problem`, `<Name>Input` (a type alias where the data is a
  shared type), `<Name>Config`, its output estimate and `<Name>Export`. A
  problem's output is the final solve's `*Estimate`; step returns are
  `<Stage><Action>Result`; every step takes `Option<<Stage><Action>Options>`
  and its `*_with_seed` variant a `<Stage>ManualInit`; `run_calibration`
  takes only the session in every problem. `handeye` is one word, so the
  CamelCase spelling is `Handeye` everywhere. Config groups pair a stage
  with a full-word action (`handeye_optimize`), and fields no longer repeat
  their group's stage. Python names are the Rust names. Export JSON (`kind`,
  field names) and every numeric result are unchanged; a saved config needs
  the four field renames below. The table also applies to names introduced
  by the other entries of this release.

  | Before | After |
  |---|---|
  | **Configs (JSON and Rust)** | |
  | `rig.refine_intrinsics_in_rig_ba` | `rig.refine_intrinsics` |
  | `init.init_iterations` / `intrinsics.init_iterations` (`IntrinsicsInitConfig::init_iterations`) | `init.iterations` / `intrinsics.iterations` |
  | `RigHandeyeConfig.handeye_ba: HandeyeBaConfig` | `handeye_optimize: HandeyeOptimizeConfig` |
  | `RigHandeyeLaserlineConfig.joint_ba: RigHandeyeLaserlineBaConfig` | `joint_optimize: JointOptimizeConfig` |
  | `LaserlineDeviceOptimizeConfig` (`LaserlineDeviceConfig.optimize`) | `DeviceOptimizeConfig` |
  | **Hand-eye casing** | |
  | `HandEyeMode`, `HandEyeParams`, `HandEyeEstimate`, `HandEyeDataset`, `HandEyeSolveOptions` | `HandeyeMode`, `HandeyeParams`, `HandeyeEstimate`, `HandeyeDataset`, `HandeyeSolveOptions` |
  | `linear::handeye::HandEyeInit` | `HandeyeInit` |
  | `analysis::ReprojLevel::HandEye` (JSON `"hand_eye"`) | `ReprojLevel::Handeye` (JSON `"handeye"`) |
  | **Outputs and inputs** | |
  | `laserline_device::LaserlineDeviceOutput` | `LaserlineDeviceEstimate` |
  | `rig_handeye_laserline::RigHandeyeLaserlineOutput { estimate, handeye_mode }` | the output is `RigHandeyeLaserlineEstimate` itself (`output.estimate.x` → `output.x`); the mode is `config.handeye.handeye_init.handeye_mode` |
  | `planar_intrinsics` input `PlanarDataset` | `PlanarIntrinsicsInput` (an alias of `PlanarDataset`) |
  | (not re-exported) | `single_cam_handeye::{HandeyeEstimate, HandeyeParams}`, `rig_laserline_device::RigLaserlineEstimate`, `rig_handeye_laserline::RigHandeyeLaserlineEstimate` |
  | `dataset_runner::PlanarRunResult.dataset`, `RigRunResult.dataset` | `.input`, as in the other `*RunResult`s |
  | **Step results** | |
  | `planar_intrinsics::PlanarInitResult`, `PlanarOptimizeResult` | `IntrinsicsInitResult`, `IntrinsicsOptimizeResult` |
  | `single_cam_handeye::SingleCamIntrinsicsInitResult`, `SingleCamIntrinsicsOptimizeResult`, `SingleCamHandeyeInitResult`, `SingleCamHandeyeOptimizeResult` | `IntrinsicsInitResult`, `IntrinsicsOptimizeResult`, `HandeyeInitResult`, `HandeyeOptimizeResult` |
  | `laserline_device::LaserlineDeviceInitResult`, `LaserlineDeviceOptimizeResult` | `DeviceInitResult`, `DeviceOptimizeResult` |
  | `rig_extrinsics::RigIntrinsicsInitAllResult`, `RigIntrinsicsOptimizeAllResult` | `IntrinsicsInitResult`, `IntrinsicsOptimizeResult` |
  | `rig_handeye::RigHandeyeIntrinsicsInitAllResult`, `RigHandeyeIntrinsicsOptimizeAllResult`, `RigHandeyeRigInitResult`, `RigHandeyeRigOptimizeResult`, `RigHandeyeHandeyeInitResult`, `RigHandeyeHandeyeOptimizeResult` | `IntrinsicsInitResult`, `IntrinsicsOptimizeResult`, `RigInitResult`, `RigOptimizeResult`, `HandeyeInitResult`, `HandeyeOptimizeResult` |
  | **Manual seeds** | |
  | `planar_intrinsics::PlanarManualInit` | `IntrinsicsManualInit` |
  | `single_cam_handeye::SingleCamIntrinsicsManualInit`, `SingleCamHandeyeManualInit` | `IntrinsicsManualInit`, `HandeyeManualInit` |
  | `laserline_device::LaserlineDeviceManualInit` | `DeviceManualInit` |
  | `rig_extrinsics::RigIntrinsicsManualInit`, `RigExtrinsicsManualInit` | `IntrinsicsManualInit`, `RigManualInit` |
  | `rig_handeye::RigHandeyeIntrinsicsManualInit`, `RigHandeyeRigManualInit`, `RigHandeyeHandeyeManualInit` | `IntrinsicsManualInit`, `RigManualInit`, `HandeyeManualInit` |
  | `rig_laserline_device::RigLaserlineDeviceManualInit` | `PlaneManualInit` |
  | **Step options and signatures** | |
  | `planar_intrinsics::FilterOptions`; `step_filter(s, opts)`, `run_calibration_with_filtering(s, opts)` | `IntrinsicsFilterOptions`; `step_filter(s, Some(opts))`, `run_calibration_with_filtering(s, Some(opts))` (`None` uses the defaults) |
  | `rig_extrinsics` / `rig_handeye` `step_rig_init(s)`, `step_rig_init_with_seed(s, seed)` | `step_rig_init(s, None)`, `step_rig_init_with_seed(s, seed, None)`; options type `RigInitOptions` |
  | `RigOptimizeOptions` (defined in each rig module) | one type, also at `common::RigOptimizeOptions` |
  | `rig_laserline_device::StepOptions` | `PlaneOptimizeOptions` |
  | `rig_laserline_device::step_init(s)`, `step_init_with_seed(s, seed)` | `step_init(s, None)`, `step_init_with_seed(s, seed, None)`; options type `PlaneInitOptions` |
  | `laserline_device::run_calibration(s, Some(config))` | `s.set_config(config)?; run_calibration(s)` |
  | **Python** | |
  | `PlanarCalibrationConfig`, `SingleCamHandeyeCalibrationConfig`, `RigExtrinsicsCalibrationConfig`, `RigHandeyeCalibrationConfig`, `LaserlineDeviceCalibrationConfig`, `RigLaserlineDeviceCalibrationConfig`, `RigHandeyeLaserlineCalibrationConfig` | `PlanarIntrinsicsConfig`, `SingleCamHandeyeConfig`, `RigExtrinsicsConfig`, `RigHandeyeConfig`, `LaserlineDeviceConfig`, `RigLaserlineDeviceConfig`, `RigHandeyeLaserlineConfig` |
  | `PlanarCalibrationResult`, `SingleCamHandeyeResult`, `RigExtrinsicsResult`, `RigHandeyeResult`, `LaserlineDeviceResult`, `RigLaserlineDeviceResult`, `RigHandeyeLaserlineResult` | `PlanarIntrinsicsExport`, `SingleCamHandeyeExport`, `RigExtrinsicsExport`, `RigHandeyeExport`, `LaserlineDeviceExport`, `RigLaserlineDeviceExport`, `RigHandeyeLaserlineExport` |
  | `PlanarDataset`, `SingleCamHandeyeDataset`, `RigExtrinsicsDataset`, `RigHandeyeDataset`, `LaserlineDataset`, `RigHandeyeLaserlineDataset` | `PlanarIntrinsicsInput`, `SingleCamHandeyeInput`, `RigExtrinsicsInput`, `RigHandeyeInput`, `LaserlineDeviceInput`, `RigHandeyeLaserlineInput` |
  | `HandeyeBaConfig`, `RigHandeyeLaserlineBaConfig`, `LaserlineDeviceOptimizeConfig` | `HandeyeOptimizeConfig`, `JointOptimizeConfig`, `DeviceOptimizeConfig` |
  | `RigLaserlineUpstreamCalibration`, `LaserlineEstimateParams`, `LaserlinePlane` | `RigUpstreamCalibration`, `LaserlineParams`, `LaserPlane` |
  | `types.HandEyeMode`, `types.PlanarInput` | `types.HandeyeMode`, `types.PlanarIntrinsicsInput` |
  | `types.CameraModel` (an alias of the `CameraParams` payload) | `types.CameraParams`; `CameraModel` is the camera object, as in Rust |
  | fields `handeye_ba`, `joint_ba`, `refine_intrinsics_in_rig_ba`, `init_iterations` | `handeye_optimize`, `joint_optimize`, `refine_intrinsics`, `iterations` |

- Breaking: single-camera hand-eye and the laserline device take the camera
  model from the config, like every other workflow, and the laser workflows
  evaluate each camera in its own model. `SingleCamHandeyeConfig` and
  `LaserlineDeviceConfig` gain `camera: CameraConfig { model, fix }` (any
  distortion kind, an identity or Scheimpflug sensor; fix lists named per
  slot); their exports carry the camera as `CameraParams`. The rig laser
  workflows (`rig_laserline_device`, `rig_handeye_laserline`) accept every
  distortion kind per camera. The joint rig laser stage names its fixed
  parameters per camera. The Brown-Conrady fix masks are removed.

  | Before | After |
  |---|---|
  | `SingleCamHandeyeExport.camera: PinholeCamera` (JSON `{"k", "dist", "proj", "sensor", "_phantom"}`) | `camera: CameraParams` (JSON `{"projection", "distortion", "sensor", "intrinsics"}`, each slot tagged by `type`) |
  | `SingleCamIntrinsicsInitResult.camera` / `SingleCamIntrinsicsOptimizeResult.camera: PinholeCamera` | `CameraParams` |
  | `SingleCamHandeyeConfig` (Brown-Conrady, untilted) | gains `camera: CameraConfig`; default Brown-Conrady with an identity sensor, as before |
  | `SingleCamIntrinsicsManualInit { intrinsics, distortion, target_poses }` | gains `sensor: Option<ScheimpflugParams>` (a Scheimpflug camera's tilt seed; a non-zero tilt for an untilted camera is a typed error) |
  | `LaserlineDeviceConfig.sensor_init: ScheimpflugParams` | `camera.model.sensor = scheimpflug`, plus the manual-init seed `LaserlineDeviceManualInit.sensor` for the mount tilt (zero when unseeded, as `sensor_init`'s default) |
  | `LaserlineDeviceConfig.optimize.fix_camera: CameraFixMask` | `camera.fix.intrinsics` / `camera.fix.distortion`, e.g. `["cx", "cy"]`, `["k3", "p1", "p2"]` |
  | `LaserlineDeviceConfig.optimize.fix_sensor: bool` (default `true`) | `camera.fix.sensor`; unset (the default) holds the tilt at its seed, as before. An explicit list estimates it (`[]` refines both angles; unseeded, the linear tilt sweep and the staged tilt multi-start estimate it) |
  | `LaserlineDeviceInitResult { intrinsics, distortion, … }` | `{ camera: CameraParams, … }` |
  | `RigHandeyeLaserlineConfig.joint_ba.default_camera_fix: CameraFixMask` + `fix_scheimpflug: ScheimpflugFixMask` | `joint_ba.fix: Vec<CameraFix>`, one per camera; empty (the default) is the stage default: intrinsics and `k1`, `k2` (`lambda`) refined, the higher-order distortion terms and the tilt held — the old default masks |
  | rig laser workflows: Brown-Conrady or no distortion only | any distortion kind per camera |
  | optim `LaserlineParams { intrinsics, distortion: BrownConrady5, sensor: ScheimpflugParams, poses, plane }`; `LaserlineParams::new(k, d, s, poses, plane)` | `LaserlineParams { camera: CameraParams, poses, plane }`; `LaserlineParams::new(camera, poses, plane)` |
  | optim `LaserlineSolveOptions { fix_camera: CameraFixMask, fix_sensor: bool, … }` | `{ fix: CameraFix, … }` (the model's defaults) |
  | core `IntrinsicsFixMask`, `DistortionFixMask`, `CameraFixMask`; optim `ScheimpflugFixMask` | removed; `CameraFix` names the fixed parameters |
  | optim `PlanarIntrinsicsParams::pinhole_camera()` | removed; read `params.camera` (`CameraParams`) |
  | linear `LaserlinePlaneSolver::from_views` over a Brown-Conrady camera | over any `Camera<Real, P, D, Sm, K>`, e.g. a `CameraModel` from `CameraParams::build` |
  | Python `SingleCamHandeyeResult.camera: PinholeBrownConradyCamera`; `LaserlineEstimateParams.intrinsics` / `.distortion` / `.sensor` | `PinholeCamera` or `PinholeScheimpflugCamera`; `LaserlineEstimateParams.camera`. `PinholeBrownConradyCamera` and `CameraFixMask` are removed |
  | Python `LaserlineDeviceCalibrationConfig.sensor_init`, `LaserlineDeviceOptimizeConfig.fix_camera` / `.fix_sensor` | `LaserlineDeviceCalibrationConfig.camera` (`CameraConfig`) |
  | Python `RigHandeyeLaserlineBaConfig.default_camera_fix` / `.fix_scheimpflug` | `.fix: list[CameraFix]` |

  Behaviour notes. An unset `intrinsics.fix_tangential` / `init.fix_tangential`
  of the single-camera hand-eye follows the sensor, as in planar intrinsics
  (`true` for a Scheimpflug sensor), and its Scheimpflug camera without a
  tilt seed gets the planar tilt route: the linear tilt sweep, then the
  staged tilt multi-start on the camera's target views. The laserline device
  keeps its behaviour: a held tilt initializes as an untilted camera with the
  untilted distortion defaults (`k3` held, tangential terms free); only an
  explicitly estimated tilt takes the planar tilt route (and its
  `fix_tangential` and radial-only defaults) in `step_init`, before the plane
  fit. A camera model changed after the camera's init step is a typed
  error. A single-camera hand-eye camera with a Scheimpflug sensor and no
  distortion is a configuration error, as in planar intrinsics. The laserline
  device's untilted camera has no sensor block, so the fixed zero-tilt labels
  (`tilt_x`, `tilt_y`) leave its covariance. The default configurations give
  the same results: the regression suite and solver benchmarks are
  unchanged.

- The shared linear-algebra primitives have one public path,
  `vision_calibration_core::linalg` (`normalize_points_2d/3d`, `null_space`,
  the polynomial root solvers, the SVD reshaping helpers).
  `vision_calibration_linear::math` no longer re-exports them (it keeps
  `ridge_lstsq` and `project_to_so3`), and `vision_geometry::math` is private.
  Import from `vision_calibration_core::linalg` instead.

- Breaking: the `scheimpflug_intrinsics` workflow is merged into
  `planar_intrinsics`. A tilted sensor is a value of the camera model, not a
  separate problem type: planar intrinsics calibrates any distortion model
  with an identity or a Scheimpflug sensor, and its export
  (`PlanarIntrinsicsExport`, `kind: "planar_intrinsics"`) states the
  calibrated sensor in `params.camera.sensor`. The config takes a `camera`
  group, `CameraConfig { model: CameraModelSpec, fix: CameraFix }`, whose
  `fix` lists name the parameters held fixed per slot; a slot left out keeps
  the model's default (`k3` for Brown-Conrady; everything but `k1`, `k2` /
  `lambda` under a Scheimpflug sensor) and an unknown name is a validation
  error. The Scheimpflug behaviour is unchanged and follows the sensor: the
  tangential-free linear stage, the radial-only default fix list, the
  seeded k1 sweep and the cold-start staged multi-start.

  | Before | After |
  |---|---|
  | `ScheimpflugIntrinsicsProblem`, module `scheimpflug_intrinsics` | `PlanarIntrinsicsProblem` with `camera.model.sensor = SensorKind::Scheimpflug` (JSON `"camera": {"model": {"sensor": "scheimpflug", …}}`) |
  | `ScheimpflugIntrinsicsConfig` | `PlanarIntrinsicsConfig` |
  | `distortion_model` | `camera.model.distortion` |
  | `fix_camera.intrinsics` (`IntrinsicsFixMask`) | `camera.fix.intrinsics`, e.g. `["cx", "cy"]` |
  | `fix_camera.distortion` (`DistortionFixMask`) | `camera.fix.distortion`, e.g. `["k3", "p1", "p2"]`; under a Scheimpflug sensor `k4`–`k6` and `s1`–`s4` are named directly instead of following the `k3` / `p1`+`p2` bits |
  | `fix_scheimpflug: { tilt_x, tilt_y }` | `camera.fix.sensor`, e.g. `["tilt_y"]` |
  | `init.fix_tangential: bool` (every config) | `init.fix_tangential: Option<bool>`; unset is the camera's default — `true` for a Scheimpflug camera, `false` otherwise, for each rig camera too |
  | planar `solver.max_iters` default 50 | 120 (sized for the tilt valley; the solver stops at convergence, and every planar regression and benchmark result is unchanged) |
  | `ScheimpflugManualInit` | `PlanarManualInit`, which gains `sensor: Option<ScheimpflugParams>` (a typed error for an untilted camera); its `distortion` seed stays Brown-Conrady and is embedded into the configured model |
  | `ScheimpflugIntrinsicsResult` / `ScheimpflugIntrinsicsExport` / `ScheimpflugIntrinsicsParams` / `…InitResult` / `…OptimizeResult` | `PlanarIntrinsicsEstimate` / `PlanarIntrinsicsExport` / `PlanarIntrinsicsParams` / `PlanarInitResult` (gains `sensor`) / `PlanarOptimizeResult` |
  | `scheimpflug_intrinsics::run_calibration(session, Some(config))` | `session.set_config(config)?; planar_intrinsics::run_calibration(session)` |
  | `ExportKind::ScheimpflugIntrinsics` (`"scheimpflug_intrinsics"`) | `ExportKind::PlanarIntrinsics` |
  | dataset `Topology::ScheimpflugIntrinsics` (`"topology": "scheimpflug_intrinsics"`) | `"planar_intrinsics"`; the manifest describes the data, the config names the sensor |
  | `device_seed::scheimpflug_seed` → `ScheimpflugManualInit` | returns `PlanarManualInit` with the tilt in `sensor` |
  | optim `ScheimpflugIntrinsicsParams`, `ScheimpflugIntrinsicsSolveOptions`, `ScheimpflugIntrinsicsEstimate`, `optimize_scheimpflug_intrinsics` | `PlanarIntrinsicsParams` (a `CameraParams` with a Scheimpflug sensor), `PlanarIntrinsicsSolveOptions`, `PlanarIntrinsicsEstimate`, `optimize_planar_intrinsics` |
  | optim `PlanarIntrinsicsSolveOptions { fix_intrinsics, fix_distortion, … }` | `{ robust_loss, fix: CameraFix, fix_poses, bounds: Option<CameraBounds> }` |
  | optim `ScheimpflugBounds` | `CameraBounds` (tilt bounds need a Scheimpflug sensor) |
  | `optimize_scheimpflug_intrinsics_staged(…, &ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsSolveOptions, …)` | the same name over `&PlanarIntrinsicsParams` and `PlanarIntrinsicsSolveOptions`; a non-Scheimpflug camera is a typed error |
  | optim `params::distortion::{fix_mask_indices, distortion_kind}` | `CameraFix::resolve`, `CameraParams::spec` |
  | facade `vision_calibration::optim::DistortionKind` | `vision_calibration::core::DistortionKind`, beside `CameraModelSpec`, `CameraFix`, `SensorKind` and `ProjectionKind`; `common::config::CameraConfig` is new |
  | Python `run_scheimpflug_intrinsics`, `ScheimpflugIntrinsicsCalibrationConfig`, `ScheimpflugIntrinsicsResult` | `run_planar_intrinsics` with `PlanarCalibrationConfig(camera=CameraConfig(model=CameraModelSpec(sensor="scheimpflug")))`; `PlanarCalibrationResult.camera` is a `PinholeScheimpflugCamera` for a tilted camera |
  | Python `PlanarCalibrationConfig.distortion_model` / `.fix_camera` | `.camera.model.distortion` / `.camera.fix` (`CameraFix`) |

  A planar camera with a general homography sensor, a Scheimpflug sensor
  without distortion, and a sensor change between `step_init` and
  `step_optimize` are typed errors. The regression suite's `synth_scheimpflug`
  entry runs through `planar_intrinsics` with the same data and results; its
  observation file's `problem` tag and its stored result's `kind` read
  `planar_intrinsics`. The benchmark's `scheimpflug_intrinsics` problem is
  the `planar_intrinsics/scheimpflug/…` scenes, and a registry entry selects
  the tilted route with `"problem": "planar_intrinsics", "planar_intrinsics":
  {"sensor": "scheimpflug"}`.

- Breaking: rig extrinsics and rig hand-eye take the camera model from the
  config. `SensorMode` is removed; `RigExtrinsicsConfig` and
  `RigHandeyeConfig` (and so `RigHandeyeLaserlineConfig.handeye`) take
  `camera: CameraConfig` for every camera and `camera_overrides:
  Vec<CameraOverride { camera, model, fix }>` for single cameras. A camera
  index out of range or listed twice, and a fix name the model does not have,
  are validation errors. Outputs and exports carry one `CameraParams` per
  camera, which states the calibrated distortion and sensor.

  | Before | After |
  |---|---|
  | `sensor: SensorMode::Pinhole` / `Scheimpflug { … }`; JSON `"sensor": {"kind": …}` | `camera.model.sensor` = `identity` \| `scheimpflug`; a mixed rig uses `camera_overrides` |
  | `sensor.distortion_model` (Brown-Conrady only) | `camera.model.distortion`: any kind (`none`, `brown_conrady5`, `rational8`, `thin_prism9`, `division1`), per camera |
  | `fix_scheimpflug` | `camera.fix.sensor`, e.g. `["tilt_y"]` |
  | `distortion_mask_in_percam_ba` | `camera.fix.distortion`; a Scheimpflug camera's default holds everything but `k1`, `k2` |
  | `init_tilt_x` / `init_tilt_y` | removed; seed the tilt through manual init `per_cam_sensors` (rig hand-eye `manual_init`, rig extrinsics `RigIntrinsicsManualInit` with `step_intrinsics_init_all_with_seed`). Zero tilts need nothing: the linear tilt sweep already contains zero. A seed is used verbatim and skips the sweep |
  | `refine_scheimpflug_in_rig_ba` | `rig.refine_sensor` |
  | `handeye_ba.refine_scheimpflug` | `handeye_ba.refine_sensor` |
  | `RigExtrinsicsOutput::{Pinhole, Scheimpflug}` | `RigExtrinsicsEstimate { params: RigExtrinsicsParams { cameras, cam_to_rig, rig_from_target }, … }` |
  | `RigHandeyeOutput::{Pinhole, Scheimpflug}` | `HandEyeEstimate { params: HandEyeParams { cameras, cam_to_rig, handeye, target_poses }, … }` |
  | exports' `cameras: Vec<PinholeCamera>` + `sensors` (`RigExtrinsicsExport`, `RigHandeyeExport`, `RigLaserlineDeviceExport`, `RigHandeyeLaserlineExport`) | `cameras: Vec<CameraParams>`; `sensors` is gone |
  | `RigUpstreamCalibration { intrinsics, distortion, sensors, … }` | `RigUpstreamCalibration { cameras, cam_se3_rig, rig_se3_target }` |
  | step results `per_cam_intrinsics` + `per_cam_sensors` | `cameras: Vec<CameraParams>` (`RigIntrinsicsInitAllResult`, `RigIntrinsicsOptimizeAllResult`, `RigOptimizeResult`, and the rig hand-eye equivalents) |
  | optim `optimize_rig_extrinsics_scheimpflug`, `optimize_handeye_scheimpflug` and their `*Scheimpflug*` params, options and estimates | removed; `optimize_rig_extrinsics` and `optimize_handeye` take any camera model per camera |
  | optim `RigExtrinsicsSolveOptions` / `HandEyeSolveOptions` `default_fix`, `camera_overrides`, `*scheimpflug*` | `fix: Vec<CameraFix>`, one per camera (empty = the models' defaults) |
  | optim `RigHandeyeLaserlineParams.sensors` | removed; `cameras: Vec<CameraParams>` carries the sensors |
  | optim `RigHandeyeLaserlineSolveOptions` `fix_intrinsics` / `fix_scheimpflug` | `fix: Vec<CameraFix>` |
  | optim `RigLaserlineUpstream { intrinsics, distortion, sensors, … }`; `compute_rig_laserline_feature_residuals(…, intrinsics, distortion, …)` | `RigLaserlineUpstream { cameras, … }`; `…(dataset, cameras: &[CameraParams], cam_se3_rig, rig_se3_target, planes)` |
  | optim `evaluate_rig_handeye_laserline(…)` | returns a `Result` |
  | Python rig configs `sensor=` (`PinholeSensorMode`, `ScheimpflugSensorMode`, `SensorMode`) | `camera=CameraConfig(…)` and `camera_overrides=[…]`; the `SensorMode` classes are removed |

  Behaviour notes. `rig.refine_intrinsics_in_rig_ba` holds each camera's fix
  lists in the rig bundle adjustment (it held only `k3` of every camera;
  identical for the default Brown-Conrady camera). A non-zero tilt seed for
  an untilted camera is an `InvalidInput` error (it was ignored). The
  joint rig laser bundle uses each camera's own model: an untilted
  camera has no sensor block, so the fixed zero-tilt labels of its
  covariance (`cam*.tilt_*`) are gone. The regression suite and solver
  benchmarks are unchanged.

- The camera-model kinds live in `vision-calibration-core` with their
  parameter metadata: `ProjectionKind`, `DistortionKind`, `SensorKind` and
  `CameraModelSpec` (re-exported from `vision_calibration_optim::ir`).
  `CameraModelDesc` is renamed `CameraModelSpec`; its combination constants
  (`PINHOLE4_DIST5`, …) are replaced by
  `CameraModelSpec::pinhole(distortion, sensor)`. `SensorKind::None` /
  `Scheimpflug2` are renamed `Identity` / `Scheimpflug`. `CameraParams::spec()`
  returns a parameter set's spec.

- Breaking: camera intrinsics are exactly `fx, fy, cx, cy`; skew is gone.
  `FxFyCxCySkew` is renamed `FxFyCxCy` and loses its `skew` field (its
  `k_matrix()` has a zero in the skew slot). `IntrinsicsParams` and
  `AnyIntrinsics` are removed: `CameraParams::intrinsics` is the plain
  `FxFyCxCy<Real>`, and `CameraModel` is
  `Camera<Real, AnyProjection, AnyDistortion, AnySensor, FxFyCxCy<Real>>`.
  `zero_skew` is removed from `IntrinsicsInitConfig` (and the Python
  `IntrinsicsInitConfig`), `IterativeIntrinsicsOptions` and
  `ScheimpflugIntrinsicsInitOptions`; Zhang's initialization always drops the
  skew of its closed-form solve, as `zero_skew: true` did. `pack_intrinsics`
  no longer returns a `Result`. Migration: rename `FxFyCxCySkew` to
  `FxFyCxCy`, delete `skew:` initializers and `.skew` reads, replace
  `IntrinsicsParams::FxFyCxCySkew { params }` with the struct itself, and
  drop `zero_skew` from configs. JSON intrinsics objects lose `skew` and the
  `"type": "fx_fy_cx_cy_skew"` tag (`{"fx": …, "fy": …, "cx": …, "cy": …}`);
  the config schemas, the Python `PinholeIntrinsics` (no `skew`) and the
  regression and benchmark records follow.

- Camera projection takes the camera-frame point and back-projects to a
  ray. `ProjectionModel` is now `in_domain` / `project` / `unproject` /
  `is_central` (was `project_dir` / `unproject_dir`), and
  `Camera::backproject_pixel` returns `Ray { origin, direction }` (was
  `Ray { point }` on the `z = 1` plane). Replace `ray.point` with
  `ray.point_at_unit_depth()` (an `Option`; always `Some` for a pinhole
  camera) or use `ray.direction`, which for a pinhole camera is the same
  point. The distortion, sensor and projection traits drop their `Copy`
  bound (`T: RealField`), `HomographySensor` is generic over the scalar, and
  `ScheimpflugParams<T = f64>` is a sensor model in its own right.

- The Scheimpflug intrinsics defaults (`ScheimpflugIntrinsicsConfig::fix_poses`,
  `ScheimpflugIntrinsicsSolveOptions::fix_poses`, Python
  `ScheimpflugIntrinsicsCalibrationConfig.fix_poses`) no longer hold view 0
  fixed: the board fixes the target frame, so no pose needs to be, and a
  fixed pose would make the reported uncertainty conditional on it. Results
  move slightly; pass `fix_poses=[0]` for the previous behaviour. The
  laserline device defaults (`LaserlineDeviceOptimizeConfig::fix_poses`,
  `LaserlineSolveOptions::fix_poses`, Python
  `LaserlineDeviceOptimizeConfig.fix_poses`) likewise hold no pose fixed.

- Python: `ScheimpflugSensor.from_payload` reads only `tilt_x` / `tilt_y`;
  the `tau_x` / `tau_y` spellings, which no export emits, are not accepted.
- `SolveReport.final_cost` is the robust objective `½ Σ ρ(‖rᵢ‖²)` at the
  solution (`½‖r‖²` without a loss). For robust solves the reported value
  changes and is not comparable with earlier releases.
- `BackendSolveOptions` has a `backend` field (default
  `SolverBackend::TinySolver`). Struct literals that list every field add it
  or end with `..BackendSolveOptions::default()`.
- `optimize_planar_intrinsics_with_backend` and
  `optimize_scheimpflug_intrinsics_with_backend` are removed:
  `optimize_planar_intrinsics` / `optimize_scheimpflug_intrinsics` take the
  backend from `BackendSolveOptions::backend`.
- `PlanarIntrinsicsConfig::backend_opts()` and
  `LaserlineDeviceConfig::backend_opts()` are removed; use
  `config.solver.backend_options()`, which carries the backend too.
- `SolverConfig` has a `backend` field (JSON `"backend": "tiny_solver" |
  "factrs"`, default `tiny_solver`). Existing configs load unchanged.
- `BackendSolveOptions::linear_solver` and `LinearSolverKind` are removed:
  the Levenberg–Marquardt loop always solves with sparse Cholesky. Drop the
  field from struct literals; JSON that still carries `"linear_solver"`
  loads unchanged, and the field is ignored.
- **The laser robust loss defaults to `Huber { scale: 1.0 }`** in
  `LaserlineDeviceOptimizeConfig`, `LaserlineSolveOptions` and Python's
  `LaserlineDeviceOptimizeConfig`, instead of `0.01`. A loss scale is in its
  residual's unit, and the default laser residual is the line distance in
  pixels, where 0.01 px made the fit nearly L1: less precise, with a
  covariance that understated the scatter. Laser device and rig laser-plane
  results change slightly; set `laser_loss: {"Huber": {"scale": 0.01}}` to
  keep the previous fit, or a scale in meters with the point-to-plane
  residual.
- `LaserFeatureResidual::projected_line_px` is removed. Each view's projected
  laser line is now `PerFeatureResiduals::laser_lines` (one `LaserLineTrace`
  per view and camera, a polyline in `points_px`), and
  `compute_laserline_feature_residuals` /
  `compute_rig_laserline_feature_residuals` return `LaserResidualRecords`
  (`features`, `lines`) instead of the record list.
- A `single_cam_handeye` or rig manifest with a plain chessboard target
  whose views of one camera are all partial or turned by 90° now fails with
  the new `RunError::NoWholeChessboardViews` instead of calibrating a moved
  target frame; capture the whole board, or switch to a ChArUco or
  PuzzleBoard target. Add an arm for the variant to exhaustive matches on
  `RunError`. `detector.min_features_per_view = rows × cols` is no longer
  needed to keep only complete boards and can be removed.
- **Robot-pose priors are weighed against the corner noise, and every view's
  robot pose is refined.** Hand-eye results with `robot_poses.refine` (the
  default in `single_cam_handeye`, `rig_handeye` and
  `rig_handeye_laserline`) change. The priors were weighed against the
  corner residuals as if those had 1 px of noise, so with sub-pixel corners
  they held the robot poses too tightly and the robot's error leaked into
  the camera parameters; and view 0's correction was fixed, which took that
  robot pose as exact. Now `rot_sigma` and `trans_sigma` are the standard
  deviations of the robot's pose error, the solve estimates the corner noise
  (one solve at 1 px, then one from that solution at the noise its
  residuals show) and weighs the priors against it, and every view gets a
  correction. On synthetic eye-in-hand scenes the hand-eye errors shrink by
  up to 3× and the standard deviations become honest (normalized errors
  with an RMS near 1, against 2.2–2.6 before). On `kuka_1`, σ(fx) is 0.31 px
  instead of 0.95 (OpenCV, whose intrinsics have a free pose per view:
  0.33), the hand-eye translation σ 0.59–0.74 mm instead of 0.79–1.09, and
  the mean reprojection error 0.1346 px instead of 0.1359; on `ds8` (a rig
  hand-eye with the intrinsics refined) 0.2574 px instead of 0.2900. The target pose
  in the robot base frame gets the standard deviation the robot's rotation
  error gives it at its distance from the base: 4.5 mm instead of 0.8–1.1 mm
  on `kuka_1`, whose target is 2.8 m from the base. No config field
  changes. Set `rot_sigma` and `trans_sigma` to your robot's accuracy: the
  covariance takes them as true, and the defaults (0.5°, 1 mm) are loose;
  `refine: false` takes the robot poses as exact, as before.
  `HandeyeSolveOptions` and `RigHandeyeLaserlineSolveOptions` (optim) behave
  the same way; their results' `robot_deltas` now have a non-zero first
  entry, and their covariance a block for view 0's correction.
- **A tilted (Scheimpflug) camera's own solve reaches its minimum in every
  workflow.** Results change for Scheimpflug cameras in `rig_extrinsics`
  and `rig_handeye`, and for the per-camera tilt solve of
  `single_cam_handeye` and of an unseeded `laserline_device`. Two
  defaults held that solve off the minimum. The rig workflows' linear stage
  fitted the tangential terms `p1`, `p2` of every camera, while a
  Scheimpflug camera's default fix list holds them in every later solve:
  the fitted value, a decentering that mimics a tilt, stayed frozen and
  moved the tilt and principal point. And the camera's own solve held view
  0's target pose at its linear-init value. An unset
  `intrinsics.fix_tangential` now holds the tangential terms of a
  Scheimpflug camera in every workflow and for each rig camera (it was
  `false` for every rig camera), and the camera's own solve frees every
  pose. On the synthetic two-camera Scheimpflug rig of `calib-bench
  solver` at 0.1 px noise, the reprojection RMS falls from 0.17–0.19 px to
  the noise floor (0.141 px), the principal-point error from 28–44 px to
  0.4–2.1 px, the tilt error from 1.5–2.5° to 0.03–0.15° and the rig
  rotation error from 2–3° to 0.06–0.15°; at 0.5 px noise every error is
  within 1.5 standard deviations. Set `intrinsics.fix_tangential: false`
  to fit the tangential terms in the linear stage as before (free them in
  `camera.fix` too, or the fitted value stays frozen).
  `IntrinsicsInitConfig::iterative_opts` and `distortion_fit_opts` take
  the camera's `CameraModelSpec` instead of a `bool` default.

### Added

- **Corner covariance from the chessboard and ChArUco detectors.** Setting
  `[detector.chess_corners.covariance]` in a manifest (`image_noise =
  "estimate"`, the default, or `{ known = σ }` in gray levels) makes the
  detector report each corner's position covariance, which the dataset
  runner attaches to the observations: residuals are whitened and the
  report gives σ̂₀ instead of σ̂ in pixels. Off by default. `Feature` gains
  `cov_px` (`[xx, xy, yy]`, px²; `Features.cov_px` in Python, carried by
  `to_observation()` and the detection cache), `ChessCornersConfig` gains
  `covariance`, and the runner fails with `RunError::CornerCovariance` when
  a requested covariance is missing or unusable on an image (typically JPEG
  input with the estimated noise: give `known`), and with
  `RunError::CovarianceUnsupported` for other targets.
- **Projection covariance at given pixels.**
  `CalibratedCameras::projection_covariance(camera, pixels)` (implemented by
  every `*Export`) and `vision_calibration_optim::projection_covariance_at`
  return the full 2 × 2 covariance, in px², of where the camera-frame ray
  through each pixel lands, or `None` where no ray reaches the pixel. Its
  major-axis σ is what `projection_uncertainty` reports, which now computes
  through it, so its values are unchanged. Rust only.
- **App: project folders.** A project keeps a dataset's manifest and its
  runs in one folder (`calibration-project.json`, `dataset.json`,
  `runs/run-NNN/` with the run's manifest, config, export and diagnostics).
  The header's **Project** menu creates one from the loaded dataset, opens
  one, saves the edited manifest and reopens recent projects; the dataset's
  images may live in the project folder or anywhere else. While a project is
  open, each successful run is saved as the next `run-NNN` (never over an
  existing one) and reopened from disk, and a navigator on the left lists
  the runs. A folder without `calibration-project.json` is reported as not a
  project rather than guessed at.
- **Per-point pixel covariance.** An observation may state the 2 × 2
  covariance of each image point (`PointCovariance`, px²). The solve then
  whitens each reprojection residual by it (`L e` with `LᵀL = Σ⁻¹`), in both
  solver backends, so uncertain points count less and elongated ones are
  trusted across their long axis. The reprojection noise is reported as
  `σ̂₀` (JSON `sigma0`), the dimensionless a-posteriori standard deviation of
  unit weight, about 1 when the stated covariances hold, and the parameter
  covariance is scaled by it. The quality report's variance-factor check
  tests `σ̂₀²` against 1 without `expected_feature_noise_px`, and its
  view-outlier check reads each target feature's new `error_whitened`
  (`√(eᵀ Σ⁻¹ e)`); reprojection errors stay in pixels. A robust loss's
  scale applies to the whitened residual, in units of each point's σ. An
  input that states covariances for some views and not others is rejected
  (`Error::MixedCovariance`). Without stated covariances every result is
  unchanged. On synthetic scenes with covariances of 0.1–1 px and up to 4:1
  anisotropy, `σ̂₀` was 0.98 ± 0.03, the predicted σ matched the
  Monte-Carlo scatter within 4 %, and the errors were 2.5–3 times smaller
  than with every point weighed alike. Python: `PointCovariance`,
  `Observation.cov_px`, `ResidualNoise.whitened`,
  `TargetFeatureResiduals.error_whitened`.

- **Planning the next view.** `analysis::predict_with_views(export,
  candidates, target_points, image_size)` predicts the covariance a planar
  intrinsics calibration would have with candidate target poses added,
  without a re-solve: each view's information about the camera, its own
  pose marginalized, is added to the result's on the pixel-noise scale the
  result's covariance implies (`effective_noise_px`). It returns a
  `PredictedUncertainty` (the predicted `ParameterCovariance`, its
  projection uncertainty over the image, the points each candidate sees).
  `analysis::rank_candidates` scores candidates one at a time by the largest
  projection σ they would leave and the reduction of `ln det Σ`
  (`RankedCandidate`, best first, ties in candidate order), and
  `analysis::candidate_poses` generates a deterministic candidate set from a
  `CandidateGrid` (image regions × tilts at one distance). Any camera model
  the solver estimates. On synthetic scenes the prediction matches a
  re-solve with the view added to within 0.1–2.1 % on average. Python:
  `predict_with_views`, `rank_candidates`, `candidate_poses`,
  `CandidateGrid`, `PredictedUncertainty`, `RankedCandidate`; schemas
  `candidate_grid`, `predicted_uncertainty`, `ranked_candidate`. In optim,
  `view_information` gives one view's information about a camera's
  parameters.
- **Python reaches the facade.** The package now depends on `numpy`.
  - Sessions: one class per problem type (`PlanarIntrinsicsSession`,
    `RigHandeyeSession`, ...) with `set_input`, `set_config`, the problem's
    `step_*` methods and their `*_with_seed` variants, `run_calibration`,
    `export`, the operation log (`LogEntry`), `metadata` and JSON
    checkpoints (`to_json` / `from_json`). Step options are dataclasses at
    the package root (`IntrinsicsInitOptions`, `RigOptimizeOptions`, ...);
    each problem's manual-init seeds and step results are dataclasses in the
    problem's module, which mirrors the Rust module
    (`vision_calibration.rig_extrinsics.RigManualInit`). Long steps release
    the GIL.
  - Manifest runs: `calibrate_dataset(manifest, config)` runs a TOML or JSON
    manifest end to end; `vision_calibration.dataset_runner` has the
    `build_*_input` converters and their run results; `sniff_folder` drafts
    a manifest; `FsDetectionCache` and `CacheKey` bind the detection cache.
    Runner failures raise `RunError` with the Rust variant as `kind`; the
    ask-user ambiguity raises its subclass `AskUserError` with `field`,
    `prompt` and `suggestions`. A Python object with `name` and
    `extract(image, spec)` serves as the laser-line extractor of the laser
    topologies.
  - Detection: `detect_target(image, target, detector=None)` runs a
    manifest target's detector on a `(H, W)` `uint8` numpy image and returns
    `Features` (`image_xy`, `world_xyz` arrays); `reject_ambiguous_detection`.
  - `CameraModel`: a camera built from a result's camera or a `CameraParams`
    payload (any distortion, identity or Scheimpflug sensor) with vectorized
    `project`, `unproject` (ray origins and directions) and `undistort` (to a
    pinhole image with the same intrinsics) over numpy arrays.
  - Every result's `per_feature_residuals`: `PerFeatureResiduals`, with the
    target and laser records as numpy tables (`TargetFeatureResiduals`,
    `LaserFeatureResiduals`; unset values are NaN), the projected laser
    lines and the per-camera histograms.
  - The manifest types (`DatasetSpec`, `TargetSpec`, `DetectorSpec`, ...) are
    generated payload `TypedDict`s in `vision_calibration.types`.
- `vision_calibration::dataset_runner::detect_target` detects a manifest's
  target (and optional `detector` overrides) in one in-memory image, lowered
  exactly as a manifest run lowers it; the facade re-exports
  `dataset::DetectorSpec` for it.
- The step options and step results of every problem type derive
  `Serialize` and `Deserialize`.

- `vision_calibration::mvg::RansacOptions`: the facade re-exports the RANSAC
  settings the robust two-view estimators take, so MVG users no longer
  depend on `vision-calibration-core` for them.

- Per-camera `camera_overrides` in the rig extrinsics and rig hand-eye
  configs (mixed lenses and sensors in one rig), and every distortion kind
  (`none`, `brown_conrady5`, `rational8`, `thin_prism9`, `division1`) with
  either sensor in both workflows, per camera.

- **factrs solver backend.** Every workflow can solve with `factrs` instead
  of `tiny-solver` through `solver.backend` (Rust `SolverBackend`, JSON,
  Python `SolverConfig(backend="factrs")`, the app's config forms). Both
  backends run one shared Levenberg–Marquardt loop over the same residual
  kernels and reach the same minimizer; factrs builds Jacobians with
  static-size dual numbers and applies a robust loss by iterative
  reweighting, tiny-solver with dynamic-size dual numbers and the Triggs
  correction. `tiny-solver` stays the default.
- **`RobustLoss::rho`** evaluates a loss at a squared residual norm, the
  definition both backends use.
- **Synthetic scene generators in `vision_calibration_core::synthetic`.**
  `synthetic::poses` (`make_iso`, `pose_error` / `PoseError`,
  `tilted_board_poses`, `centered_board_poses` over `BoardPoseSpec`, seeded
  tilt and board-pose generators, `robot_stations`, `rig_layout`) and
  `synthetic::laser` (`laser_stripe_pixels`, `BoardExtent`: the laser
  plane's intersection with a planar target, projected through any camera
  model). Both are re-exported by the facade's `synthetic` module. Every
  generator is deterministic for a given seed.
- **`calib-bench solver`.** A deterministic synthetic benchmark of the
  non-linear solve across the eight problem types, with JSON records,
  Markdown tables and a `compare` subcommand. `--backend` picks the solver
  backend for every scene, so `compare` doubles as a backend comparison;
  `calib-bench run` and `accept` take `--backend` too.
- **Parameter uncertainty in every result.**
  `PlanarIntrinsicsExport`, `ScheimpflugIntrinsicsExport`,
  `SingleCamHandeyeExport`, `RigExtrinsicsExport`, `RigHandeyeExport`,
  `RigLaserlineDeviceExport` and `RigHandeyeLaserlineExport` (and their
  optimizer estimates) carry `uncertainty`; the laserline device result
  carries it in `estimate.uncertainty`. It is the covariance of the
  estimated parameters by label (`fx`, `k1`, `tilt_x`, `cam1.fx`,
  `cam1.cam_se3_rig.tx`, `gripper_se3_rig.rz`, `plane.nx` … `plane.d`, …) with `std_dev` and
  `correlation`, one block per view for the board or rig poses and the
  robot-pose corrections, the fixed parameters it is conditional on, and a
  robust estimate of the noise of each kind of residual (corners, laser;
  per camera for the rig laserline, whose planes are solved apart). When the data
  does not determine a parameter the result says which
  (`rank_deficient`), and likewise for a parameter on a bound or a solve
  that ran out of iterations. Robust losses and robot-pose priors are
  accounted for. Python:
  `result.uncertainty` (`ParameterUncertainty`). See the book chapter
  "Parameter Uncertainty".
- **Projection uncertainty.** `CalibratedCameras::projection_uncertainty`
  (implemented by every `*Export`) and `optim::projection_uncertainty` map
  how precisely a calibrated camera places a camera-frame ray in its image:
  the major-axis σ in pixels on a 16 × 12 grid across the image, with the
  centre, the largest corner and the largest value. Python:
  `vision_calibration.projection_uncertainty(result, (width, height),
  camera=0)` returns a `ProjectionUncertainty`; every result keeps the
  export it was read from as `result.export`.
- **Quality verdict.** `analysis::assess(&export, image_sizes,
  &QualityThresholds)` judges any calibration export and returns a
  `QualityReport`: a `pass` / `warn` / `fail` verdict and typed checks, each
  with its status, measured value, unit, threshold and a message. The
  checks: observability (fails on `rank_deficient` or no redundancy, naming
  the parameters; warns on a parameter at a bound), convergence, a χ² test
  of the reprojection noise against a stated feature noise
  (`expected_feature_noise_px`, which has no default), the largest
  projection σ over each camera's image, the share of each image the
  features reached (8 × 6 cells), views whose mean residual exceeds 3 σ̂,
  and camera parameters that correlate above 0.99. A check whose inputs the
  result lacks — an image size, a stated noise, intrinsics the final solve
  held fixed — is `not_applicable` and does not count. Thresholds are
  JSON-serializable and reject unknown fields. `CalibratedCameras` also
  gives an export's feature residuals and image manifest. Python:
  `vision_calibration.assess(result, thresholds=None, image_sizes=None)`
  returns a `QualityReport`. `calib-bench accept` fails an entry whose
  report fails and prints its warnings; a registry entry can set
  `accept.quality`. See "Judging a Result" in the book.
- **Uncertainty in the desktop app.** Diagnose has an **Uncertainty** panel:
  each estimated parameter's value ± σ, the fixed parameters and residual
  noise, a correlation heatmap, and the shown camera's projection σ map.
  `common::calibrated_cameras` reads any export JSON by its `kind`, and
  `CalibratedCameras::camera_parameter` and `optim::camera_parameters` give
  a camera parameter's value by its covariance label.
- **Public regression suite** in `regression/`: image-free observations of
  `kuka_1`, the stereo pair (each camera and the rig) and dataset 8 of Tabb
  and Ahmad Yousef's robot-world hand-eye data, checksummed in a standard
  `SHA256SUMS` file, with the full reference result of each.
  `calib-bench suite check` recalibrates every entry and fails on any drift;
  CI runs it on every change. The observations are plain JSON (the problem
  inputs plus image sizes), so other tools can run on the same data.
  `suite check --sigma k` compares the estimated parameters in units of the
  reference's standard deviations, and `--backend` reruns the suite on the
  other solver backend; CI checks that factrs agrees with the references
  within 1σ. Six synthetic entries (planar, Scheimpflug, hand-eye, rig,
  laserline device, joint rig hand-eye laserline) carry their ground truth
  in `truth/<id>.json`, and `suite check` fails one whose parameters lie
  5σ or more from the truth. An OpenCV harness (`tools/compare/opencv`,
  pinned with `uv`) calibrates every entry OpenCV has a method for, with
  the parameter set calibration-rs estimates (on the Scheimpflug entry the
  tangential terms are held at zero, as they alias with the tilt), and
  `calib-bench suite compare` sets its results against the references:
  differences in units of our σ, the two libraries' standard deviations,
  reprojection error and, on synthetic entries, each library's error
  against the truth. `regression/schema/` holds JSON Schemas of the
  observation, truth and external-result files, generated by `cargo xtask
  emit-schemas`; the problem inputs (views, datasets, the hand-eye and laser
  metadata, the rig laserline input) derive `JsonSchema` under the
  `schemars` feature. See the book chapter "Regression Suite".
- **`calib-bench suite fetch`** downloads the regression suite's public
  images from their publishers (`kuka_1` from the RWHE-Calib repository at
  a pinned commit, `ds8` as `DS8.zip` from the Ag Data Commons) into
  `target/suite-images/<id>/` and verifies every file against the entry's
  `regression/fetch/<id>.sha256`; files already there and verified are not
  downloaded again. Entries whose publisher needs a login (the Kaggle
  stereo set) print their download page instead. `suite freeze --images
  <dir>` re-extracts those entries' observations from the fetched images,
  so a checkout without the image files reproduces `SHA256SUMS` (byte for
  byte on Apple-silicon macOS, where the observations are frozen; corners
  detected on x86-64 Linux differ by up to about 0.001 px). A weekly CI
  job on macOS does exactly that. The `ds8` manifest reads the images as
  published (RGB TIFF and colour JPEG, no grayscale PNG conversion), and
  its observations and reference result are frozen from them: its largest
  parameter change is 0.59σ.
- **`counted4x4` robot-pose format** for dataset manifests: the matrix
  count, then that many row-major 4×4 transforms on any line layout — the
  layout of the robot-world hand-eye datasets. The folder sniffer
  recognizes it.

### Changed

- Hand-eye solves estimate the robot's pose error. With robot-pose
  refinement, `robot_poses.estimate_sigma` (new, on by default) treats
  `rot_sigma` and `trans_sigma` as starting values: each prior's variance
  factor is estimated from the corrections and their redundancy, and the
  solve repeats until it settles. Exports carry the result as
  `robot_pose_sigma` (`RobotPoseSigma`: `rot_sigma` in radians,
  `trans_sigma` in metres), in Rust and Python. On synthetic scenes it
  recovers the simulated robot error from a prior ten times too tight or too
  loose. With the default prior the hand-eye standard deviations were about
  2.5 times too large; from a too-tight prior they were about 8 times too
  small, and the hand-eye translation error halves. Turn it off to take the
  configured standard deviations as the robot's true accuracy.

- The public datasets of `calib-bench run`, `diagnose` and `feature-cov` are
  the regression suite's image entries, read through their dataset
  manifests; `--images <dir>` reads the fetchable ones from the `suite
  fetch` cache. `calib-bench accept` and `basin` read only the `--registry`
  files they are given: the public datasets are checked by `suite check`.
  Run records now count the expected corners of a chessboard or ChArUco
  manifest target, so their detection coverage is no longer zero.

- The detector's per-corner covariance (`detector.chess_corners.covariance`)
  is not recommended for calibration. On real chessboard and ChArUco images,
  weighting by it did not improve held-out accuracy, made it slightly worse
  on some cameras, and left the stated uncertainty as far from the real
  scatter as before. It stays off by default. The book's uncertainty limits
  now state how much larger than the predicted σ the real split-half
  scatter is.

- **Detectors run on `calib-targets` 0.16.1 (`chess-corners` 2.1) and
  `ringgrid` 0.13.2.** On the public regression suite every corner is
  unchanged. ChArUco detection finds more corners and views on small,
  high-resolution boards: a corner re-detected near its marker must agree with
  the board instead of being dropped when its saddle fit fails, and frames
  whose board placement was ambiguous are now resolved. Use 0.16.1 rather than
  0.16.0, which could label a dark, low-contrast ChArUco frame with the whole
  board shifted by whole squares; after upgrading, run `cargo update` on the
  whole `calib-targets` family, since updating the `calib-targets` crate alone
  leaves its detector sub-crates at their old version. Cached detections are
  keyed on the detection crates' versions as well as this crate's, so
  upgrading them recomputes them.
- **App: the workbench shell.** The app's frame is the shared
  `@vitavision/workbench` studio shell (workspace rail with `⌘1`…`⌘5`,
  resizable project navigator, status line, notifications), on current
  `@vitavision/*` packages (ui 0.14, workbench 0.3, three and three-react
  0.2, stage2d 0.13, forms and charts 0.8). Source images load straight from
  disk through Tauri's asset protocol instead of as base64 text over IPC,
  and only from folders you opened (a project's dataset, an export's
  images); the app now runs under a Content Security Policy. An export whose
  image root is an absolute path shows its images. The Run workspace keeps
  its setup when you switch workspaces.
- A result without a covariance is a typed
  `Error::CovarianceUnavailable { reason }` from
  `CalibratedCameras::projection_uncertainty` and the next-view functions,
  in place of `Error::InvalidInput` with the reason in its message. Python
  still raises `ValueError`.
- **The Python payload types are generated from the Rust types.**
  `vision_calibration.types` re-exports `TypedDict`s that
  `cargo xtask emit-schemas` renders from the JSON Schemas of every config,
  input and export, the quality report and thresholds, and the projection
  uncertainty, under their Rust names; `emit-schemas --check`, which CI runs,
  fails when they are stale. The input, export and quality schemas are now
  written too, to `schemas/` (the config schemas stay in `app/src/schemas/`).
  The dataclass API is unchanged. The hand-written types had drifted; the
  shapes that change:
  - a config `TypedDict` requires the keys the library requires — a partial
    config was always rejected;
  - the exports gain `kind`, `per_feature_residuals` and `image_manifest`, the
    rig exports `rig_se3_target`;
  - `LaserlineParams` carries `camera` (a `CameraParams`) in place of
    `intrinsics`, `distortion` and `sensor`;
  - `CameraFix`'s lists, `IntrinsicsInitConfig.fix_tangential`,
    `ResidualNoise.camera` and `QualityCheck.camera`/`measurement` admit `None`;
    `RigHandeyeConfig` gains `manual_init`;
  - inputs, cameras (`CameraParams`), poses (`Iso3`) and laser planes are typed
    instead of `dict[str, Any]`;
  - an enum with data (`Uncertainty`, `DistortionParams`, `SensorParams`,
    `RobustLoss`, …) is a union of one `TypedDict` per variant, named
    `<Enum>_<Variant>`;
  - a fixed-size array is `tuple[...] | list[...]`: the extension returns a
    tuple and reads either.

  The Python spellings `Transform`, `CameraModel`, `IntrinsicsPayload`,
  `DistortionPayload`, `DistortionParamsPayload`, `ScheimpflugPayload`,
  `LaserPlanePayload`, `DistortionModel`, `ProjectionModel`, `SensorModel`,
  `DistortionBrownConrady5`, `DistortionDivision`, `DistortionRational`,
  `DistortionThinPrism`, `RobustLossHuber`, `RobustLossCauchy`,
  `RobustLossArctan` and `UncertaintyUnavailableReason` remain, as aliases of
  `Iso3`, `CameraParams`, `FxFyCxCy`, `DistortionParams`, `ScheimpflugParams`,
  `LaserPlane`, `DistortionKind`, `ProjectionKind`, `SensorKind`,
  `DistortionParams_<Variant>`, `RobustLoss_<Variant>` and
  `UncertaintyUnavailable`.

- Breaking: the rig family is configured through the camera spec, and its
  outputs and exports carry `CameraParams` per camera; the pinhole and
  Scheimpflug rig builders of the optimizer are one. See Migration.

- Each camera model is implemented once, in `vision-calibration-core`: the
  optimizer evaluates the same distortion, Scheimpflug and pinhole code
  under autodiff instead of its own copy. Laser residuals now undistort with
  the core's Newton solver (previously a fixed 5- or 10-step fixed-point
  iteration), so laser results can move in their last digits; the
  regression suite moves by at most 2·10⁻⁶ σ.

- **Examples and tutorials run on data the repository ships.** `planar_real`
  (Rust and Python) calibrates the `data/kuka_1` images and prints each
  parameter's standard deviation; `stereo_session` (Rust and Python) and
  `manual_init` (renamed from `manual_init_proof`) read the detected corners
  in `regression/observations/stereo_rig.json`, so they need no images. The
  five-minute, manual-initialization and per-feature-residuals tutorials and
  the book's real-data chapters follow them. The `stereo_charuco_session`
  (Rust and Python), `dense_stereo_real` and `viewer_fixtures` examples are
  removed: they needed image sets the repository does not ship.
- App: the quick-start presets are the `kuka_1` and DS8 datasets. Presets for
  other local data go in an untracked
  `app/src/workspaces/RunWorkspace/presets.local.ts`.
- The regression suite credits the publisher of each real-image entry:
  `kuka_1` comes from RWHE-Calib (Ali et al., *Sensors* 2019) and the
  stereo pair from the Kaggle dataset *Chessboard Pictures for Stereocamera
  Calibration*, neither of which states a licence; DS8 is U.S. public
  domain.
- **App: the Run workspace's config forms are `SchemaValueForm`.** The
  hand-written `configForm.tsx` is replaced by the shared `@vitavision/forms`
  (0.7; `@vitavision/ui` moves to 0.11). Same schemas and the same whole-value
  in/out, now with labelled groups, switches, bounds and units, switches for
  nullable blocks and an inline error for invalid JSON. `cargo xtask
  emit-schemas` also writes `app/src/schemas/<name>.default.json`
  (`Config::default()` for every config type, drift-checked), and a test
  round-trips each schema through the form with its default.

- **App: the vitavision design system.** The Tauri app drops its local
  component kit and HSL tokens for the published `@vitavision/ui` (0.8):
  IBM Plex type, the shared light and dark palettes, and a system / light /
  dark theme toggle (stored under `calib-theme`, as before).
- **Self-contained documentation.** The READMEs, the book, the tutorials and
  the API docs state their rationale inline rather than pointing at internal
  design records. The app README is a user guide; the tutorials use only
  the datasets shipped with the repository; the book's link and install
  version are corrected, and its solver overview describes the sparse
  Cholesky solve the loop uses.
- **Faster Levenberg–Marquardt iterations.** The loop builds the sparsity
  pattern of `JᵀJ` and the symbolic Cholesky analysis once per solve. Each
  iteration then fills only the lower triangle, accumulating runs of
  Jacobian rows that touch the same parameters in small dense blocks, and a
  damping retry reuses the matrix instead of copying it. Results are
  bit-for-bit unchanged. On the synthetic benchmark, solves are about 5 %
  (tiny-solver) and 7 % (factrs) faster with all cores, and 9 % faster with
  factrs on one core; the gain grows as the Jacobian gets cheaper to
  evaluate.
- **Detector dependencies.** `calib-targets` 0.15 (with `chess-corners`
  1.3) and `ringgrid` 0.13. Chessboard and ChArUco calibrations of the
  bundled datasets and ring-grid detections are unchanged; on a real
  puzzleboard rig the new decoder recovers up to 11 % more corners per
  camera. The app moves to Tauri 2.12.1.

- The book is the one user guide. The walkthroughs of `docs/tutorials/` are
  merged into it (Seeding with Prior Knowledge, Per-Feature Residuals,
  Choosing a Distortion Model, Laser Datasets and the multiple-view geometry
  and desktop-app parts) and the directory is gone. New: Getting Started with
  Rust and Python side by side, Calibration Targets and Detection, The
  Dataset Manifest with a field reference generated from its JSON schema
  (`cargo xtask emit-schemas` writes it, `--check` verifies it), the
  Uncertainty part split into four chapters, and Multiple-View Geometry
  chapters for pose recovery, bundle adjustment, rectification and dense
  stereo. The Python README gains an uncertainty section.

### Fixed

- The seeded Scheimpflug route (`step_init_with_seed` with a sensor, then
  `step_optimize`) lets the tilt move ±0.35 rad from the seed; it was ±0.10.
  A seed off by more, such as a mount specification that says untilted for a
  sensor tilted 16°, held the tilt on its bound. The result then had a worse
  fit and no covariance. On real cameras seeded that way, each camera's
  tilt ended on that bound; now each ends 0.21–0.26 rad from its seed,
  inside the bound, with a covariance, and views left out of the fit are
  reprojected better on five cameras of six.
  Seeds within 0.1 rad of the truth give the same results as before.

- A tilted (Scheimpflug) camera calibrated from a cold start, alone or as
  a rig camera, now reaches its principal point under a robust loss. The
  last stage of the staged solve, which frees the principal point, solved
  without the robust loss and was kept only if the mean reprojection error
  did not grow. That error counts outliers at full weight, so with
  outliers the stage was rejected and the principal point kept its seed.
  It now solves under the caller's loss and is kept unless it raises that
  objective. On synthetic rigs with 5 % outliers under Huber or Cauchy,
  principal-point errors drop from 27–45 px to 1.4–13 px, and tilt errors
  from 1.7–2.9° to 0.08–0.8°, about what clean data at the same noise
  gives. Results change for every robust-loss cold-start and rig
  Scheimpflug calibration; without a robust loss they are unchanged.

- **Robot-pose corrections get their rotation Jacobian from the first
  iteration.** The SE(3) exponential behind the corrections returned the
  identity for an exactly zero rotation, where every correction starts, so
  the solver saw no rotation derivative until a translation step happened to
  move it. With the priors weighed against sub-pixel corner noise this
  stopped the solve after one iteration; it now uses the exponential's
  series near zero, exact to working precision. `apply_robot_delta` applies
  the same exponential as the solver (it had skipped the `V(ω)` factor of
  the translation, a second-order difference).
- **Rig hand-eye step result reports the mean reprojection error.**
  `rig_handeye::step_handeye_optimize` returned the square root of the final
  solver cost as `HandeyeOptimizeResult::mean_reproj_error` (and in the
  session state), not the mean per-point error; on a scene with 0.43 px
  mean error it reported 4.7 px. It now reports the same value as the
  export.
- **The laserline device reaches the noise floor when some target corners
  are outliers.** The device solve used to hold view 0's pose at its linear
  initial value, which outlying corners corrupt, so a robust calibration
  loss could not discount them. It now holds no pose fixed (see the
  Migration note on `fix_poses`). With 5 % of the corners displaced
  10–30 px and Huber or Cauchy at 1 px, at 0.1 px noise: inlier RMS
  0.17–0.32 px → 0.135–0.147 px (0.138–0.141 px without outliers), focal
  error 0.4–1.8 % → 0.009–0.08 %, laser plane up to 0.26° / 7.6 mm off →
  within 0.012° / 0.26 mm.
- **Laser residuals in pixels are the ones the solve minimizes.**
  `LaserFeatureResidual::residual_px`, the per-camera laser histograms and
  the Diagnose laser overlay measured the distance to a straight segment
  through two points of the laser line taken 12 cm either side of the image
  centre, far outside the image. Under strong distortion that misplaced the
  line by tens of pixels. `residual_px` is now the line-distance residual
  of the laser solve, and the overlay line is a polyline over the extent of
  the view's laser pixels, which distortion bends.
- **`detector.min_features_per_view` applies to every topology.** Rig,
  hand-eye and laser manifests drop views below the floor, as planar ones
  do.
- **Hand-eye and rig manifests keep a plain chessboard view only when it
  spans the whole board.** A plain chessboard has no markers, so each
  detection is labelled from its own top-left corner. Where views share one
  target frame (`single_cam_handeye` and every rig topology), a partial
  board, or one seen turned by 90°, moved that frame: on a three-camera
  hand-eye rig, keeping partial boards raised the worst camera's mean
  reprojection error from 0.29 px to 4.7 px. The dataset runner now keeps such a view only when its
  corners reach all four edges of the board, `cols` across and `rows` down;
  interior corners the detector missed do not matter. `planar_intrinsics`
  and `laserline_device` estimate one target pose per view and still keep
  partial boards; ChArUco, PuzzleBoard and ring-grid targets are labelled
  by their codes and are unaffected. Every `*RunResult` lists the
  detections it left out in `rejected` (`RejectedDetection`: camera, image,
  `RejectReason::{TooFewFeatures, PartialChessboard, TurnedChessboard}`),
  and a camera left with no whole-board view fails with
  `RunError::NoWholeChessboardViews` before any solve. In Python the run
  results carry `rejected` as `dataset_runner.RejectedDetection` with the
  reason in snake case, and the error is `RunError` with `kind`
  `"no_whole_chessboard_views"`. A board seen upside
  down, or a square board seen turned by 90°, cannot be told from an
  upright one: capture the board the same way up in every view, or use a
  ChArUco or PuzzleBoard target for partial views.
- **Rotation updates exact at every step size.** The solver retracts SE(3)
  and quaternion parameters through tiny-solver's rotation exponential,
  which is now accurate to machine precision for small steps and keeps its
  sign for rotations beyond 2π. Requires tiny-solver 0.18.7 or newer.
- **No more uphill `SparseQR` steps.** The `SparseQR` linear solver solved
  for the negated step, so the loop rejected every step and returned the
  initial parameters. It also factored `JᵀJ` rather than `J`, so it was no
  more robust than Cholesky; it is removed (see Migration).
- **Robust loss documentation matches the solver.** The book and the
  `RobustLoss` docs state every loss as `ρ(s)` of the squared residual norm,
  the convention the solver minimizes; the Arctan loss's scale is in
  squared-residual units (down-weighting sets in near `‖r‖ ≈ √scale`).
- **Robust solves reach the robust minimum.** The Levenberg–Marquardt loop
  judges each step by the robust objective Σρ(‖r‖²) instead of the squared
  norm of the loss-corrected residuals, which stopped Huber, Cauchy and
  Arctan solves short of their minimum; it also stops once a step can no
  longer change the parameters. Built on `tiny-solver` 0.18.3, which also
  corrects the Cauchy loss value.
- **Solves are bit-for-bit reproducible.** The solver lays out its
  variables in a fixed order, so the same input gives identical results run
  to run (it used to vary in the last digits). The non-linear solve is also
  13–30 % faster on the synthetic benchmark.
- **Detection caches follow detector upgrades.** A cache entry is keyed on
  the `vision-calibration-detect` release as well as the image, detector
  and config, so upgrading recomputes detections instead of serving those
  of the previous detector. Existing caches are recomputed once.
- **A cached detection is the detection that was stored.** `FsDetectionCache`
  read some coordinates back one unit in the last place off, so a run on a
  warm cache could differ in the last digits from the run that filled it.
  Cache files are now read with correctly rounded float parsing (serde_json's
  `float_roundtrip`, which applies to every JSON read in a build that links
  `vision-calibration-detect`), and a cache hit is bit-identical to the fresh
  detection.

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

## [0.8.3] - 2026-09-28

Patch release. Hand-eye calibration no longer rejects a camera mounted
square to the flange. No public signature changes.

### Fixed

- **Hand-eye initialisation accepts a camera mounted square to the flange**
  ([#124](https://github.com/VitalyVorobyev/calibration-rs/issues/124)).
  The Tsai–Lenz initialisers dropped every motion pair whose robot and
  camera rotation axes were parallel. That is true of *every* pair when the
  hand-eye rotation is the identity, so such setups failed with
  `NoValidMotionPairs` even with well-spread robot poses. Parallel A/B axes
  are not ill-conditioned; the solve needs the robot motions to rotate
  about at least two non-parallel axes, which is now what is checked.
  Acceptance results are unchanged.

## [0.8.2] - 2026-09-27

Patch release. Undistortion now converges at the corners of wide-angle
images, and `vision-calibration-core` builds for the browser. No public
signature changes; `iters` keeps its meaning as the iteration cap.

### Fixed

- **Iterative undistortion converges at the image corners**
  ([#120](https://github.com/VitalyVorobyev/calibration-rs/issues/120)).
  `BrownConrady5`, `RationalPolynomial` and `ThinPrism` `undistort` now solve
  the forward map by Newton's method with analytic 2×2 Jacobians and a
  residual-decreasing step, instead of a fixed number of fixed-point steps.
  A wide-angle barrel lens (k1 −0.35) at the corners of a 2048×1536,
  f = 1800 px image was off by 0.32 px after the default 8 steps (0.65 px
  with a 4.2° Scheimpflug tilt); Newton reaches ~5e-13 px in four.
  `iters` keeps its meaning as the iteration cap; `0` now means 20 for all
  three models (was 8 or 10). Explicit small caps (5, 8) remain enough.
- **`vision-calibration-core` builds for `wasm32-unknown-unknown`**

## [0.8.1] - 2026-09-07

Maintenance release. No public API change in any published crate — the
dependency work is confined to the `publish = false` Python bindings crate,
a feature-gated binary, and CI. Cut so the advisory fixes reach the
published crates and the PyPI wheel.

### Security

- **`pyo3` 0.28 → 0.29** in `vision-calibration-py`, closing
  [RUSTSEC-2026-0176](https://rustsec.org/advisories/RUSTSEC-2026-0176)
  (out-of-bounds read in `nth`/`nth_back` for the `PyList` and `PyTuple`
  iterators) and
  [RUSTSEC-2026-0177](https://rustsec.org/advisories/RUSTSEC-2026-0177)
  (missing `Sync` bound on `PyCFunction::new_closure`). `pythonize` and
  `pyo3-build-config` move with it. The Python API and the `abi3-py310`
  wheel tag are unchanged; MSRV is unaffected (pyo3 0.29 requires 1.83).
- **`app/src-tauri` lockfile**, which nothing had ever audited: `plist`
  1.9 → 1.10 pulls `quick-xml` 0.42, closing
  [RUSTSEC-2026-0194](https://rustsec.org/advisories/RUSTSEC-2026-0194)
  (quadratic run time on duplicate-attribute checks) and
  [RUSTSEC-2026-0195](https://rustsec.org/advisories/RUSTSEC-2026-0195)
  (unbounded namespace-declaration allocation); `crossbeam-epoch`
  0.9.18 → 0.9.21 closes
  [RUSTSEC-2026-0204](https://rustsec.org/advisories/RUSTSEC-2026-0204).
- **`chacha20`** 0.10.1 → 0.10.2 (the 0.10.1 release was yanked; reaches us
  transitively through `rand`).

### Changed

- **`toml` 0.8 → 1** in `vision-calibration-dataset`'s optional `cli`
  feature. Internal to the `generate-manifest` binary.

### Fixed

- **The `Security Audit` workflow is load-bearing again.** It had been red
  every Monday since 2026-06-15 without anyone noticing, for three
  independent reasons, all now fixed: `createIssues: false` meant a failing
  scheduled run notified nobody; there was no `pull_request` trigger, so the
  PR that introduced a vulnerable dependency was green and the failure
  surfaced up to six days later; and it scanned only the root lockfile, so
  `app/src-tauri` — outside the workspace, and what ships as
  the desktop bundle — was never audited at all. The job now matrixes over
  both lockfiles, runs on manifest- and lockfile-touching PRs, and files an
  issue when the weekly run on `main` goes red. Advisory ignores moved to a
  new `.cargo/audit.toml` so a local `cargo audit` gives the same answer
  as CI.
- **Stale GitHub Action pins**: `actions/cache` v4 → v6,
  `actions/upload-artifact` v4 → v7 (aligning `app-bundle.yml` with
  `release-pypi.yml`), `actions/setup-python` v6 → v7. A new
  `.github/dependabot.yml` keeps the action pins current from here on.

## [0.8.0] - 2026-08-16

Upstream detector-library migration: `calib-targets` 0.9 → 0.12.1,
`chess-corners` 0.11 → 1.x, `ringgrid` 0.7 → 0.11. Each carried breaking
API changes, and one carried a behavioural change that moves calibration
numbers — hence a minor bump rather than a patch.

### Changed (breaking, pre-1.0)

- **Problem configs reject unknown fields.** All sixteen pipeline `*Config`
  types (the eight problem configs and the grouped sub-structs they share)
  now carry `#[serde(deny_unknown_fields)]`; the generated JSON Schemas say
  `additionalProperties: false` to match. Previously a mistyped key was
  silently dropped and the field fell back to its Rust default — a wrong
  calibration with no error, which the Python test suite had been guarding
  against by hand. **Migration:** fix the key. Any config that was already
  correct is unaffected.
- **`CameraParams::build` returns `Result`.** It could panic on a
  non-invertible sensor homography, and `CameraParams` deserializes straight
  from an export file — so a hand-edited or corrupted export aborted the
  process instead of returning an error.
  `PlanarIntrinsicsParams::build_camera` follows.
- **`vision-calibration-optim` no longer exposes the optimization IR.**
  `ProblemIR`, `ResidualBlock`, `FactorKind`, `ParamSlotSpec`,
  `ManifoldKind`, `CameraModelDesc`, `LaserChain`, `ReprojChain`,
  `SensorKind`, `ProjectionKind`, `FixedMask`, `BackendKind`,
  `BackendSolution`, `solve_with_backend`, and the `pack_*`/`unpack_*`
  helpers are now private — 22 items with no consumer anywhere, which
  publishing would have frozen into the semver contract. They were also a
  hazard: the factor kernels assert invariants only this crate's builders
  establish, so a hand-built IR panicked inside the solver loop. Callers use
  the `optimize_*` entry points.
- **`vision-geometry` items are reached through their module.** The crate
  root globbed five submodules, giving every solver two public paths and
  silently widening the root whenever a submodule gained a `pub fn`. Use
  `vision_geometry::epipolar::fundamental_8point` rather than
  `vision_geometry::fundamental_8point`; `GeometryError` and `Result` stay at
  the root. Every in-tree caller already used the module path.
- **`ReprojLevel::Laser` is removed.** It was never constructed by any
  builder — a public enum variant, and therefore a semver commitment, to a
  feature that does not exist.
- **`threshold_mode` is gone from the detector-override vocabulary.**
  `chess-corners` 1.0 collapsed its `Threshold::{Absolute, Relative}` enum
  into a single absolute `f32`; the `"relative"` mode (a fraction of the
  image maximum response) has no successor. `ChessThresholdMode` is
  deleted from `vision-calibration-detect` and `vision-calibration-dataset`,
  along with the bench registry's parallel `BenchChessThresholdMode` /
  `ChessCornersDetectorOverride`. `ChessCornersDetectorSpec` keeps only
  `threshold_value`, now documented as an absolute floor on the raw ChESS
  response. **Migration:** delete the `threshold_mode` key from manifests
  and bench registries; `deny_unknown_fields` rejects it rather than
  ignoring it. `"absolute"` values carry over unchanged — the workspace
  default is still `15.0`. `"relative"` values have no equivalent and must
  be re-tuned as absolute thresholds.
- **The bench registry's ChESS override is now
  `vision_calibration_dataset::ChessCornersDetectorSpec`** (re-exported
  from `vision_calibration_bench::registry`) instead of a third
  redeclaration of the same two fields.
- **Detector behaviour: `min_corner_strength` now defaults to `33.0`**
  (was `0.0`) via `calib-targets` 0.12, dropping weak, defocused corners
  before the grid builder. `GraphBuildAlgorithm` is gone — `calib-targets`
  collapsed its two grid builders into one, so the former `Topological`
  opt-in is the only path. Chessboard and ChArUco detections change
  accordingly; committed Fit baselines were re-run.
- **`vision-calibration-detect` no longer depends on `chess-corners`**, and
  `calib-targets` is taken with `default-features = false, features =
  ["image"]` — its default set pulled `clap` and a CLI binary into every
  consumer's dependency tree.

### Added

- `DetectError::Backend { detector, message }` in
  `vision-calibration-detect`, for detector-backend failures that are not
  "no target in this frame". Previously the ringgrid wrapper flattened
  every backend error into an empty feature list, so a misconfigured board
  surfaced much later as an unexplained initialisation failure.
- **Facade exports for three error payloads and the detection-cache item
  type.** `vision_calibration::Error` has `Core`/`Linear`/`Optim` variants
  whose payload types the facade never exported, so a facade-only consumer
  could match a variant but not inspect it; they are now `core::Error`,
  `linear::Error` and `optim::Error`. Likewise
  `detect::{Feature, DetectError, reject_ambiguous_detection}` — without
  `Feature`, the exported `DetectionCache` trait could not be implemented
  through the facade at all.
- `reject_ambiguous_detection` in `vision-calibration-detect`, applied by
  all four detectors. A planar target's image-to-board map is a homography
  and so injective; a detection that labels one pixel as several board
  points has mislabelled its grid, and the whole view is rejected (reported
  as a skipped frame) rather than contributing a handful of correspondences
  with no consistent pose.

### Fixed

- **ChArUco and chessboard corner labels are projectively consistent again**,
  via the `calib-targets` 0.12.1 floor. 0.12.0's grid builder could accept a
  mirrored symmetry element when merging lattice components on a thin
  overlap, gluing a sub-block in with a reversed axis — plausible edge
  lengths and pixel spacings, but a labelling no homography fits. It could
  also re-label one physical corner at a run of lattice coordinates. Both
  survive a corner-count check while making the view's pose meaningless. On a
  six-camera ChArUco rig with 720×540 tiles the worst affected camera sat at
  10.9 px mean reprojection under 0.12.0. **Anyone who ran 0.12.0 through
  `vision-calibration` 0.8.0-rc builds should re-run detection** — the
  detection cache keys on image content and detector parameters, not on the
  detector's version, so stale entries will be reused. Delete the cache
  directory to force re-detection.
- **`app/src/schemas/*.json` were being reformatted by Prettier**, drifting
  from `cargo xtask emit-schemas` output. The `.prettierignore` entry meant
  to protect them named `schemas-generated` (the *wire*-schema directory)
  and never matched `src/schemas`. Both are ignored now.
- **`cargo xtask emit-schemas --check` now runs in CI.** Only the
  `app/src-tauri` wire-schema drift check was wired up, so a change to any
  of the nine user-facing config types could silently leave the app's
  schema-driven forms stale.

### Internal

- `nalgebra`, `faer` and `faer-ext` stay at 0.34 / 0.23 / 0.7. They are
  pinned by `tiny-solver` 0.18, which `vision-calibration-optim` exchanges
  both nalgebra and faer types with; 0.35 / 0.24 / 0.8 do not compile until
  tiny-solver moves. Documented at the pin site.
- `criterion` 0.5 → 0.8; benches switched to `std::hint::black_box`
  (`criterion::black_box` is deprecated in 0.8).
- Removed the dead `num-dual` workspace-dependency entry — no crate
  declared it.
- `actions/checkout` v6 → v7 across all workflows.
- Deleted unused IR helpers, four byte-identical copies of
  `format_init_source`, and nineteen `*State` accessors that existed only to
  be called by their own unit tests.
- Float sorts in `vision-mvg`'s degeneracy/homography paths and
  `vision-calibration-linear`'s laser-plane fit use `total_cmp` instead of
  `partial_cmp(..).unwrap()`, which panicked on a NaN produced upstream by a
  divergent solve.
- `session::current_timestamp` returns `0` rather than panicking on a
  pre-1970 clock. It stamps every `session.export()`.
- The manifest's ChESS override is lowered to the detector's option type by
  an exhaustive destructuring rather than an untyped JSON round-trip, so a
  new manifest knob that is not wired through fails to compile.

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

- [`0.7.0`](docs/changelog/0.7.0.md) - 2026-07-10
- [`0.6.0`](docs/changelog/0.6.0.md) - 2026-06-17
- [`0.5.1`](docs/changelog/0.5.1.md) - 2026-05-23
- [`0.5.0`](docs/changelog/0.5.0.md) - 2026-05-21
- [`0.4.0`](docs/changelog/0.4.0.md) - 2026-04-29
- [`0.3.0`](docs/changelog/0.3.0.md) - 2026-04-12
- [`0.2.0`](docs/changelog/0.2.0.md) - 2026-03-07
- [`0.1.2`](docs/changelog/0.1.2.md)
