# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration

- **The facade reaches every type it uses, each by one path** (breaking).
  - The `prelude` module is gone. Import from the modules:

    ```rust
    // before
    use vision_calibration::prelude::*;
    // after
    use vision_calibration::core::{BrownConrady5, FxFyCxCy, PlanarDataset, View, make_pinhole_camera};
    use vision_calibration::planar_intrinsics::PlanarIntrinsicsProblem;
    use vision_calibration::session::CalibrationSession;
    ```

  - `mvg::MvgError` is `mvg::Error` (and `vision_mvg::MvgError` is
    `vision_mvg::Error`), like the `Error` of every other module.
  - `mvg::solvers::dlt_homography_ransac` is removed: the robust homography
    is `mvg::robust::estimate_homography`, which scores the symmetric
    transfer error and returns a `HomographyEstimate`.

    ```rust
    // before
    let (h, inliers) = dlt_homography_ransac(&src, &dst, &opts)?;
    // after
    let corrs: Vec<Correspondence2D> = src.iter().zip(&dst)
        .map(|(s, d)| Correspondence2D::new(*s, *d)).collect();
    let HomographyEstimate { homography: h, inliers, .. } = estimate_homography(&corrs, &opts)?;
    ```

  - Tuples became named types: `fundamental_8point_ransac` returns a
    `FundamentalEstimate { fundamental, inliers }`, `init::pnp::dlt_ransac` a
    `PoseEstimate { pose, inliers }`, `decompose_essential` a
    `Vec<PoseCandidate>` (`rotation`, `translation`) and
    `mvg::residuals::epipolar_line_distance` an `EpipolarLineDistance`
    (`in_first`, `in_second`). `rq_decompose` keeps its `(K, R)` pair, the
    order of the product `M = K R`.

    ```rust
    // before
    let (f, inliers) = fundamental_8point_ransac(&p1, &p2, &opts)?;
    let (d1, d2) = epipolar_line_distance(&f, &c);
    // after
    let FundamentalEstimate { fundamental: f, inliers } = fundamental_8point_ransac(&p1, &p2, &opts)?;
    let EpipolarLineDistance { in_second: d1, in_first: d2 } = epipolar_line_distance(&f, &c);
    ```

  - `rig_laserline_device::step_init`, `step_init_with_seed` and
    `step_optimize` return a `PlaneInitResult` (the initial planes and each
    fit's RMSE) and a `PlaneOptimizeResult` (the planes in the rig and camera
    frames, the mean reprojection error and the per-camera statistics) instead
    of `()`; in Python the session's steps return the same values.
  - `common::CalibratedCameras` is sealed: the library's exports implement
    it, and a type of yours cannot.
  - `RansacOptions` is documented API of the robust estimators
    (`mvg::RansacOptions`).
  - New names for types the facade returned but did not export:
    `core::Ray`, `core::FixedParams`, `core::HomographySensor`,
    `dataset::ValidationError` and the manifest field types
    (`RobotPoseSource`, `PoseConvention`, `PoseColumnMap`,
    `TransformConvention`, `TranslationUnits`, `RotationFormat`,
    `RobotPoseFormat`, `ChessCornersDetectorSpec`, `CornerStrategySpec`,
    `CornerCovarianceSpec`, `ImageNoiseSpec`), so a hand-eye manifest can be
    built in code; `laserline_device::{LaserlineParams, LaserlineEstimate,
    LaserlineStats}`; `rig_handeye_laserline::{RigHandeyeLaserlineParams,
    RigHandeyeLaserlinePerCamStats}`; `mvg::solvers::GeometryError`, the error
    of the closed-form solvers and of `init::pnp`; `init::PlaneFitView`, the
    input of `init::LaserlinePlaneSolver` (it was `LaserlineView`).

- **Public types are `#[non_exhaustive]`; foreign error types are no longer
  exposed** (breaking). The library's signatures name only its own types and
  `nalgebra`:
  - `Error::Serde`, `DeviceSpecError::Json`, `RunError::BadGlob` and
    `RunError::Decode` carry the rendered message (and the path or pattern)
    as a `String`, no longer a `serde_json`, `glob` or `image` error.
  - `common::calibrated_cameras`, `common::check_export_version` and
    `common::read_export` take the export's JSON text (`&str`) instead of a
    `serde_json::Value`: pass `&std::fs::read_to_string(path)?` or
    `&value.to_string()`.
  - Enums that will grow a variant need a `_` arm in a `match`:
    `UncertaintyUnavailable`, `ResidualKind`, `FrameKind`, `ProblemKind`,
    `RobustLoss`, `SolverBackend`, `LaserlineResidualType`, `RobotPose`
    (information), `Detection`, the manifest enums (`TargetSpec`,
    `ImagePattern`, `PosePairing`, `RobotPoseFormat`, `RotationFormat`,
    `TransformConvention`, `TranslationUnits`, `CornerStrategySpec`,
    `ImageNoiseSpec`), `CornerStrategy`, `ImageNoise` and the error enums
    (`RunError`, `DeviceSeedError`, `DeviceSpecError`, `SniffError`,
    `CacheError`, `ValidationError`). `RunError` gains
    `UnsupportedOption` for a manifest value this version does not support.
  - Result structs can no longer be built with a struct literal outside the
    library, and a pattern over them needs `..`: `CameraParams`, the
    `*Estimate` and `*Params` of the optimization problems, `SolveReport`,
    `LaserlineStats`, `RobotPoseSigma`, `RobotRefinement`,
    `ProjectionUncertainty`, `LevelReport`, `ReprojReport`,
    `RigStageReprojection`, and the multiple-view results (`RelativePose`,
    `SceneDiagnostics`, `HomographyEstimate`, `HomographyDecomposition`,
    `StereoRectification`, `RectifiedMatch`, `DisparityMap`, `DenseCloud`,
    `DensePoint`, `TriangulatedPoint`). Build them with `Type::new(..)`;
    `CameraParams` also has `with_projection`, `with_distortion`,
    `with_sensor` and `with_intrinsics` to change one part.
  - `RansacOptions` is built with `RansacOptions::new(thresh)` and the
    `with_max_iters`, `with_min_inliers`, `with_confidence`, `with_seed` and
    `with_refit_on_inliers` methods, no longer with `..RansacOptions::new(t)`.
    `BlockMatchOptions` is built from `BlockMatchOptions::default()` by
    assigning the fields to change.

- **Exports and manifests carry their format version** (breaking). Every
  `*Export` has a required `schema_version` right after `kind`, `1` for all
  seven; an export without it, or with another number, is rejected with an
  error naming both versions, by the Rust readers, the Python `*Export`
  classes and the app. The dataset manifest's `version` is required (it used
  to default to `1`) and must be `1`. Add `"schema_version": 1` to saved
  exports and `version = 1` to manifests that lack it.

- **Serialized enums are snake_case** (breaking). Every enum value in a
  config, input or export is now spelled in snake_case, like the rest of the
  JSON (`"tiny_solver"`, `"eye_in_hand"` in manifests). Saved configs and
  exports written before need their values renamed:

  | Type | Before | After |
  |---|---|---|
  | `HandeyeMode` (`handeye_init.handeye_mode`) | `"EyeInHand"`, `"EyeToHand"` | `"eye_in_hand"`, `"eye_to_hand"` |
  | `HandeyeSolution`, `SingleCamHandeyeSolution`, `HandeyeManualInit` (the `mode` tag) | `"mode": "EyeInHand"` | `"mode": "eye_in_hand"` |
  | `LaserlineResidualType` (`laser_residual_type`) | `"PointToPlane"`, `"LineDistNormalized"` | `"point_to_plane"`, `"line_dist_normalized"` |
  | `RobustLoss` (every loss field) | `"None"`, `{"Huber": {"scale": 1.0}}` | `{"kind": "none"}`, `{"kind": "huber", "scale": 1.0}` |

  `RobustLoss` is now tagged by `kind` like `TargetSpec`, with the variants
  `none`, `huber`, `cauchy` and `arctan`; the Python helpers
  `robust_none()`, `robust_huber(scale)` and the like build the new shape.

  ```json
  // before
  { "handeye_mode": "EyeToHand", "laser_loss": { "Huber": { "scale": 1.0 } },
    "laser_residual_type": "PointToPlane" }
  // after
  { "handeye_mode": "eye_to_hand", "laser_loss": { "kind": "huber", "scale": 1.0 },
    "laser_residual_type": "point_to_plane" }
  ```

- **Hand-eye seeds and init results carry one tagged value per mode**
  (breaking). `HandeyeManualInit` of `single_cam_handeye` and `rig_handeye`
  is an enum with the variants of the export's `handeye`, tagged by `mode` in
  JSON; `Default` is gone (`HandeyeManualInit::unseeded(mode)` is the seed
  with nothing set). A seed whose variant is not the config's
  `handeye_init.handeye_mode` is an `Error::InvalidInput` before any work; it
  used to have its other-mode fields ignored. `HandeyeInitResult` holds
  `handeye`, a `SingleCamHandeyeSolution` or `HandeyeSolution` with both
  poses, in place of the optional poses (single camera) or `initial_handeye`
  and `initial_mode_target_pose` (rig). The serialized state of a hand-eye
  session changes with it (`initial_handeye` replaces the separate initial
  poses), so a saved session from before does not load.

  ```rust
  // before (rig)
  let seed = HandeyeManualInit { handeye: Some(mount), mode_target_pose: None };
  let init = step_handeye_init_with_seed(&mut session, seed, None)?;
  let (he, target) = (init.initial_handeye, init.initial_mode_target_pose);
  // after
  let seed = HandeyeManualInit::EyeInHand { gripper_se3_rig: Some(mount), base_se3_target: None };
  let init = step_handeye_init_with_seed(&mut session, seed, None)?;
  let HandeyeSolution::EyeInHand { gripper_se3_rig, base_se3_target } = init.handeye else { .. };
  ```

  ```python
  # before
  seed = rig_handeye.HandeyeManualInit(handeye=mount)
  init = session.step_handeye_init_with_seed(seed)  # init.initial_handeye
  # after
  seed = rig_handeye.EyeInHandManualInit(gripper_se3_rig=mount)
  init = session.step_handeye_init_with_seed(seed)  # init.handeye.gripper_se3_rig
  ```

  The single-camera seed follows with `gripper_se3_cam` / `cam_se3_base`
  (`single_cam_handeye.EyeInHandManualInit`, `EyeToHandManualInit`).

- **`CalibrationSession` keeps its fields behind methods** (breaking). The
  `config` and `exports` fields are private: read them with `config()` and
  `exports()`, and change the config with `set_config` or `update_config`,
  which validate it and apply the problem's invalidation policy. The
  step-function plumbing is no longer public (`set_output`, `log_success`,
  `log_success_with_notes`, `log_failure`), nor are the mutable accessors
  that bypassed validation (`input_mut`, `require_input_mut`, `output_mut`).
  `reset_output` is removed (it was `clear_output`), as is `reset_state`.

  ```rust
  // before
  let mode = session.config.handeye_init.handeye_mode;
  // after
  let mode = session.config().handeye_init.handeye_mode;
  ```

- **`HandeyeEstimate` records its `mode`** (breaking for code that builds
  one). The saved sessions of `single_cam_handeye` and `rig_handeye` carry it,
  and their `schema_version` is bumped (3 and 2), so a session saved before
  this release is refused with a version error instead of a decoding error.

- **`mvg::dense::GrayImage` and `DisparityMap` check their size** (breaking).
  `new` returns a `Result` (an empty image, or a buffer whose length is not
  `width × height`, is an `InvalidInput` error; it used to panic), and the
  fields are private: read them with `width()`, `height()`, `data()` and
  `into_data()`. `StereoRectification::rectify_image` returns a `Result`
  (a `side` other than 0 or 1 is an error; it used to panic).

  ```rust
  // before
  let img = GrayImage::new(w, h, pixels);
  let rect = rectification.rectify_image(0, &img, w, h);
  // after
  let img = GrayImage::new(w, h, pixels)?;
  let rect = rectification.rectify_image(0, &img, w, h)?;

- **Exports are flat and poses use one spelling** (breaking). The
  `planar_intrinsics` and `laserline_device` exports no longer wrap their
  result in `params`: the camera, the per-view poses and the laser plane sit
  at the top level, like in the other five exports. Every pose names its
  camera frame `cam`, as `cam_se3_rig` always did, and a view index is called
  `view`. Saved exports, residual records and image manifests written before
  need the renames below; the uncertainty labels follow
  (`camera_se3_target.rx` is now `cam_se3_target.rx`).

  ```json
  // before (planar_intrinsics)
  { "kind": "planar_intrinsics",
    "params": { "camera": { ... }, "camera_se3_target": [ ... ] },
    "mean_reproj_error": 0.4, "per_cam_reproj_errors": [0.4], ... }
  // after
  { "kind": "planar_intrinsics",
    "camera": { ... }, "cam_se3_target": [ ... ],
    "mean_reproj_error": 0.4, "per_cam_reproj_errors": [0.4], ... }
  ```

  The `laserline_device` export reads `{ kind, camera, cam_se3_target,
  laser_plane, stats, ... }`; its `params.poses` is `cam_se3_target` and
  `params.plane` is `laser_plane`.

  | Before | After |
  |---|---|
  | `camera_se3_target` (planar and laserline per-view poses, hand-eye views) | `cam_se3_target` |
  | `gripper_se3_camera`, `camera_se3_base` (single-camera hand-eye) | `gripper_se3_cam`, `cam_se3_base` |
  | `RigExtrinsicsParams.cam_to_rig`, `.rig_from_target` | `rig_se3_cam`, `rig_se3_target` |
  | `HandeyeParams.cam_to_rig`, `RigHandeyeLaserlineParams.cam_to_rig`, `RigLaserlineUpstream.cam_to_rig` | `rig_se3_cam` |
  | `LaserlineParams.poses` | `cam_se3_target` |
  | `IntrinsicsManualInit.poses`, `IntrinsicsInitResult.poses`, `DeviceManualInit.poses`, `DeviceInitResult.poses` | `cam_se3_target` |
  | single-camera hand-eye `IntrinsicsManualInit.target_poses`, `IntrinsicsInitResult.target_poses`, `IntrinsicsOptimizeResult.target_poses` | `cam_se3_target` |
  | rig `per_cam_target_poses` (`[view][cam]`, the target's pose in each camera) | `cam_se3_target` |
  | `TargetFeatureResidual.pose`, `LaserFeatureResidual.pose`, `LaserLineTrace.pose`, `FrameRef.pose` | `view` |
  | `export.params.camera`, `.params.camera_se3_target` (planar) | `export.camera`, `export.cam_se3_target` |
  | `export.params.camera`, `.params.poses`, `.params.plane` (laserline) | `export.camera`, `export.cam_se3_target`, `export.laser_plane` |
  | Python `LaserlineDeviceExport.params` (`LaserlineParams`) | `camera`, `cam_se3_target`, `laser_plane` |
  | Python `res.target.pose` | `res.target.view` |

  ```rust
  // before
  let k = export.params.camera.intrinsics;
  let plane = laserline_export.params.plane;
  // after
  let k = export.camera.intrinsics;
  let plane = laserline_export.laser_plane;

  ```

- **Configs, seeds and exports cannot state impossible combinations**
  (breaking). Five shapes changed:

  - `RobotPoseConfig` is an enum, `exact` or `refine` with its prior
    (`RobotPosePrior`: `rot_sigma`, `trans_sigma`, `estimate_sigma`). The
    sigmas used to sit beside a `refine` flag that made them meaningless,
    and the problems checked them differently; now a `refine` prior is
    validated the same way everywhere, and an `exact` config has none.

    ```json
    // before
    { "robot_poses": { "refine": true, "rot_sigma": 0.0087, "trans_sigma": 0.001, "estimate_sigma": true } }
    { "robot_poses": { "refine": false, "rot_sigma": 0.0087, "trans_sigma": 0.001, "estimate_sigma": true } }
    // after
    { "robot_poses": { "kind": "refine", "rot_sigma": 0.0087, "trans_sigma": 0.001, "estimate_sigma": true } }
    { "robot_poses": { "kind": "exact" } }
    ```

    ```rust
    // before
    config.robot_poses.refine = false;
    // after
    config.robot_poses = RobotPoseConfig::Exact;
    config.robot_poses = RobotPoseConfig::Refine(RobotPosePrior::new(0.0087, 1e-3));
    ```

    Python: `RobotPoseConfig.exact()`, `RobotPoseConfig.refine(rot_sigma=...,
    trans_sigma=..., estimate_sigma=...)` and the field `prior`.

  - The robot-pose correction of a hand-eye result is one optional value,
    `robot_refinement { deltas, sigma }`, in the single-camera, rig and
    rig-laser exports (and in `HandeyeEstimate`). It replaces the two
    optional fields `robot_deltas` and `robot_pose_sigma`, which were always
    set together.

    ```json
    // before
    { "robot_deltas": [[0, 0, 0, 0, 0, 0]], "robot_pose_sigma": { "rot_sigma": 0.001, "trans_sigma": 0.0005 } }
    // after
    { "robot_refinement": { "deltas": [[0, 0, 0, 0, 0, 0]], "sigma": { "rot_sigma": 0.001, "trans_sigma": 0.0005 } } }
    ```

    Python: `export.robot_refinement.deltas` and `.sigma`.

  - `RigManualInit` seeds the rig poses as one `poses: Option<RigPoses>`
    (`RigPoses::new(cam_se3_rig, rig_se3_target)`) in place of the two
    options that had to be set together.

    ```rust
    // before
    seed.cam_se3_rig = Some(cam_se3_rig);
    seed.rig_se3_target = Some(rig_se3_target);
    // after
    seed.poses = Some(RigPoses::new(cam_se3_rig, rig_se3_target));
    ```

    Python: `RigManualInit(poses=RigPoses(cam_se3_rig=..., rig_se3_target=...))`.

  - `RigHandeyeConfig.manual_init` now seeds `step_intrinsics_init_all` as it
    already seeded `run_calibration`; a step-by-step run with a configured
    seed starts from it instead of a full auto-init.
    `step_intrinsics_init_all_with_seed` takes an explicit seed that
    overrides the config's.

  - `RigLaserlineDeviceInput` no longer has `initial_planes_cam`; seed the
    laser planes with `step_init_with_seed(PlaneManualInit { laser_planes_cam })`
    (Python: `RigLaserlineDeviceSession.step_init_with_seed`).

    ```rust
    // before
    RigLaserlineDeviceInput { dataset, upstream, initial_planes_cam: Some(planes) }
    // after
    RigLaserlineDeviceInput { dataset, upstream }
    ```

- **One config grammar across the problem types** (breaking). Every config
  is grouped by stage, in the order the steps run, and a stage has the same
  group name in every problem: `camera`, `intrinsics_init`, the refinement
  stages (`optimize`, `intrinsics_optimize`, `rig`, `handeye_optimize`,
  `joint_optimize`) and `solver`. `solver` holds the solver engine only
  (`max_iters`, `verbosity`, `backend`). Each refinement stage carries the
  robust loss of the residuals it solves, named by their family:
  `reprojection_loss` for the target corners, `laser_loss` for the laser
  residuals. Where one `solver.robust_loss` drove several stages, each stage
  now has its own loss: set the old value in every one of them to keep a
  config's results (the defaults are unchanged). A stage loss of its own
  lets a convex loss (`huber`) start a problem and a redescending one
  (`cauchy`, `arctan`) finish it.

  | Problem | Before | After |
  |---|---|---|
  | `planar_intrinsics` | `init` | `intrinsics_init` |
  | `planar_intrinsics` | `fix_poses` | `optimize.fix_poses` |
  | `planar_intrinsics` | `solver.robust_loss` | `optimize.reprojection_loss` |
  | `laserline_device` | `init` | `intrinsics_init` |
  | `laserline_device` | `optimize.calib_loss`, `optimize.calib_weight` | `optimize.reprojection_loss`, `optimize.reprojection_weight` |
  | `laserline_device` | `solver.robust_loss` | removed (see below) |
  | `single_cam_handeye` | `intrinsics` | `intrinsics_init` |
  | `single_cam_handeye` | `robot_poses` | `handeye_optimize.robot_poses` |
  | `single_cam_handeye` | `solver.robust_loss` | `intrinsics_optimize.reprojection_loss`, `handeye_optimize.reprojection_loss` |
  | `rig_extrinsics` | `intrinsics` | `intrinsics_init` |
  | `rig_extrinsics` | `solver.robust_loss` | `intrinsics_optimize.reprojection_loss`, `rig.reprojection_loss` |
  | `rig_handeye` | `intrinsics` | `intrinsics_init` |
  | `rig_handeye` | `solver.robust_loss` | `intrinsics_optimize.reprojection_loss`, `rig.reprojection_loss`, `handeye_optimize.reprojection_loss` |
  | `rig_laserline_device` | `laser_residual_type` | `optimize.laser_residual_type` |
  | `rig_laserline_device` | (fixed Huber at 1) | `optimize.laser_loss` (default Huber at 1) |
  | `rig_laserline_device` | `solver.robust_loss` | removed (it was not read) |
  | `rig_handeye_laserline` | `handeye.…`, `laserline_init.…` | as `rig_handeye` and `rig_laserline_device` above |
  | `rig_handeye_laserline` | `joint_optimize.solver` | `solver` |
  | `rig_handeye_laserline` | `joint_optimize.calib_loss`, `joint_optimize.calib_weight` | `joint_optimize.reprojection_loss`, `joint_optimize.reprojection_weight` |
  | `rig_handeye_laserline` | `laserline_init.solver.robust_loss`, `joint_optimize.solver.robust_loss` | removed (they were not read) |

  A laserline device's `solver.robust_loss` applied only to the initial
  camera-only solve of a Scheimpflug camera whose tilt is estimated and not
  seeded; that solve is now plain least squares, as it was by default. New
  Rust group types: `common::config::IntrinsicsOptimizeConfig`,
  `planar_intrinsics::PlanarOptimizeConfig`,
  `single_cam_handeye::SingleCamHandeyeOptimizeConfig` and
  `rig_laserline_device::PlaneOptimizeConfig`; `RigConfig` and the rig
  `HandeyeOptimizeConfig` gain `reprojection_loss`, and `JointOptimizeConfig`
  loses `solver`. Python has the same classes.

  ```json
  // planar_intrinsics, before → after
  { "init": { "iterations": 2, "fix_k3": true }, "fix_poses": [],
    "solver": { "max_iters": 120, "verbosity": 0, "robust_loss": { "kind": "huber", "scale": 1.0 }, "backend": "tiny_solver" } }
  { "intrinsics_init": { "iterations": 2, "fix_k3": true },
    "optimize": { "reprojection_loss": { "kind": "huber", "scale": 1.0 }, "fix_poses": [] },
    "solver": { "max_iters": 120, "verbosity": 0, "backend": "tiny_solver" } }

  // laserline_device
  { "init": { … }, "optimize": { "calib_loss": { … }, "calib_weight": 1.0, … }, "solver": { …, "robust_loss": { "kind": "none" } } }
  { "intrinsics_init": { … }, "optimize": { "reprojection_loss": { … }, "reprojection_weight": 1.0, … }, "solver": { … } }

  // single_cam_handeye
  { "intrinsics": { … }, "handeye_init": { … }, "robot_poses": { "kind": "refine", … },
    "solver": { …, "robust_loss": { "kind": "huber", "scale": 1.0 } } }
  { "intrinsics_init": { … },
    "intrinsics_optimize": { "reprojection_loss": { "kind": "huber", "scale": 1.0 } },
    "handeye_init": { … },
    "handeye_optimize": { "reprojection_loss": { "kind": "huber", "scale": 1.0 }, "robot_poses": { "kind": "refine", … } },
    "solver": { … } }

  // rig_extrinsics
  { "intrinsics": { … }, "rig": { "reference_camera_idx": 0, … }, "solver": { …, "robust_loss": { "kind": "huber", "scale": 1.0 } } }
  { "intrinsics_init": { … },
    "intrinsics_optimize": { "reprojection_loss": { "kind": "huber", "scale": 1.0 } },
    "rig": { "reference_camera_idx": 0, …, "reprojection_loss": { "kind": "huber", "scale": 1.0 } },
    "solver": { … } }

  // rig_handeye: as rig_extrinsics, plus
  { "handeye_optimize": { "robot_poses": { … }, "refine_intrinsics": true, … } }
  { "handeye_optimize": { "reprojection_loss": { "kind": "huber", "scale": 1.0 }, "robot_poses": { … }, "refine_intrinsics": true, … } }

  // rig_laserline_device
  { "laser_residual_type": "line_dist_normalized", "solver": { …, "robust_loss": { "kind": "none" } } }
  { "optimize": { "laser_residual_type": "line_dist_normalized", "laser_loss": { "kind": "huber", "scale": 1.0 } }, "solver": { … } }

  // rig_handeye_laserline
  { "handeye": { … }, "laserline_init": { "laser_residual_type": "point_to_plane", "solver": { … } },
    "joint_optimize": { "solver": { "max_iters": 30, … }, "calib_loss": { … }, "calib_weight": 1.0, … } }
  { "handeye": { … }, "laserline_init": { "optimize": { "laser_residual_type": "point_to_plane", "laser_loss": { … } }, "solver": { … } },
    "joint_optimize": { "reprojection_loss": { … }, "reprojection_weight": 1.0, … },
    "solver": { "max_iters": 30, … } }
  ```

  ```rust
  // before
  config.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
  // after (rig_handeye)
  let huber = RobustLoss::Huber { scale: 1.0 };
  config.intrinsics_optimize.reprojection_loss = huber;
  config.rig.reprojection_loss = huber;
  config.handeye_optimize.reprojection_loss = huber;
  ```

- **One name per statistic, per laser plane and per problem kind**
  (breaking). Saved exports and code that reads them need the renames below.

  | Where | Before | After |
  |---|---|---|
  | `rig_laserline_device`, `rig_handeye_laserline` exports; `RigLaserlineEstimate` | `per_camera_stats` | `per_cam_stats` |
  | `rig_handeye_laserline` per-camera stats | `mean_reproj_error_px`, `max_reproj_error_px`, `reproj_histogram_px` | `mean_reproj_error`, `max_reproj_error`, `reproj_histogram` |
  | `rig_handeye_laserline` per-camera stats | `mean_laser_err_m`, `max_laser_err_m`, `mean_laser_err_px`, `max_laser_err_px` | `mean_laser_error_m`, `max_laser_error_m`, `mean_laser_error_px`, `max_laser_error_px` |
  | `RigHandeyeLaserlineEstimate` | `planes_rig`, `mean_reproj_error_px` | `laser_planes_rig`, `mean_reproj_error` |
  | `RigHandeyeLaserlineParams` | `planes_cam` | `laser_planes_cam` |
  | `LaserlineParams`, `DeviceManualInit`, `DeviceInitResult` | `plane` | `laser_plane` |
  | `PlaneManualInit`, `JointManualInit` | `planes_cam` | `laser_planes_cam` |
  | every export's `diagnostics.per_feature_residuals` | `target_hist_per_camera`, `laser_hist_per_camera` | `target_hist_per_cam`, `laser_hist_per_cam` |
  | manifest `topology`, export `kind` (Rust) | `dataset::Topology`, `common::ExportKind` | `dataset::ProblemKind` (same JSON values) |
  | manifest and detector `image_noise` | `{ known = σ }`, `ImageNoiseSpec::Known` | `{ stated = σ }`, `ImageNoiseSpec::Stated` |

  The manifest's `topology` and every export's `kind` are one enum,
  `vision_calibration::dataset::ProblemKind`; their JSON spellings are
  unchanged. The saved sessions of all seven problem types carry the new
  configs, and their `schema_version` is bumped, so a session saved before
  this release is refused with a version error.

- **Python names are the Rust names; `calibrate_dataset` is the library's
  runner** (breaking, Python). Where a Python name and its Rust type
  disagreed, the Python name follows Rust, one capability has one path, and
  `calibrate_dataset` returns what the Rust runner returns.

  | Before | After |
  |---|---|
  | `vc.ParameterUncertainty` | `vc.ParameterCovariance` |
  | `vc.Camera` | `vc.CameraParams` |
  | `vc.RigHandeyeSolution` | `vc.HandeyeSolution` |
  | `vc.run_planar_intrinsics(input, config)` | `vc.planar_intrinsics.run_calibration(input, config)` |
  | `vc.run_single_cam_handeye(...)` | `vc.single_cam_handeye.run_calibration(...)` |
  | `vc.run_rig_extrinsics(...)` | `vc.rig_extrinsics.run_calibration(...)` |
  | `vc.run_rig_handeye(...)` | `vc.rig_handeye.run_calibration(...)` |
  | `vc.run_laserline_device(...)` | `vc.laserline_device.run_calibration(...)` |
  | `vc.run_rig_laserline_device(...)` | `vc.rig_laserline_device.run_calibration(...)` |
  | `vc.run_rig_handeye_laserline(...)` | `vc.rig_handeye_laserline.run_calibration(...)` |
  | `dataset_runner.CalibrationResult` | `dataset_runner.CalibrationExport` |

  `calibrate_dataset` now runs the Rust runner: the manifest's topology picks
  the converter and the problem there, not in Python. It returns a
  `dataset_runner.DatasetCalibration` (the export, `image_sizes`,
  `usable_views`, `total_views` and the `rejected` detections) instead of
  the export alone, and its arguments are unchanged:

  ```python
  # before
  result = vc.calibrate_dataset(manifest, cache=cache)
  # after
  run = vc.calibrate_dataset(manifest, cache=cache)
  result = run.export
  report = vc.assess(result, image_sizes=run.image_sizes)
  ```

  The one-shot `run_calibration` of a problem module runs a session, so an
  input the problem rejects while running raises `ValueError`, as it does
  from the session, where `run_*` raised `RuntimeError`.

  A config dataclass of another topology than the manifest's raises
  `ValueError` before any image is read, and a manifest payload that does not
  parse raises `ValueError` rather than `TypeError`.

- **One manual intrinsics seed type per family, in `common`** (breaking).
  Planar intrinsics and single-camera hand-eye share one
  `IntrinsicsManualInit`; the rig problems' per-camera seed is
  `RigIntrinsicsManualInit`. Both are reached through `common` only. The
  JSON of the seeds is unchanged; in the config schemas the definition the
  rig hand-eye `manual_init` names is `RigIntrinsicsManualInit`.

  | Before | After |
  |---|---|
  | `planar_intrinsics::IntrinsicsManualInit` | `common::IntrinsicsManualInit` |
  | `single_cam_handeye::IntrinsicsManualInit` | `common::IntrinsicsManualInit` |
  | `rig_extrinsics::IntrinsicsManualInit` | `common::RigIntrinsicsManualInit` |
  | `rig_handeye::IntrinsicsManualInit` | `common::RigIntrinsicsManualInit` |
  | Python `rig_extrinsics.IntrinsicsManualInit`, `rig_handeye.IntrinsicsManualInit` | `vc.RigIntrinsicsManualInit` |

  In Python `vc.IntrinsicsManualInit` and `vc.RigIntrinsicsManualInit` are
  importable from the package root; the problem modules re-export them.

- **`vision_calibration.types` carries only the generated payload types and
  `ProgressCallback`** (breaking, Python). Its other aliases are removed; use
  the generated type each named: `Transform` → `Iso3`, `IntrinsicsPayload` →
  `FxFyCxCy`, `DistortionPayload` and `DistortionParamsPayload` →
  `DistortionParams`, `ScheimpflugPayload` → `ScheimpflugParams`,
  `LaserPlanePayload` → `LaserPlane`, `DistortionModel` → `DistortionKind`,
  `ProjectionModel` → `ProjectionKind`, `SensorModel` → `SensorKind`,
  `DistortionBrownConrady5` / `DistortionDivision` / `DistortionRational` /
  `DistortionThinPrism` → `DistortionParams_BrownConrady5` / `_Division` /
  `_Rational` / `_ThinPrism`, `RobustLossHuber` / `RobustLossCauchy` /
  `RobustLossArctan` → `RobustLoss_Huber` / `_Cauchy` / `_Arctan`,
  `UncertaintyUnavailableReason` → `UncertaintyUnavailable`. `JsonObject`
  and `JsonArray` are `dict[str, Any]` and `list[Any]`; `Se3Delta` is
  `list[float]`.

### Changed

- **A hand-eye seed of one pose closes the other through the first view, in
  both modes and both hand-eye workflows** (a changed initialization for
  four seed combinations). `single_cam_handeye` and `rig_handeye` now run
  the same hand-eye stages, and `step_handeye_init_with_seed` follows one
  rule: a seeded pose is used as given; with one pose seeded the other is
  derived from it through the first view, without the closed-form solve;
  with neither seeded the closed-form solver gives both. Before, a seed of
  the target pose alone (`base_se3_target` eye-in-hand, `gripper_se3_target`
  eye-to-hand in a rig) and a seed of `cam_se3_base` alone (single camera,
  eye-to-hand) ran the closed-form solve and paired its estimate of the
  other pose with the seed, so the two did not describe the same chain and
  a robot motion too poor for the closed-form solve failed the step despite
  the seed. A seed of the hand-eye alone, of both poses or of neither gives
  the same result as before, as does every `run_calibration`. On synthetic
  captures (nine stations, 0.2 px noise, the seed 0.86° and 5.4 mm off) the
  derived pose starts further off than the closed-form estimate did (for
  example 1.12° and 17 mm instead of 0.28° and 5.9 mm for a single camera's
  hand-eye from a seeded `base_se3_target`), and the bundle adjustment ends
  at the same result: identical to the printed digits in three of the four
  combinations on both solver backends, and within 0.003 of its standard
  deviation (7 µm, 0.0003°) for a rig's eye-to-hand `rig_se3_base`.
- `single_cam_handeye::HandeyeOptimizeResult` and
  `rig_handeye::HandeyeOptimizeResult` are one type, as the two problems'
  hand-eye bundle adjustments are one stage. The rig's `handeye_init` log
  note names the hand-eye pose (`gripper_se3_rig |t|=…`) as the single
  camera's does, instead of `translation_norm=…`.

### Added

- **The regression suite gates a measurement.** The `opencv_stereo` entry
  also measures its held-out boards: its even views calibrate a stereo rig,
  the corners of each odd view are triangulated with the calibration's
  uncertainty, and the 2070 distances along board rows and columns, 1 to 7
  squares, are compared with the board's size. The reference result stores
  the mean and RMS error, the mean stated σ and the RMS of error over σ per
  held-out view and length, and `calib-bench suite check` fails on any
  change (with `--sigma k`, on a mean length that moves by `k` of its σ).
  A suite entry asks for it with `measure_board = { square_size_m = ... }`.

- **`init` offers the fisheye and distortion initializers.**
  `estimate_planar_pose_from_bearings` (a board's pose from pixel bearings),
  `estimate_extrinsics_from_cam_target_poses` (a rig's extrinsics from per-camera
  board poses, with `ExtrinsicPoses`),
  `estimate_equidistant_intrinsics` (with `EquidistantInitOptions` and
  `EquidistantIntrinsicsLinearInit`) initializes a Kannala-Brandt camera from
  planar views, and `estimate_distortion_from_homographies` (with
  `DistortionView` and `MetaHomography`) fits Brown-Conrady distortion for
  known intrinsics, so the book's chapters on them compile against the facade.
- **Every Rust block of the book compiles.** The book's chapters are doctests
  of the facade crate, so `cargo test --doc -p vision-calibration` fails on
  a block that no longer matches the API.

### Fixed

- `optimize_rig_handeye_laserline` returns an `Error::Numerical` naming the
  block when the solver's solution lacks one of the problem's parameter
  blocks; it used to panic.

- `calibrate_dataset` asks for a manifest's missing `pose_pairing`
  (`RunError::AskUser`) before it reads any image; it used to report the
  first unreadable image instead.

- `vcal run --report` writes its files all or none. A report that could not
  be written used to leave `calibration.json` and `quality.json` behind; the
  run now stops with the error and leaves the output directory as it was.
- `vcal run -c` reads a configuration file named `config.TOML` (any case of
  the extension) as TOML; it was read as JSON and failed.

- `rig_handeye::step_handeye_init` and `step_handeye_init_with_seed` that
  failed (the closed-form solve finding too little robot rotation, say) left
  the previous run's initialization in place, and `step_handeye_optimize`
  then started from it. A run now clears the initialization once it starts
  its work, as the single-camera step already did, so a failed run leaves
  none.

- A hand-eye session whose `handeye_mode` was changed after solving exported
  its result in the new mode: the export took the mode from the current
  config, which a config change does not clear. The export now reads the
  mode the hand-eye was solved in.
- `mvg::homography::decompose_homography` decomposes a homography of either
  sign. A homography with a negative determinant (`−H`, as a DLT that fixes
  `h33 = 1` returns when the true `h33` is negative) gave every candidate a
  rotation off by a half-turn about the plane normal. Its rank test is now
  relative to the largest singular value, and non-finite entries are a
  `Degenerate` error.
- `init::pnp::dlt` and `dlt_ransac` return a `Degenerate` error for coplanar
  (or collinear) world points instead of an arbitrary pose: a planar object
  leaves the 12-unknown DLT underdetermined. Use `epnp` for a planar object.
- The closed-form solvers of `mvg::solvers` and `init::pnp`, and
  `mvg::homography::decompose_essential`, return an `InvalidInput` error for a
  NaN or infinite coordinate. `fundamental_7point` and `p3p` used to hang
  on one, in an SVD or Schur iteration that never converges.
- `RansacOptions::validate` checks every field: `confidence` must be strictly
  between 0 and 1 and `max_iters` at least 1 (a confidence of 1 or NaN ended
  the search after the first model). A refit that leaves fewer than
  `min_inliers` inliers no longer replaces the sample's model.
- The measurement functions refuse a pixel outside the region where the
  camera model inverts (beyond where a strong barrel distortion folds over),
  instead of measuring along the undistortion's best-effort ray, which no
  pixel of the camera sees.
- A laser point seen head-on no longer reports a NaN incidence angle.

- The Daniilidis hand-eye initialization lost accuracy when robot motions
  between two views came within noise of a half turn (a 180° wrist flip):
  the two quaternions of such a motion could enter with opposite signs. With
  six such flips among 14 views it was off by up to 2.2° and 14 mm where the
  other methods stayed within 0.3°. Signs are now matched through the
  Tsai-Lenz rotation; motions away from 180° give the same result as before.

- `rig_laserline_device` stops with an error naming the camera when a
  camera's laser plane cannot be fitted in closed form (no laser pixels, or
  pixels that do not determine a plane). It used to start that camera's
  solve from a made-up plane 0.2 m in front of it.
- Robot poses are checked as they load: a quaternion whose norm is not 1
  (within 1e-3), a rotation matrix that is a reflection or not orthonormal
  (within 1e-3), and a `nan` or `inf` value are `pose_parse` errors. They
  used to be normalized, projected onto the nearest rotation, or accepted.
- A camera's `roi_xywh` must lie inside its images: a region that runs past
  them is a `RunError::RoiOutOfBounds` (`roi_out_of_bounds` in Python), where
  it used to be cropped short silently. `camera_image_sizes`, and so
  `vcal check`, now read the images of a camera with a region too.
- `calibrate_dataset` checks the config before reading any image, and the
  rig hand-eye and rig laser runs load the robot poses and the upstream
  calibration before detection, so an invalid config or an unreadable pose
  or upstream file fails the run first.
- An export with no target feature to average no longer reports a mean
  reprojection error of 0 (`rig_laserline_device`) or infinity (a tilted
  planar camera): the calibration fails with an error instead.

## [0.18.0] - 2026-11-06

### Migration

- **`vision-mvg` is module-first** (breaking). The flat re-exports at its
  crate root are gone; only `MvgError` remains there. Reach each item through
  its module: `vision_mvg::bundle_adjust::bundle_adjust`,
  `vision_mvg::pose_recovery::recover_relative_pose`,
  `vision_mvg::types::PixelMatch`, `vision_mvg::robust::estimate_essential`,
  and so on. The closed-form solvers that were re-exported from
  `vision-geometry` (`dlt_homography`, `essential_5point`, ...) are imported
  from `vision_geometry` itself. Through the facade, `mvg::bundle_adjust` is
  now only the module.

- **`mvg::error` and `mvg::degeneracy` are gone** (breaking). `mvg::MvgError`
  stays; write `Result<T, MvgError>` where you used `mvg::error::Result<T>`.
  `SceneDiagnostics` moved to `mvg::pose_recovery::SceneDiagnostics`, next to
  `RelativePose`, which holds it in `scene`.

- **`vision_mvg::types::{EssentialMatrix, FundamentalMatrix, HomographyMatrix}`
  are removed** (breaking). They were aliases of `Mat3`; use `core::Mat3`.

- **The closed-form solvers moved to `mvg::solvers`** (breaking):
  `mvg::dlt_homography` becomes `mvg::solvers::dlt_homography`, and likewise
  `dlt_camera_matrix`, `decompose_camera_matrix`, `CameraMatrixDecomposition`,
  `Mat34`, `essential_5point`, `fundamental_7point`, `fundamental_8point`,
  `decompose_essential` and `triangulate_point_linear`. The module also
  holds the RANSAC wrappers `dlt_homography_ransac` and
  `fundamental_8point_ransac` and the RQ decomposition `rq_decompose`, which
  the facade did not reach before.

- **`analysis` lists what it exports** (breaking for the items removed). It no
  longer re-exports the multi-level reprojection reports (`ReprojLevel`,
  `ReprojReport`, `LevelReport`, `LevelStats`, `RigStageReprojection` and the
  `*_report` builders) nor the helpers that build an export's per-feature
  residuals (`compute_*_residuals`, `compute_*_feature_residuals`,
  `build_feature_histogram`, `handeye_observer_se3_target`,
  `LaserResidualRecords`). They stay in `vision-calibration-pipeline`,
  `vision-calibration-core` and `vision-calibration-optim`. The quality,
  report and next-view items and the residual types that the exports hold
  are unchanged.

- **`optim::problem::Noise::Known` is now `Noise::Stated`** (breaking), the
  word `PixelNoise::Stated` already uses for a noise the caller gives.

## [0.17.0] - 2026-11-06

### Migration

- **The rig workflows refine each Scheimpflug camera's tilt in their joint
  solves** (a changed result). `RigConfig::refine_sensor` (`rig.refine_sensor`)
  and `HandeyeOptimizeConfig::refine_sensor` (`handeye_optimize.refine_sensor`)
  now default to `true`, also when a config leaves them out, and the
  `rig_handeye_laserline` joint stage's default fix list no longer holds the
  tilt. A tilt held at the camera's own estimate left its error in the focal
  length and principal point, and the covariance, conditional on it, stated
  their σ far too small: on a four-camera tilted rig seen in six nearly
  fronto-parallel views (60 noise draws), the focal lengths were 17–26 px
  low on average and scattered 27–32 px against a stated σ of 3.4–3.6 px;
  refined in the rig solve they are unbiased (≤ 0.3 px), with a scatter of
  7.2–7.8 px against a stated σ of 6.9–7.6 px, and the tilts scatter
  0.05° against a stated 0.05°. On a well-conditioned two-camera tilted rig
  the estimates barely move, but the principal points' σ was 20 times too
  small and is now within 5 % of the scatter. A tilted camera with no
  distortion cannot tell its tilt from a principal-point shift, so its rig
  covariance is now `rank_deficient` instead of conditional on a tilt the
  data do not determine. Rigs of untilted cameras are unchanged. Set the flags to `false` (or name the tilts in the joint
  stage's `fix`) for the previous behaviour; the covariance then lists the
  tilts under `fixed`.

- **The multiple-view-geometry solves take a run control** (breaking).
  `mvg::triangulation::triangulate` and `StereoRectification::triangulate`
  gain a last argument `control: &RunControl`; pass
  `&RunControl::default()` for the old behaviour. `PoseOptions` and
  `BundleAdjustOptions` gain a `control` field (default: no observers, no
  token), and `BundleAdjustOptions` is no longer `Copy` or `PartialEq`.
  `MvgError` gains `Cancelled`.
- **The robust two-view estimates take a run control and carry their
  covariance** (breaking). `mvg::robust::estimate_essential` and
  `recover_relative_pose_robust` gain a last argument
  `control: &RunControl`. `EssentialEstimate` and `RobustRelativePose`
  gain an `uncertainty` field and are `#[non_exhaustive]` (build them
  through the functions, not as literals). Their `inliers` are the matches
  the estimate is fitted to.
- **The refined essential matrix and relative pose move** (a changed
  result). The consensus set is now refined to the minimum of the weighted
  Sampson error itself, the variance's dependence on `E` included. The
  previous polish held each match's variance fixed within a step and
  settled at a fixed point that is not that minimum: over 300 noise draws
  of a fisheye matched with a pinhole it was biased by up to 0.9 σ of the
  pose with 120 matches and up to 2.4 σ with 1000, and its scatter was up
  to 1.5 (2.6) times the pose's σ; the refined estimate's bias is at most
  0.1 σ and its scatter 0.97–1.07 σ. The estimates move by up to
  4° of translation direction on the test scenes, the inlier RMS falls,
  and an inlier set can gain a match.
- **Measurements name the frame of their captures** (breaking).
  `triangulate(rig, rig_se3_world, tracks, control)` is now
  `triangulate(rig, captures, tracks, control)` with `captures` a
  `mvg::types::Captures`: `Captures::World(&rig_se3_world)` for the old
  behaviour, `Captures::Rig` for one capture in the rig frame (what
  `&[Iso3::identity()]` was). `triangulate_laser(rig, observations)` is
  `triangulate_laser(rig, Captures::Rig, observations)`, and
  `LaserObservation::new(camera, pixel, noise)` is
  `LaserObservation::new(capture, camera, pixel, noise)` (`capture` 0 for
  one capture; the JSON form gains a required `capture`). In Python,
  `mvg.triangulate`'s second argument is `captures` (the poses as before, or
  a `RobotCaptures`), `mvg.triangulate_laser` takes an optional `captures`,
  and `LaserObservation` gains `capture` (default 0).
- **`rig_laserline_device::pixel_to_gripper_point` is removed** (breaking),
  in Python too: it gave a laser point in the gripper frame without its σ.
  Measure with `triangulate_laser` on a hand-eye laser export's
  `calibrated_rig()` and `Captures::Handeye(RobotFrame::Gripper)`
  (eye-in-hand) or `Captures::Robot { frame: RobotFrame::Gripper, robot }`
  (eye-to-hand, with the robot's pose); for a staged laser export, attach
  the hand-eye with `CalibratedRig::with_handeye`, which then counts as
  exact. In Python, `mvg.triangulate_laser(result, observations,
  mvg.RobotCaptures("gripper"))`.
- **The hand-eye exports carry their hand-eye as one tagged value**
  (breaking: the exports' JSON schemas). `SingleCamHandeyeExport`,
  `RigHandeyeExport` and `RigHandeyeLaserlineExport` no longer have
  `handeye_mode` and four optional poses; they have `handeye`, which holds
  exactly the two poses of its mode, tagged by `mode`. An export with the
  wrong pose for its mode cannot be written or read, and exports from before
  this change no longer load. The covariance labels are unchanged.

  ```json
  // before (rig_handeye, eye-in-hand)
  "handeye_mode": "EyeInHand",
  "gripper_se3_rig": {…}, "rig_se3_base": null,
  "base_se3_target": {…}, "gripper_se3_target": null
  // after
  "handeye": { "mode": "EyeInHand", "gripper_se3_rig": {…}, "base_se3_target": {…} }
  ```

  In Rust, `export.handeye` is a `common::HandeyeSolution`
  (`EyeInHand { gripper_se3_rig, base_se3_target }` | `EyeToHand
  { rig_se3_base, gripper_se3_target }`) for the rig exports and a
  `single_cam_handeye::SingleCamHandeyeSolution` (`gripper_se3_camera` /
  `camera_se3_base`) for the single-camera export; match on it instead of on
  `export.handeye_mode`. In Python, `result.handeye` is a `RigEyeInHand` /
  `RigEyeToHand` or a `CameraEyeInHand` / `CameraEyeToHand`, with `.mode`;
  `result.handeye_mode` and the four optional pose attributes are gone.

### Fixed

- **The desktop app's Quick Start presets work in an installed app.** They
  download their sample dataset (`kuka_1`, `ds8`) into the platform cache on
  first use instead of reading it from the source checkout the app was built
  in.
- **`ParameterCovariance` gains `second_order`** (breaking for code that
  builds one as a struct literal: add `second_order: None`). The JSON field
  is optional, so exports written before load unchanged.

### Changed

- **The quality report gains a `covariance_linearity` check**, so a result
  that passed may now warn. It warns when the result's second-order
  prediction says a combination of its shared parameters scatters more
  than `max_second_order_scatter` (default 1.2) times its σ, and names
  the combination. A stereo rig calibrated from eight centred views tilted
  by at most 0.2 rad at 0.3 px warns at 2.8 (measured: 3.0); with the
  views tilted up to 0.5 rad and moved into the image corners it passes at
  1.06. Raise the threshold to keep the earlier verdicts.
- **Planar intrinsics initializes its camera as every other workflow does**
  (a changed result for one seed combination). A tilted (Scheimpflug)
  camera seeded with its tilt but not its intrinsics now fits the
  intrinsics, distortion and poses at the seeded tilt, as single-camera
  hand-eye, the laserline device and the rigs already did; before, they were
  fitted at the tilt of an unseeded sweep, which the seed then replaced. On
  synthetic cameras tilted by 0.20 rad (11°) and seeded within 0.05 rad, the
  solve reaches the true tilt with its covariance in 20 of 20 noise draws
  instead of 3 (the old start put the principal point about 200 px off and
  the solve ended on its bound); at 0.05 rad the results agree to 0.5 % of
  their error. Of 12 real Scheimpflug cameras seeded with their mount tilt
  alone, 11 meet their reprojection gate instead of 10 and 9 reach the
  result of the full device seed instead of 7. One camera that the old start
  calibrated now ends on the tilt bound, and with strong barrel distortion
  (`k1 = -0.30`) one noise draw in 20 starts from a degenerate linear fit.
  Every other seed combination, the unseeded init and the recommended full
  device seed give the same result as before.
- **A zero tilt seed is accepted for an untilted camera** in
  `planar_intrinsics::step_init_with_seed`, as in the other workflows: it is
  the identity a device specification gives a frontal camera. A non-zero
  tilt is still an `InvalidInput`. A seed that does not fit the camera is
  rejected before any work, and an init failure is logged for every camera
  model. The init log of a tilted camera reports `cx`, `cy` like any other.
  A step override of zero init iterations runs for a tilted camera as it
  does for an untilted one (the linear fit without its distortion rounds)
  instead of failing.
- **Rig extrinsics recovers a tilted camera calibrated from scratch as rig
  hand-eye does** (a changed result for unseeded Scheimpflug cameras).
  `rig_extrinsics::step_intrinsics_optimize_all` now solves each tilted
  camera of a fully auto-initialized rig through the wide multi-start over
  linear tilt seeds, and re-solves a Brown-Conrady tilted camera that still
  fits worse than 0.5 px from a nominal seed of the well-fitted ones. A
  camera's own staged solve could settle in a wrong tilt / principal-point
  basin that the rig solve, which holds the tilt, cannot leave. On three
  real six-camera Scheimpflug rigs the mean reprojection error falls from
  0.63, 1.03 and 0.93 px to 0.30, 0.84 and 0.59 px; on a synthetic
  four-camera rig the wrong basin put one camera's principal point 290 px
  and its tilt 14° off, and every camera now ends within 3.3 px and 0.3° of
  the truth. Where the staged solve already reached the minimum, the result
  is unchanged. The intrinsics stage takes longer: up to about 6× on the
  real rigs (2.4 s to 13.9 s), about 20 % on well-conditioned scenes. Seeded
  cameras, untilted cameras and rig hand-eye are unaffected.
- **The planar intrinsics session state no longer keeps the per-view
  homographies**, and its initial camera holds the distortion in the
  configured model. Session files written before load unchanged.

### Added

- **Dense point clouds with the calibration's uncertainty.**
  `StereoRectification::dense_cloud(&rig, &disparity, sigma_d)` (Python
  `StereoRectification.dense_cloud(result, disparity, sigma_d)`) turns a
  disparity map into a `DenseCloud`: one point per pixel with a finite,
  positive disparity, in the rig frame, with each point's covariance
  computed on demand. The covariance is the disparity's share plus the
  calibration's (both cameras' parameters and extrinsics, with their
  correlations), propagated in closed form through the matcher's
  measurement: the left pixel exact, the disparity measured along the
  rectified row. `point_covariance(i)` gives one point's,
  `covariance(&indices)` the joint covariance of chosen points (the shared
  calibration correlates them), and `PointSet::from_dense_cloud` (Python
  `PointSet.from_dense_cloud`) feeds the measurements. A calibration without
  a covariance gives points without one, with the reason. On a stereo rig
  calibrated 200 times, the depth error over the stated σ is 0.91 at
  σ_d = 0.2 px; the disparity's share alone is 2.5 times too small. A
  640 × 480 cloud builds in a few milliseconds; a point's covariance takes
  about 15 µs.
- **The second-order error of a calibration's parameters.** Every
  covariance a calibration or the optimization problem builder returns
  carries `second_order` (`SecondOrderError`; Python
  `ParameterUncertainty.second_order`): how far the error of the shared
  parameters is predicted to depart from the covariance once its
  second-order term is counted. Weak views pin some combinations down only
  through larger errors of other parameters that cancel in them; each
  parameter's σ then holds, but the joint error along such a combination is
  several times its σ, which a use of the whole covariance (a bundle
  adjustment prior, a NEES) inherits. `max_scatter_ratio` is the worst
  combination's predicted scatter over its σ, `nees_ratio` the predicted
  NEES per parameter, and `combination` names it as weights of the
  parameters in units of their σ. It is computed from the residuals'
  curvature along the covariance's own errors, without sampling: on
  synthetic stereo rigs and single cameras it matched the Monte-Carlo NEES
  within its standard error over capture strength and noise. It costs
  `s (s + 1)` residual evaluations for `s` shared parameters: 45 ms for a
  two-camera rig with 48 views (22 shared parameters, 9216 residuals),
  a third of that calibration's time, and 5 ms for a single camera with
  32 views. It is not computed above 64 shared parameters (a bundle
  adjustment's points count as shared), for noise-free data, or for a rig
  laserline device result that joins its planes to its upstream
  calibration's covariance (read the upstream export's own).
- **The covariance of the robust relative pose and essential matrix.**
  `RobustRelativePose::uncertainty` holds the covariance of the pose in the
  gauge `|t| = 1`: the rotation's right-perturbation tangent
  (`rotation.rx`, `.ry`, `.rz`, rad) and the unit translation's components
  (`direction.x`, `.y`, `.z`; rank 2, no variance along the translation),
  from the inliers' stated pixel noise; `pose_covariance()` gives it as a
  6×6. `EssentialEstimate::uncertainty` holds that of `E`'s nine entries
  (`essential.e00` … `essential.e22`, row-major, rank 5;
  `essential_covariance()` as a 9×9). Over 600 noise draws the scatter of
  each pose coordinate is 0.98–1.06 times its σ. Both functions report the
  refinement's iterations to the control's observers and stop with
  `MvgError::Cancelled`. In Python, `mvg.EssentialEstimate` and
  `mvg.RobustRelativePose` gain `uncertainty` with `essential_covariance()`
  and `pose_covariance()`, and `mvg.estimate_essential` and
  `mvg.recover_relative_pose_robust` take `progress=`.
- **Rotation blocks in the problem builder.** `optim::problem::Manifold`
  gains `So3`: a rotation stored as a unit quaternion `[qx, qy, qz, qw]`,
  updated on SO(3), with its covariance over the right-perturbation tangent
  (`rx`, `ry`, `rz` by default). `problem::so3` reads it in a residual at
  any scalar type, and `Solution::rotation` at the solution.

- **Progress and cancellation for the multiple-view-geometry solves.**
  `triangulate`, `estimate_pose`, `bundle_adjust` and
  `StereoRectification::triangulate` report each solver iteration to the
  control's observers and stop with `MvgError::Cancelled` when its token is
  cancelled; an observer does not change the result. In Python,
  `mvg.triangulate`, `mvg.estimate_pose`, `mvg.bundle_adjust` and
  `StereoRectification.triangulate` take a keyword `progress=` callable;
  raising from it, or Ctrl-C, stops the call.
- **Laser and triangulated points in the robot's gripper or base frame,
  with the hand-eye's and the robot pose's σ.** A hand-eye export's
  `calibrated_rig()` carries its hand-eye (`CalibratedRig::handeye`, a
  `core::Handeye`; `with_handeye` attaches one), and `triangulate` and
  `triangulate_laser` take `Captures::Handeye(frame)` for the frame the rig
  is fixed in or `Captures::Robot { frame, robot }` with the robot's pose
  at each capture (`core::RobotPose`, exact or with the covariance of its
  error) for the frame it moves in. The points' covariance holds the
  hand-eye's share, correlated with the cameras and laser planes as the
  calibration estimated them, and each robot pose's; on a joint rig
  hand-eye laserline calibration the scatter of laser and triangulated
  points in the base frame is 1.01 times the stated σ over 300
  calibrations. `optim::problem::Problem::add_rig_pose` adds such a pose to
  your own problem. In Python, `mvg.RobotPose` and `mvg.RobotCaptures`.

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

- [`0.16.0`](docs/changelog/0.16.0.md) - 2026-11-04
- [`0.15.0`](docs/changelog/0.15.0.md) - 2026-11-03
- [`0.14.0`](docs/changelog/0.14.0.md) - 2026-11-03
- [`0.13.0`](docs/changelog/0.13.0.md) - 2026-11-03
- [`0.12.0`](docs/changelog/0.12.0.md) - 2026-11-02
- [`0.11.0`](docs/changelog/0.11.0.md) - 2026-11-01
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
