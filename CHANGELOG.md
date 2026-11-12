# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Migration

- **Serialized enums are snake_case** (breaking). Every enum value in a
  config, input or export is now spelled in snake_case, like the rest of the
  JSON (`"tiny_solver"`, `"eye_in_hand"` in manifests). Saved configs and
  exports written before need their values renamed:

  | Type | Before | After |
  |---|---|---|
  | `HandeyeMode` (`handeye_init.handeye_mode`) | `"EyeInHand"`, `"EyeToHand"` | `"eye_in_hand"`, `"eye_to_hand"` |
  | `HandeyeSolution`, `SingleCamHandeyeSolution`, `HandeyeManualInit` (the `mode` tag) | `"mode": "EyeInHand"` | `"mode": "eye_in_hand"` |
  | `LaserlineResidualType` (`laser_residual_type`) | `"PointToPlane"`, `"LineDistNormalized"` | `"point_to_plane"`, `"line_dist_normalized"` |
  | `RobustLoss` (`robust_loss`, `calib_loss`, `laser_loss`) | `"None"`, `{"Huber": {"scale": 1.0}}` | `{"kind": "none"}`, `{"kind": "huber", "scale": 1.0}` |

  `RobustLoss` is now tagged by `kind` like `TargetSpec`, with the variants
  `none`, `huber`, `cauchy` and `arctan`; the Python helpers
  `robust_none()`, `robust_huber(scale)` and the like build the new shape.

  ```json
  // before
  { "handeye_mode": "EyeToHand", "robust_loss": { "Huber": { "scale": 1.0 } },
    "laser_residual_type": "PointToPlane" }
  // after
  { "handeye_mode": "eye_to_hand", "robust_loss": { "kind": "huber", "scale": 1.0 },
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
    laser planes with `step_init_with_seed(PlaneManualInit { planes_cam })`
    (Python: `RigLaserlineDeviceSession.step_init_with_seed`).

    ```rust
    // before
    RigLaserlineDeviceInput { dataset, upstream, initial_planes_cam: Some(planes) }
    // after
    RigLaserlineDeviceInput { dataset, upstream }
    ```

### Fixed

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
