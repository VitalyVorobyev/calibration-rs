# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.16.0] - 2026-11-04

### Migration

- **A capture whose board planes are parallel is a typed error**
  (breaking). `init::Error` (the linear initializers' error) gains
  `DegenerateViews { views, spread_deg }`. The planar initialization of
  every pinhole workflow returns it, as `Error::Linear(DegenerateViews)`,
  when the board planes are parallel in every view (all fronto-parallel, or
  all tilted the same way): the board orientations spread less than 1°
  across the views, or Zhang's method finds no camera that fits them. It
  replaced `Error::Numerical("…invalid sign for lambda; check
  homographies")` and, for parallel boards with little distortion, an
  arbitrary focal length. Its message names the fix: tilt the board about
  both of its axes. Python raises it as a `ValueError`. Match on the variant
  instead of the old message; `estimate_intrinsics_from_homographies`
  returns it in place of `Error::Numerical` and of the `Error::Singular` of
  an indefinite solution.
- **The hand-eye initializers take a method and return both transforms**
  (breaking). `init::estimate_handeye_dlt(robot, target_se3_camera, deg)`
  is now `init::estimate_eye_in_hand(HandeyeMethod::TsaiLenz, robot,
  camera_se3_target, deg)?.gripper_se3_camera` — note the target poses,
  not their inverses — and `init::estimate_gripper_se3_target_dlt(robot,
  camera_se3_target, deg)` is `init::estimate_eye_to_hand(HandeyeMethod::TsaiLenz,
  robot, camera_se3_target, deg)?.gripper_se3_target`. The estimates also
  carry the fixed transform at the other end of the chain
  (`base_se3_target`, `camera_se3_base`). `HandeyeInit` of
  `vision-calibration-linear` is gone.
- **The dataset runner and `vcal` take a run control** (breaking).
  `dataset_runner::calibrate_dataset` and every `build_*_input` converter
  take a last argument `control: &RunControl`; pass
  `&RunControl::default()` for the previous behaviour.
  `vision_calibration_cli::run(args)` is `run(args, &control)`.
  `optim::problem::SolveOptions` gains a `control` field and is no longer
  `Copy`, `PartialEq` or `Eq`; it is still built from its `Default`. A
  cancelled run is a new variant of each error: `Error::Cancelled`,
  `optim::Error::Cancelled`, `RunError::Cancelled` and
  `CalibrateDatasetError::Cancelled` (`RunError` is matched exhaustively by
  some callers, who add the arm).
- **`RigUpstreamCalibration` has an `uncertainty` field** (breaking for a
  struct literal): the covariance of the upstream calibration the rig laser
  planes are fitted against. Add `uncertainty: Uncertainty::default()` to
  keep the upstream exact; inputs read from JSON without the field parse as
  before. The Python dataclass gains `uncertainty` (a serde payload, `None`
  by default), and the Python `RigHandeyeExport` gains `rig_se3_target`.
- **`RunError::NoWholeChessboardViews` has an `ambiguous` count**
  (breaking): the views of the camera dropped because the board's turn could
  not be matched with the view's other cameras. Code that destructures the
  variant names the new field, or ends the pattern with `..`.

### Changed

- **The rig problem types share their seed and step-result types.**
  `rig_handeye::IntrinsicsManualInit`, `RigManualInit`,
  `IntrinsicsInitResult` and `RigInitResult` are now the same types as
  their `rig_extrinsics` namesakes, re-exported under both modules; fields
  and JSON are unchanged, and so is every result. A rig manual-init error now
  names its seed and is logged for both problem types alike.
- **A rig camera no longer needs a view in common with the reference
  camera.** Rig initialization places the cameras through their
  co-visibility graph (two cameras are linked when they see the board in the
  same view), each along the shortest chain of shared views, so a ring of
  cameras where only neighbours overlap calibrates. A camera that shares
  views with the reference is placed from them as before, so existing rigs
  give the same result. A camera no chain reaches is an `InvalidInput` that
  names it.
- **A rig keeps a chessboard seen turned.** In every rig topology the dataset
  runner turns each camera's chessboard labels to agree with the other
  cameras of the view, instead of dropping every view where the board is
  seen turned by 90°. The labelling is picked from the board's direction in
  each image and the cameras' relative turn, which the views they share
  establish (the board's motion through both images tells a camera mounted
  turned or upside down); a detection whose turn cannot be told is dropped
  with the new reason `RejectReason::AmbiguousChessboardTurn` (Python
  `"ambiguous_chessboard_turn"`). OpenCV's stereo sample, whose 6×9 board is
  turned in most views, now calibrates as a rig from all 13 views instead of
  4. Where the target is fixed through the views (hand-eye and the laser
  rigs), the frame is the one camera 0 labels: a board camera 0 sees turned
  is still dropped, and a camera that disagrees with camera 0 is turned to
  agree. Data whose cameras already labelled alike gives the same input as
  before.
- A partial chessboard that also overflows the declared extent is now
  dropped as `PartialChessboard` rather than `TurnedChessboard`.
- **A rig laserline device's covariance carries its upstream calibration.**
  `RigHandeyeExport::to_upstream_calibration` (and the dataset runner, from
  the `upstream_calibration` export) now passes the hand-eye calibration's
  covariance into the upstream, in its terms: the cameras and extrinsics as
  they are, the hand-eye and target poses as each laser view's board pose.
  The planes' covariance then adds the upstream's share as consider
  parameters, which correlates the planes of different cameras, and lists
  the cameras' parameters and extrinsics with their cross terms against the
  planes, so `triangulate_laser` on the export's `calibrated_rig()` gives a
  laser point's σ with the whole calibration in it instead of reporting the
  calibration as unavailable. The planes themselves are unchanged. On a
  two-camera rig on a robot arm, the depth of a laser point scatters by 0.99
  times this σ over 300 repeated calibrations; the planes' own covariance
  alone understated it 18-fold. The robot poses count as exact. An upstream
  without a covariance holds it exact and gives the previous result, bit for
  bit.

### Added

- **`rig_upstream_calibration(export, rig_se3_target)` in Python**: the
  rig laserline upstream built from a hand-eye export, its covariance with
  it (Rust's `RigHandeyeExport::to_upstream_calibration`).
- **Hand-eye initialization methods.** `handeye_init.method` (Python
  `HandeyeInitConfig.method`, and per step `HandeyeInitOptions.method`)
  picks the closed-form solver of every hand-eye workflow: `tsai_lenz` (the
  default, unchanged), `park_martin`, `daniilidis` (dual quaternions) or
  `shah` (`AX = ZB` from the absolute poses, which also estimates the target
  pose — eye-in-hand — or the camera pose — eye-to-hand — over all views
  instead of through the first one). A config without the field reads as
  before. On synthetic captures with robot rotations from ±45° down to ±2°
  (and noise up to 0.5 px and 0.05° / 0.3 mm of robot error) the bundle
  adjustment converged from every method's estimate; Daniilidis is the most
  accurate rotation at ±2°, Shah the most accurate target pose under noise,
  and no method is better everywhere, so the default stays Tsai-Lenz. The
  book's hand-eye initialization chapter has the comparison.
- **Progress reports and cancellation.** A `RunControl`
  (`vision_calibration::control`) carries an optional `CancelToken` and
  observers. A session takes one with `CalibrationSession::set_control`;
  `calibrate_dataset`, the `build_*_input` converters and
  `optim::problem::solve` take it as an argument or option. The call reports
  each image read (`Progress::Detection`, `Progress::LaserExtraction`), each
  step `run_calibration` begins (`Progress::Step`) and each solver iteration
  (`Progress::Iteration`, with the cost), and stops at the next of these
  once the token is cancelled, returning a `Cancelled` error and no result.
  A run with observers returns exactly what it returns without them. The
  report's JSON schema is `schemas/progress.json`.
- **`vcal run` shows its progress and stops on Ctrl-C.** On a terminal one
  line of standard error follows the detection, the steps and the solver
  iterations; Ctrl-C stops the run without writing files and exits with
  status 130.
- **Python: a `progress` callable and Ctrl-C.** The `run_*` functions,
  `calibrate_dataset` and the `build_*_input` converters take
  `progress=`, and every session `set_progress(...)`: the callable receives
  each report as a dict (typed `types.Progress`). Ctrl-C during a long call
  raises `KeyboardInterrupt` once the Rust side has stopped, and an
  exception the callable raises stops the call and propagates.
- **Sample `opencv_stereo`**: both cameras of OpenCV's stereo calibration
  sample (13 views of a 6×9 chessboard, 640×480, Apache-2.0), as a rig.
- The `measure_board` examples (Rust and Python) take the sample to measure:
  `opencv_fisheye_stereo` (the default) or `opencv_stereo`, a pinhole pair.

### Fixed

- **A lone corner no longer pads a partial chessboard into a whole one.**
  The detector can label a stray corner beside the board as the lattice's
  first column, which closed the board's extent while the board's real edge
  was missing and shifted its labels by a column; a rig then diverged on
  OpenCV's stereo sample declared the other way round. A chessboard
  detection now needs at least half of each extent edge's corners (and at
  least two); one that lacks them is dropped as `PartialChessboard`.
- **Hand-eye initialization with motions past 120°.** A pair of views whose
  relative rotation exceeds 120° could enter the linear rotation solve with
  the wrong quaternion sign; from few views the initial hand-eye rotation
  came out 180° off. Data whose relative rotations stay below 120° gives the
  same result as before.

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

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
