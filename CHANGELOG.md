# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- **`RunError::NoWholeChessboardViews` has an `ambiguous` count**
  (breaking): the views of the camera dropped because the board's turn could
  not be matched with the view's other cameras. Code that destructures the
  variant names the new field, or ends the pattern with `..`.

### Changed

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

### Added

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
- **Sample `opencv_stereo`**: both cameras of OpenCV's stereo calibration
  sample (13 views of a 6×9 chessboard, 640×480, Apache-2.0), as a rig.
- The `measure_board` examples (Rust and Python) take the sample to measure:
  `opencv_fisheye_stereo` (the default) or `opencv_stereo`, a pinhole pair.

### Fixed

- **Hand-eye initialization with motions past 120°.** A pair of views whose
  relative rotation exceeds 120° could enter the linear rotation solve with
  the wrong quaternion sign; from few views the initial hand-eye rotation
  came out 180° off. Data whose relative rotations stay below 120° gives the
  same result as before.

## [0.15.0] - 2026-11-03

### Migration

- **The PnP solvers moved to `vision-geometry`** (breaking, for users of
  `vision-calibration-linear`). `vision_calibration_linear::pnp` is now
  `vision_geometry::pnp`, with the crate's `GeometryError`; the facade's
  `init::pnp` is unchanged. `PnpSolver`, a struct whose associated functions
  repeated the free functions, is removed: call `pnp::dlt`, `pnp::p3p`,
  `pnp::epnp` and `pnp::dlt_ransac` directly.
- **`LaserPlane` is valid by construction** (breaking, `optim::LaserPlane`).
  `LaserPlane::new(normal, distance)` returns a `Result` and rejects a
  normal that is not of unit length (within `1e-9`) or a value that is not
  finite, instead of normalizing the normal and keeping the distance, which
  moved a plane given with a normal of another length. JSON is validated the
  same way on the way in (exports written by the library are unchanged and
  read back as before). `LaserPlane::from_coefficients(n, d)` takes a plane
  equation `n · p + d = 0` with any nonzero `n` and divides both by `|n|`.
  The fields are private: read them with `normal()` and `distance()`.

- **`LaserPlane` moved to `vision_calibration::core`** (breaking). It is
  no longer in `vision_calibration::optim`; import it from `core`. Its JSON
  is unchanged, and so is every export.
- **`UncertaintyUnavailable::CalibrationWithoutCovariance` also means a
  calibration whose covariance omits the cameras a result uses** (a laser
  calibration that held the cameras' intrinsics fixed), not only one without
  a covariance.

### Added

- **`measure_board` example and book chapter.** Rust and Python examples calibrate
  the fisheye stereo sample on half of its views, triangulate the board
  corners of the other half and compare the distance errors with the stated
  σ, with and without the calibration's share; the chapter "Measuring a
  Board with a Stereo Calibration" gives the measured table and discusses it.
- **`vision_calibration.mvg`: the multiple-view geometry surface in Python.**
  Triangulation, object pose, bundle adjustment, laser points, two-view
  pose recovery, measurements (`PointSet`, `distance`, `angle`, `fit_plane`,
  `point_to_plane`), stereo rectification and block matching take a
  calibration result and return the Rust results with their covariances as
  numpy arrays; poses are `4x4` matrices. The Rust MVG input types
  (`PixelMatch`, `RigObservation`, `PoseObservation`, `LaserObservation`,
  `RectifiedMatch`, `Scale`) now implement `Serialize` and `Deserialize`, and
  the result types (`Triangulation`, `ObjectPose`, `RelativePose`,
  `EssentialEstimate`, `RobustRelativePose`, `PlaneFit`, `Scalar`,
  `LaserTriangulation` and their parts) implement `Serialize`.
- **`mvg::laser`: laser points with their covariance.**
  `triangulate_laser` takes a `CalibratedRig` that carries each camera's laser
  plane and the laser pixels (each with its stated noise), and returns the
  points in the rig frame with the joint covariance of the pixels, the
  camera, the plane and the camera's extrinsic (their correlations
  included), exact to first order. `PointSet::from_laser_triangulation`
  feeds it to the measurements. `CalibratedRig::with_laser_planes` attaches
  the planes; the laser workflows' exports (`laserline_device`,
  `rig_laserline_device`, `rig_handeye_laserline`) give a rig with their
  planes from `calibrated_rig()`.
- **`mvg::pose`: the pose of a known object with its covariance.**
  `estimate_pose` takes known object points with their pixels in one or
  several rig cameras (each with its stated noise) and returns the pose in the
  rig frame, the inliers and the 6x6 covariance of the pose
  (`ObjectPose::pose_covariance`), the calibration's share included. The
  closed-form start comes from the camera with the most observations; with
  `PoseOptions::ransac` it is P3P inside a RANSAC whose threshold is in σ of
  the stated noise (the same for every camera model); the refinement uses
  every inlier of every camera.
- **`mvg::metrology`: measurements with joint σ.** `PointSet` (points and their
  joint covariance, from `triangulate`, `bundle_adjust` or your own), then
  `distance`, `angle`, `fit_plane` and `point_to_plane`, each returning its
  value with a standard deviation (`Scalar`) from the points' joint covariance,
  cross terms included, by dual-number Jacobians. A plane fit carries the
  covariance of its normal (in a stated tangent basis) and distance, and
  `point_to_plane` is joint with the plane that it measures against.

### Fixed

- **EPnP** returned points behind the camera for some shallow objects (the
  sign of its null vector was arbitrary); it now returns the pose with the
  points ahead. Points on a plane (a calibration board) are solved from the
  plane's homography, where the control-point formulation had no solution, and
  collinear points are a `Degenerate` error.
- **A rig's unobserved camera in a considered calibration.** Solving a problem
  that adds a calibrated rig as consider parameters failed with a missing
  Jacobian column when some camera of the rig had no observation in it; such a
  camera now adds nothing to the covariance.

## [0.14.0] - 2026-11-03

### Migration

- **Calibrated two-view functions take pixels with their noise** (breaking, in
  `vision-mvg` and the facade's `mvg`). `robust::estimate_essential`,
  `robust::recover_relative_pose_robust` and
  `pose_recovery::recover_relative_pose` take `&[PixelMatch]` instead of
  `&[Correspondence2D]`: two pixels, each with its `PointCovariance`.
  `Correspondence2D` stays for the fundamental-matrix and homography
  estimators.
  - The residual is the covariance-weighted Sampson error: the epipolar
    constraint over its first-order standard deviation, with each pixel's
    covariance mapped onto its bearing through the camera model. The RANSAC
    `thresh` and `inlier_rms` of the essential-matrix functions are in **σ**
    of the stated noise instead of radians; `3.0` keeps about 99.7 % of the
    inliers of Gaussian noise, at any focal length.
  - The essential matrix is the least-squares fit of that error to its
    inliers (a linear fit polished on the essential manifold), no longer
    the best five-point sample, so `inlier_rms` is about 1 when the stated
    noise describes the pixels, and estimates change.
  - Migrate with
    `PixelMatch::new(pt1, PointCovariance::isotropic(0.5)?, pt2, PointCovariance::isotropic(0.5)?)`
    (state the σ your detector reaches), and a radian threshold `t` becomes
    about `t · f / σ` in σ for focal length `f` px and pixel noise `σ` px
    (`1.0e-3` rad at `f = 800` and `σ = 0.5` is about `1.6`).
- **Triangulation states its noise and returns a `Triangulation`** (breaking, in
  `vision-mvg` and the facade's `mvg`). `RigObservation::new` takes the
  pixel's `PointCovariance` as a fourth argument
  (`PointCovariance::isotropic(sigma)?`, or `rig.reprojection_noise(camera)?`
  for the calibration's noise level). `triangulate` returns a `Triangulation`
  whose `points` is the old vector, and the points are now refined on the
  reprojection error (weighted by the stated noise) instead of left at the
  linear estimate, so their values change. `vision-mvg` depends on
  `vision-calibration-optim`.

- **Bundle adjustment takes a calibration and states its gauge** (breaking, in
  `vision-mvg` and the facade's `mvg`). `bundle_adjust(&rig, &rig_se3_world,
  &points, &tracks, &opts)` replaces
  `bundle_adjust(&intrinsics, &observations, &init_poses, &init_points, &opts)`:
  - Cameras come from a `CalibratedRig`, not a `K` per camera, so every
    camera model works. What was a camera is now a **capture** of the rig:
    `init_poses` (`T_C_W` per camera) become `rig_se3_world` (world → rig per
    capture), and a rig of several cameras shares each capture's pose.
  - `BundleObservation::new(cam, point, pixel)` is gone: observations are the
    triangulation's `RigObservation::new(capture, camera, pixel, noise)`
    grouped into `tracks`, `tracks[i]` observing `points[i]`. The pixel noise
    is stated (`PointCovariance::isotropic(sigma)?`) and weighs each pixel.
  - The gauge is explicit. Capture 0 is always fixed (the old
    `fix_first_camera` is gone: it fixed the lowest observed camera, and
    could be switched off), and the scale must be stated in
    `BundleAdjustOptions::new(Scale::Rig)` or
    `Scale::Distance { points, distance, sigma }`; there is no default. A
    scale the data cannot fix is an `InvalidInput` before any solve.
  - `BundleAdjustmentOptions { max_iterations, fix_first_camera }` is
    `BundleAdjustOptions { calibration, loss, scale, max_iters }`, and
    `BundleAdjustmentResult { poses, points, initial_rms, final_rms,
    per_camera_rms }` is `BundleAdjustment { rig_se3_world, points, rig,
    uncertainty, report, reprojection_rms_px }`. `reprojection_rms_px` is the
    RMS distance (`√2 σ` for isotropic noise); `initial_rms` and
    `per_camera_rms` are gone.
  - Under-observed points are an error now: a track needs two observations.
- **Stereo rectification works from the calibration** (breaking, in
  `vision-mvg` and the facade's `mvg`). `rectification::rectify(&rig,
  [left, right], &RectifyOptions)` replaces
  `rectify_stereo_pair(&RectifyCamera, &RectifyCamera, &cam1_se3_cam0, &opts)`;
  `RectifyCamera` is removed.
  - The cameras come from a `CalibratedRig`, and **raw pixels go in**: there
    is no separate undistortion, and every camera model works, a Scheimpflug
    sensor through its own model instead of a tilt homography.
  - `StereoRectification` has accessors in place of fields:
    `rectify_left`/`rectify_right` are `rectified_pixel(side, raw)`,
    `baseline` is `baseline()`, `k_rect` (a matrix) is `k_rect()` (an
    `FxFyCxCy`), `r_rect` is `rect_r_cam(side)`. `RectifyOptions::k_rect` is
    an `Option<FxFyCxCy>` and `RectifyOptions` gained `scale`; set the fields
    on `RectifyOptions::default()`.
  - A rectification with a zero baseline or an invalid `scale` or `k_rect` is
    an error before any work; the Scheimpflug tilt limit is gone with the
    homography.
- **The `refine` features are removed** from `vision-mvg` and the facade, with
  the `tiny-solver` dependency of `vision-mvg`: bundle adjustment and
  `refine::refine_homography` are always available, and run on the problem
  builder. Drop `features = ["refine"]` from your manifest.
  `MvgError::NotConverged` is removed with the last refinement that raised it.

### Added

- `bundle_adjust` gives the joint covariance of the capture poses and the
  points in its gauge (`BundleAdjustment::point_covariance`,
  `pose_covariance`), with the calibration's share: the rig's cameras and
  extrinsics enter as consider parameters (`Use::Consider`) or are refined
  under a prior with their covariance (`Use::Prior`, the refined rig in
  `BundleAdjustment::rig`). Every reprojection can carry a robust loss
  (`BundleAdjustOptions::loss`).
- `Triangulation`, from `triangulate`: the joint covariance of the points
  (`uncertainty`, labels `p{i}.x`, `p{i}.y`, `p{i}.z`), with
  `point_covariance(i)` and `cross_covariance(i, j)`. It includes the
  calibration's share, the rig's cameras and extrinsics entering as consider
  parameters, which a camera taken as exact leaves out (depth σ 3.2 times too
  small in the book's stereo example). A rig without a covariance gives
  refined points and an uncertainty that says so.
- Stereo from a calibration, in `mvg::rectification`:
  `StereoRectification::rectify_image` resamples a raw image into the
  rectified frame; `disparity_to_depth` is the matrix `Q`; `point` gives the
  3D point of a rectified pixel and disparity in the left camera's frame;
  `disparity_covariance` its covariance from the disparity's σ (the scatter of
  the depth over the stated σ is 0.97 to 1.06 over 2000 draws);
  and `triangulate` measures matches with the covariance of the calibration
  included (scatter over stated 0.95 over 60 calibrations; the σ from the
  observations alone is 3.2 times too small). The desktop app's Depth
  workspace uses it, so depth works for every camera model, not only
  untilted Brown-Conrady cameras.
- `PointCovariance::isotropic(sigma_px)`: the covariance of one σ on both
  axes.
- `CalibratedRig::reprojection_noise(camera)`: the calibration's reprojection
  noise level as a `PointCovariance`, or an error naming why the calibration
  has none to give.

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

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
