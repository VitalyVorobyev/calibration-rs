# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`mvg::metrology`: measurements with joint σ.** `PointSet` (points and their
  joint covariance, from `triangulate`, `bundle_adjust` or your own), then
  `distance`, `angle`, `fit_plane` and `point_to_plane`, each returning its
  value with a standard deviation (`Scalar`) from the points' joint covariance,
  cross terms included, by dual-number Jacobians. A plane fit carries the
  covariance of its normal (in a stated tangent basis) and distance, and
  `point_to_plane` is joint with the plane that it measures against.

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

## [0.13.0] - 2026-11-03

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

- **`rig_handeye_laserline` takes one camera**: a single laser head on a
  robot, calibrated as a rig of one, with its camera, laser plane and
  hand-eye in one solve. Its `rig_handeye` warm start accepts one camera for
  this, and the dataset manifest's `rig_handeye_laserline` topology takes one
  or more. Zero cameras is still `InsufficientData`.

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

## Older releases

Earlier releases are archived verbatim under [`docs/changelog/`](docs/changelog/), one file per version:

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
