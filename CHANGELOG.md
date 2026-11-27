# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `analysis::projection_difference` compares two calibrations of one camera,
  of the same model or of different ones, through where they put the same
  ray: at each node of the first camera's `ImageGrid`, the second camera's
  pixel minus the first's (`b − a`), after fitting the rotation of the
  second camera's frame that brings the two closest (`Compensation::Rotation`,
  the default; `Compensation::None` compares the frames as calibrated). The
  rotation is fitted in pixels on the grid blocks the observed features
  cover (`FitRegion::Covered`) or on every node (`FitRegion::WholeImage`).
  `ProjectionDifference` holds the plane, the fitted rotation
  (`camb_so3_cama`), the fitted nodes, and the RMS and largest difference
  over the fitted nodes and over the image; nodes where either camera has no
  ray or projection are NaN and take no part in the fit. Python:
  `vision_calibration.projection_difference` (numpy planes).

- Detection detail, for finding wrong detections: `detect::DetectorConfig::detect_detailed`
  returns a `DetectionDetail` with the features `detect` returns (bit for
  bit), the image's outcome (detected; not found, with the detector's
  reason; dropped as ambiguous; or every feature filtered out), the overlay
  data of each feature for every target kind (corner labels, ids, scores
  and grid-line directions; ChArUco markers and the marker cells that read
  as another marker; PuzzleBoard code bits with whether each matches the
  master pattern, and the decode summary; ring-grid ellipses, centre
  estimators and photometric centre covariances, and a plain board's
  predicted origin dots) and the candidates left out with their reasons.
  `dataset_runner::detect_target_detail` runs it from a manifest target and
  `dataset_runner::detection_detail` returns the detail of a manifest run's
  detection of one image, through the run's cache key and region of
  interest; `FsDetectionCache` stores it beside the features as
  `<key>.detail.json` (`DetectionCache::get_detail`/`put_detail`, defaulted,
  so an existing cache keeps compiling), and a calibration never reads it.
  `RunError::ImageNotInDataset` and `RunError::DetailMismatch` are its
  errors. Published schema `schemas/detection_detail.json`; Python
  `dataset_runner.detect_target_detail`, `dataset_runner.detection_detail`
  and `DetectionDetail`. Enabling it turns on `calib-targets`' `diagnostics`
  feature under the `charuco` and `puzzleboard` features; detection results
  and speed are unchanged.

- `device_seed::DeviceSeed`: the device quantities a calibration starts
  from, by camera id — each camera's lens, its mount in the rig, its laser
  plane and the rig's hand-eye — from a device spec
  (`DeviceSeed::from_spec`) or from any calibration export
  (`DeviceSeed::from_calibration`, the camera ids and image sizes given).
  A calibrated lens is the camera whole; `nominal()` keeps its intrinsics
  and tilt for a session of another camera model, and `select()` takes some
  of the cameras. Its derivations give the manual-init seeds of every stage
  (`intrinsics_init`, `rig_intrinsics_init`, `cam_se3_rig`, `rig_init`,
  `handeye_init`, `single_cam_handeye_init`, `device_init`, `plane_init`),
  with the mounts and the hand-eye re-expressed in the frame of the
  configured reference camera. A seed is an initial value, never a prior,
  and carries no uncertainty. It has a JSON form and a published schema
  (`schemas/device_seed.json`); `DeviceSeedError` gains the variants of its
  checks. Python: `vision_calibration.DeviceSeed` (`from_export`,
  `from_spec`, `from_payload`, `nominal`, `select`). The existing
  device-spec functions return the same values as before.
- Facade re-exports under `vision_calibration`:
  `core::{CameraProject, DistortionModel, IntrinsicsModel, NoDistortion, ProjectionModel}`,
  `dataset::validate`, `mvg::refine`, and `mvg::solvers::essential_linear`.
- Documented `new()` constructors across user-constructible `#[non_exhaustive]`
  manifest, specification, option, and manual-initialization types in `core`,
  `dataset`, `detect`, `init`, `optim`, `mvg`, `analysis`, `common`, and the
  calibration workflow modules.
- `RigLaserlineDeviceExport::per_cam_reproj_errors`: the per-camera mean
  target reprojection error over every target view of the camera, as the
  other calibration exports carry it (`per_cam_stats[i].mean_reproj_error`
  averages only the views that also have laser pixels). Optional on the wire.
- Desktop app hardening: an error fallback for workspace and application
  errors, rejection of duplicate in-flight calibration runs, atomic staging
  for `save_run`, and accurate cache-hit reporting (`cache_used`).

- `analysis::fit_focus` fits the plane of sharp focus to the per-point blur
  of a calibrated planar dataset (chessboard or PuzzleBoard corners measured
  with `measure_blur`), with the camera and target poses held: the
  blur `σ` is predicted as `√(σ0² + 1/12 + (κ·s·d)²)` plus one offset per
  image (the offsets summing to zero), `d` the defocus from the plane
  `π·X = 1` along the sensor normal and `s` the spot's local scale (the
  entrance pupil in the lens plane: distortion's change of magnification and
  a tilted sensor's oblique view of the pupil). `FocusDiagnostic` is the
  `FocusFit` (the plane, the blur scale and the in-focus blur with their
  covariance; the image line where the focus reaches infinity with its
  covariance; the depth-of-field wedge for a blur limit set in
  `FocusOptions`, its near and far planes with their covariances; the
  noise scale, per-image residuals and offsets, and the blur points left
  out because they do not project); the same fit as `NotCrossed` when no
  image crosses the plane (its direction is determined, its position along
  the sensor normal rests on the images' common blur level); or why there
  is none: no blur, too few points, or views that do not determine it.
  `FocusOptions::view_jackknife` adds the delete-one-view jackknife
  covariance (`FocusJackknife`), which covers blur errors correlated
  within an image; the stated covariance assumes them independent, and on
  real detectors has been measured up to 3× short. Points are weighted by
  their stated blur variance, or all equally when none states one; a mix
  is the error `Error::MixedBlurCovariance`. A pinhole projection is
  required; a lens tilt is not derived (it needs the pixel pitch and the
  pupil magnification). Nothing reads the result: it is asked for. Python:
  `vision_calibration.fit_focus`. The JSON schemas are
  `schemas/focus_options.json` and `schemas/focus_diagnostic.json`.
- `analysis::compare_camera_models` compares candidate camera
  configurations of a planar intrinsics calibration on views each fit did
  not see. The views are split into folds (`CrossValidationOptions`:
  five by default, assigned by a seeded shuffle that gives the same folds
  on every platform); each candidate is calibrated on all folds but one,
  and every held-out view is scored with that fit's camera held fixed. The
  `ModelComparison` gives each candidate's training and held-out
  reprojection error, per fold and per view, its free parameter count, its
  difference from the best with a fold-paired standard error, and a
  ranking by held-out error; a candidate that cannot be fitted is reported
  as failed without stopping the others. Python:
  `vision_calibration.compare_camera_models`. The JSON schemas are
  `schemas/cross_validation_options.json` and
  `schemas/model_comparison.json`. `ReprojectionStats` gains a JSON
  schema.
- The quality report has a `parameter_conditioning` check per camera
  (`CheckKind::ParameterConditioning`). It measures how many times each
  camera parameter's σ is inflated by its correlations with the camera's
  other parameters, `1/√(1 − R²)` (a focal length without the other one, a
  radial coefficient without the other radial coefficients), and warns above
  `QualityThresholds::max_sigma_inflation` (default 7, about a pair at the
  0.99 correlation limit), naming each such parameter with the parameters
  it trades off against. It catches trade-offs among three or more
  parameters that the pairwise `parameter_correlation` check misses. A
  flagged Scheimpflug tilt gets its cause: free tangential or thin-prism
  terms (with how much holding them would cut its σ), or distortion too weak
  or too little observed to tell the tilt from the principal point. The
  check warns, never fails; thresholds written without the new field read
  its default. Python: `QualityThresholds.max_sigma_inflation`.
- **A confidence floor for ring-grid detection.** `RinggridConfig::min_confidence`
  and the manifest's `target.min_confidence` (optional, in `[0, 1]`) raise
  the detector's floor on a reported marker's confidence; left out, the
  detector's own floor of 0.03 applies and nothing changes. The confidence
  is a product of factors in `[0, 1]` (decode and outer-fit quality, rim
  coverage, inner-edge agreement, edge sharpness, distance to the border),
  so a floor `c` also keeps only markers whose outer edge is at most
  `3 · √(1/c − 1)` px wide (0.17: 6.6 px). An out-of-range value is
  rejected before detection.
- **Measured blur per corner.** Chessboard and PuzzleBoard targets measure
  the blur at each corner on request: `measure_blur` on `ChessboardConfig`
  and `PuzzleboardConfig` and on the manifest's chessboard and PuzzleBoard
  targets (off by default: it costs a few milliseconds per corner). Each
  corner then carries `Feature::blur` (`detect::FeatureBlur`), the
  covariance `Σ` of the Gaussian point-spread function in px² as
  `[xx, xy, yy]`, including the pixel's area integration
  (`Σ = Σ_lens + I/12`), with the 3×3 covariance of those elements; a
  corner the estimate declines has none. The blur travels through the
  detection cache and into the observations (`CorrespondenceView::with_blur`,
  `blur`, `blurs`; `core::PointBlur`, with `sigma_px() = √(tr Σ / 2)`; its
  covariance must be symmetric and positive semi-definite, all zeros
  stating no precision), and in JSON as the view's optional `blur` list.
  Python: the target's `measure_blur`, `Features.blur_psf_px` and
  `Features.blur_cov`, `Observation.blur` and `PointBlur`. Corner positions
  are the same with or without it, and no calibration reads it yet: results
  are unchanged. The detector drops an estimate that would not be valid; a
  cached feature whose blur is not valid (not finite, `Σ` not positive
  definite, its covariance not semi-definite) fails its image with a
  detection error.
- Every export from `dataset_runner::calibrate_dataset` (and so from
  `vcal run` and Python's `calibrate_dataset`) carries
  `diagnostics.image_manifest`: the image of each `(view, camera)` the
  calibration used, and of each laser frame, in the input's view order. Its
  root is `.`, the dataset folder, and each frame's path is relative to it.
  `vcal run` restates the root from the folder it writes `calibration.json`
  to: relative when that lies inside the dataset folder, the absolute
  folder otherwise. An optional field: the schemas are unchanged.
- `core::ImageManifest::rebased` restates a manifest's root for an export
  moved to another folder, relative when both lie inside a folder that
  moves as one; `dataset_runner::CalibrationExport::diagnostics_mut` gives
  the export's diagnostics to change before writing it.
- `dataset_runner::ConfigPatch`: changes to a workflow's default
  configuration, keyed like its config, read from TOML or JSON
  (`from_toml_str`, `from_json_str`); what it names replaces the default
  and what it leaves out keeps it. `CalibrationConfig::patched` applies one
  to a topology's defaults and checks the result, `ConfigPatch::from_config`
  gives the smallest patch that makes a configuration, and `then` layers one
  patch over another. `vcal run -c` reads its file as one, with the same
  results as before. Python: `dataset_runner.patched_config` (a patch is a
  mapping, `dataset_runner.ConfigPatch`).
- `dataset_runner::RejectReason::Excluded`: a detection the run was told to
  leave out, unread (Python `"excluded"`). `RejectReason` serializes in
  `snake_case`. The dataset runner can now leave out whole views, one
  camera's image of a view, or single features named by their target
  point, the groundwork of project exclusions; every converter and
  `calibrate_dataset` give the same results as before when nothing is left
  out.
- `analysis::distortion_map` shows what a camera's lens and sensor model do
  to each ray over its image, on an `analysis::ImageGrid` (block centres,
  row-major): three layers in pixels, `distortion` (the lens without a sensor
  tilt), `sensor` (what a tilted sensor adds) and `total` (their exact sum),
  each `actual − reference` at the same ray, so barrel distortion points
  inward. A fisheye's reference is its `f·θ` image. Nodes past a fold of the
  distortion, outside a fisheye polynomial's invertible range, or sent to
  infinity by the sensor are NaN. Python: `vision_calibration.distortion_map`
  (numpy planes).
- `analysis::residual_map` estimates the systematic part of a camera's
  reprojection residuals over its image: a local-linear mean of
  `observed − projected` (each record at its projection), the local RMS,
  a standard error clustered by view and a pointwise significance, with the
  kernel's bandwidth cross-validated over views (`ResidualMapOptions`,
  `Bandwidth`, `BandwidthSelection`). Gross errors are left out and listed.
  The significance is calibrated for the number of views (Hotelling's `T²`
  reported as a two-degree-of-freedom `χ`, so `χ > 3.03` is `p < 0.01` with
  few views as with many); it and the standard error are conditional on the
  fitted camera and poses. Python:
  `vision_calibration.residual_map`.
- `TargetFeatureResidual::residual_px` returns a record's residual vector,
  `observed_px − projected_px`, the sign the solver minimizes.
- Calibration projects (`project`): a folder holding
  `calibration-project.toml` (`ProjectSpec`), the runs calibrated from it
  and its detection cache. The project file names its dataset, written in
  it (with the folder its paths resolve against) or kept as a manifest file
  of its own that the library never writes (`DatasetSource`); the device it
  calibrates (`DeviceIdentity`: `product`, `id`, `label`); changes to the
  workflow's default configuration (`ConfigPatch`); the quality thresholds;
  and what to leave out (`Exclusions`: views by pairing token or image,
  one camera's image, single features by target point). An unknown field is
  an error. `Project::open`, `create`, `save` (atomic; refused with
  `ProjectError::ChangedOnDisk` when the file changed since it was read)
  and `reload`; `revision` names the content read, for a caller that keeps
  no `Project` between an edit and its save; `check` runs every check a run
  makes before it reads an image, a stale exclusion included
  (`ProjectError::Exclusion`). A project of the desktop app's earlier format
  (`calibration-project.json` version 1 with `dataset.json`) opens as is and
  is written as version 2 on its first save. The schema is
  `schemas/project_spec.json`.
- Project runs. `Project::calibrate` checks the project, detects through
  its cache (`<project>/.detections`), calibrates and writes
  `runs/run-NNN/`: `export.json`, `quality.json` and `run.json`, with
  `notes.json` and `report.html` when asked for (`ProjectRunOptions`: the
  run's own configuration changes over the project's, redetection, the
  page, the notes, the tool's name). A run is written whole or not at all,
  a cancelled one writes nothing, and run numbers are claimed so that
  several programs can write into one project at once. `run.json`
  (`RunRecord`) is written once: the tool and library version, the inputs
  as resolved (manifest, exclusions, configuration changes and the
  configuration they made, quality thresholds), what detection left out
  and why (excluded images included), feature exclusions that matched
  nothing, and a summary (verdict, each camera's RMS error, features and
  views). Its `InputDigest` hashes the manifest (with its robot pose and
  upstream calibration files), the exclusions, the configuration, the
  thresholds and the detection-cache keys read: equal digests and library
  versions give byte-identical `export.json` and `quality.json`; the
  record's time stamp and duration are never hashed, and moving the data
  changes no digest. The export's image root is restated for the run
  folder, so a moved project still finds its images. `Project::runs`
  (`RunInfo`, with the desktop app's earlier runs listed as `Unrecorded`),
  `run`, `set_run_notes` (`RunNotes`, the only file of a run that changes),
  `delete_run`, and `import_export`, which adds an export calibrated
  elsewhere (a file, a `vcal run -o` folder) after checking its workflow
  and camera count. Schemas `schemas/run_record.json`,
  `schemas/run_notes.json`, `schemas/run_info.json`.
- `project::scan_library` finds the projects under a set of folders (no
  symbolic links or hidden folders, no search inside a project, a depth
  limit in `LibraryScanOptions`, default 8) and reads each with its runs;
  a project or folder that cannot be read is listed as a `ScanFailure`,
  never an error. `LibraryScan::devices` groups the runs of every device by
  the project file's `[device]` `product` and `id` (`DeviceHistory`),
  oldest first. Nothing is indexed: the folders are the library. Schema
  `schemas/library_scan.json`.
- The project types and functions are `vision_calibration::project` in the
  facade. Python: `vision_calibration.project` with the `Project` class
  (`open`, `create`, `save`, `reload`, `spec`, `check`, `calibrate`,
  `import_export`, `runs`, `run`, `set_run_notes`, `delete_run`, ...),
  `scan_library` (with each device's history) and `ProjectError`, whose
  `kind` names the failure; the payloads are the generated
  `types.ProjectSpec`, `types.RunInfo`, `types.RunRecord`,
  `types.RunNotes` and `types.LibraryScan`. The book has a new chapter,
  Projects.
- `vcal` runs projects. `vcal check` and `vcal run` take a project when the
  path is a project folder or its project file (a manifest otherwise, with
  the same files, output and exit statuses as before); a project's run goes
  to its next `runs/run-NNN/`, `-c` changing the configuration for that run
  only and `--label` naming it (`-o` with a project, or `--label` with a
  manifest, is a command-line error, status 2). `vcal project init`
  references a manifest (`--manifest`, `--embed` to copy it in) or drafts
  one from the file names, and refuses a folder that already holds a
  manifest; `vcal project runs` (`--json`), `import`, `note` (`--label`,
  `--notes`, `--star`, `--unstar`) and `scan` (`--json`, `--max-depth`).
  A folder passed to `vcal run` or `vcal check` without a project file is
  still read as a manifest, and the error now names `vcal project init`.
- The repository's datasets (`data/kuka_1`, `data/DS8`,
  `data/opencv_stereo`, and the left, right and rig manifests of
  `data/stereo`) have calibration projects referencing their manifests, so
  `vcal run data/kuka_1` keeps its runs in `data/kuka_1/runs/`.
- `dataset_runner::input_files` lists every file a run of a manifest reads
  (images, laser frames, the robot pose file, the upstream calibration)
  without reading any. Python: `dataset_runner.input_files`.
- `Project::input_digest` (`ProjectDigest`; Python
  `Project.input_digest()`) gives the digests of a project's dataset,
  exclusions, configuration and quality thresholds as they stand, as a run
  records them, without reading an image: a run whose digest differs in one
  of them was made from other inputs than the project now states.

### Changed

- The HTML report's "mean error by image region" figure (errors binned at
  the observed pixel) is replaced by two figures from one residual map: the
  systematic error `observed − projected`, greyed where it is not
  significant, and the local RMS error.
- The desktop app is named Calibration Studio and lives in `apps/studio/`.
- Calibration Studio reads a picked or Quick Start manifest (TOML or JSON)
  with the library's manifest reader, as `vcal` does: a file that is not a
  manifest is refused when it is loaded, with the reason, rather than when
  the run starts.
- Calibration Studio's manifest and config forms, and the defaults they
  start from, come from the calibration library the app runs on, so they
  always match it; a preset's config changes apply by the library's rule,
  as `vcal run --config` applies a file.
- Calibration Studio shows TIFF, PNM and 16-bit frames (as the detectors
  see them) besides the formats a browser reads, caches what it derives from
  them, and computes a dense disparity map as a background job; the Depth
  view colours disparity and depth with the viridis colour map.
- Calibration Studio runs in a browser too: `calibration-studio-server`
  serves the whole app (projects and calibration runs included) from one
  binary with the page inside it, for a calibration station or a device on
  the network. It serves only its `--root` folders; off the machine itself
  it requires its token (printed in the start-up address, kept as an
  HttpOnly session cookie), answers only its own Host and page, and sends no
  cross-origin headers. It replaces the single-run browser viewer
  (`calib-viewer`). Recent projects are the back end's.
- Calibration Studio works on the library's calibration projects, the same
  folders `vcal` and Python use: a run always belongs to a project and is
  written by the library (record, export, quality report, report page), the
  Run workspace edits and saves the project's manifest and configuration
  (refusing to overwrite a project file changed on disk since it was opened),
  and every viewer names the run it shows. Runs take a label and a star, open
  their report, can be deleted, and say when they were made from other
  inputs than the project now states. Quick Start samples open as projects
  of their own; a dataset folder (its manifest drafted from its files) or a
  manifest file becomes a project from the Project menu; an export
  calibrated elsewhere is imported into a project as its next run; projects
  of the app's earlier format open and are written in the current format
  when first saved. The built-in preset list is gone.
- Calibration Studio has a new frame: a rail of seven screens (Library,
  Dataset, Detect, Calibrate, Inspect, Compare, Device; `⌘1` to `⌘7`) and a
  context bar that names the project and the current run with what that
  run is (the state of its files, its verdict in words, and whether the
  project changed since). The diagnose, 3D, epipolar and depth workspaces
  are the tabs of Inspect (Frames, Scene, Epipolar, Depth) and the Run
  workspace is Calibrate; the project, run and tab are in the address, so a
  link reopens the same place. A job indicator and tray follow every job
  (including another tab's) with its stage, a Cancel and the solver's cost;
  a change another tab or program makes to the open project arrives without
  a reload; a dropped connection says so and pauses what writes until it is
  back. `⌘K` opens a palette, `[` `]` `,` `.` step views and cameras, `?`
  lists the shortcuts; the server's sign-in is a page of its own. The old
  addresses still open. The screens beyond Inspect and Calibrate are not
  built yet and say so.
- Calibration Studio reads and writes only inside its root folders. The
  desktop app starts with none and asks you to add one (Settings lists and
  removes them); a project or manifest that names a path outside them is
  refused. Folders, projects, manifests and exports are chosen in an in-app
  browser over the roots, on the desktop and in a browser alike.
- **Desktop app IPC types (breaking):** `DisparityResult::plane_rms` (`planeRms`
  in TypeScript) is now `Option<f64>` (`null` when fewer than three finite
  disparities exist or the plane fit is singular/non-finite; it was an
  infinite value that serialized as `null` under a `number` type),
  and `undistort_points` now returns `Vec<Option<[f64; 2]>>` (`null` for pixels
  whose undistortion fails or is non-finite instead of `[NaN, NaN]`).
- **Desktop app run validation:** `DatasetSpec` validation and config topology
  validation now run before target detection rather than after.
- Performance and numerical improvements across optimization and MVG:
  single-pass Newton undistortion Jacobian evaluation for `BrownConrady5`,
  `RationalPolynomial`, and `ThinPrism`; stack-allocated 4×4 eigensolve in
  `mvg::solvers::triangulate_point_linear`; single-SVD decomposition in
  `mvg::solvers::decompose_essential`; analytic plane-fit Jacobian in
  `mvg::metrology`; buffer reuse, rolling SGM row buffers, and single-pass
  left-to-right ZNCC consistency reuse in `mvg::dense`; sparse 6×6 block
  hand-eye Jacobian propagation in rig laser-line bundle covariance; and
  skipping intermediate covariance solves during internal tilt-sweep and
  warm-start stages. Warm-start candidates are no longer rejected for a
  residual or Jacobian that is not finite at their solution (the final solve
  still is); the Scheimpflug tilt sweep subsamples views evenly, keeps their
  per-point covariances and respects the caller's fixed parameters, so it can
  settle in a different basin.
- **Parameter uncertainty, mixed noise.** When stated-noise residuals (pose
  priors, laser "consider" priors, stated pixel noise) are solved together
  with estimated-noise residuals, the estimated noise scale is
  `m_obs / (m_obs − tr(H⁻¹ H_obs))` — the degrees of freedom the estimated
  residuals actually have — instead of `m_obs / (m − n)`. Reported `σ` changes
  for hand-eye with robot-pose priors and for laser calibrations with stated
  priors; with estimated noise only, it is unchanged. A free parameter block
  no residual references is now reported as rank deficient (uncertainty
  unavailable) instead of receiving a covariance.
- **Laser-line residual.** The solver's line residual is the signed distance
  of the laser pixel to the projected plane–target intersection line (smooth
  through zero for the optimizer); a laser plane parallel to the target plane
  gives the distance to the line's vanishing line instead of no residual.
  Exported `residual_px` stays the absolute distance.
- **Rig hand-eye laser statistics.** `mean_laser_error_m` and
  `mean_laser_error_px` are means of absolute residuals, as documented
  (they were root mean squares), and `laser_count` counts the point-to-plane
  residuals only.
- **Validation.** `CalibrationSession::from_json` validates the stored
  configuration and input, so a session whose config is invalid under the
  current rules fails to load with the validation error. Configuration
  validation is stricter: a finite positive robust-loss scale, at least one
  camera with laser data in laser problems, finite robot poses, in-range
  `fix_poses` indices, per-view camera slot counts matching the rig, a
  positive chessboard `threshold_value`, and a Scheimpflug `max_abs_tilt`
  below π/2. A `DatasetSpec` with `robot_poses` or `pose_convention` on a
  topology without a robot, or with inconsistent target geometry, is
  rejected.
- **Minimal solvers and robust estimation.** EPnP evaluates the 1- to
  4-dimensional null-space candidates, refines each with Gauss–Newton and
  keeps the one with the lowest reprojection error; it returns
  `GeometryError::Singular` when no candidate puts every point in front of the
  camera. RANSAC scores every candidate a minimal solver returns, refits only
  a consensus that can beat the best, keeps a refit only when it keeps the
  whole consensus, only ever lowers its adaptive iteration bound, and ends
  with a least-squares refit on the winning consensus (repeated while the
  consensus grows), so the model it returns is the fit to the inliers it
  reports. The essential-matrix estimator rejects a sample
  with three collinear image points in either view. Equidistant
  initialization runs Nelder–Mead seeded and unseeded and keeps the lower
  cost. The Levenberg–Marquardt relative-step stop applies to the first
  attempt of an iteration only.
- **Error variants.** P3P, EPnP and DLT report mismatched point counts as
  `CountMismatch`; `dlt_ransac` without consensus reports `NoConsensus`.
- **Dense stereo.** Semi-global matching marks a pixel invalid when its
  matching cost is invalid at every disparity, and aggregates paths in a
  different order (sums can differ in the last bits; argmin ties can flip).
  `match_block` rejects inconsistent options (`sgm_p2 < sgm_p1`, a negative
  uniqueness ratio, a disparity range that overflows `i32`).
- `synthetic::planar::grid_points` and `grid_points_2d` return an empty grid
  for a zero dimension; `camera_matrix` decomposition normalizes `K` to
  `K₂₂ = 1`.
- The chessboard, ChArUco and PuzzleBoard detector is `calib-targets` 0.17.0.
  Two kinds of frame that used to deliver wrongly labelled corners now give
  no features: a ChArUco frame blurred past readability, whose board
  placement rested on marker reads no stronger than chance, and a
  chessboard frame whose small "grid" was assembled from saddle points that
  are not board corners. Other detections are unchanged. The detector
  version is part of the detection-cache key, so these caches are rebuilt
  on the next run. The Python examples' optional `calib-targets` package is
  0.17.0 or later.

### Documentation

- The book's sensor chapter states what an estimated Scheimpflug tilt is:
  `atan(tan τ_s / m_p)` for a sensor tilted by `τ_s` behind a lens of
  pupil magnification `m_p`, the mechanical tilt only when `m_p = 1`.
- The book has one bibliography: the References page lists every work the
  book cites, and each chapter links its citations to it.

### Fixed

- A rig laser run (`rig_laserline_device`) finds each of its views among
  the upstream rig hand-eye run's views by their images (the same file, or
  the same path within the dataset folder) or their pairing token, as the
  upstream export's image manifest records them. Before, it placed them by
  position when both runs kept the same number of views, so two runs that
  left out different views took the wrong board poses without an error. A
  view the upstream run left out, or one that matches two of its views, is
  now the error `RunError::UpstreamViewMatch` (Python `kind`
  `"upstream_view_match"`). An upstream export without an image manifest is
  matched as before.
- `mvg::decompose_homography` returns the plane normal (it returned a vector
  lying in the plane) and a proper rotation for reflected decompositions.
- Distortion fitting bounds the radius by the board points, not the pixel
  coordinates.
- `rectification::cloud` propagates covariance correctly when a point index
  repeats; bundle-adjustment parameter labels name the right blocks.
- Folder sniffing walks directory entries in sorted order and follows
  symlinks without cycles, so the manifest it writes does not depend on the
  filesystem's listing order.
- `vision_calibration::mvg::match_block` no longer panics on a 1-pixel-wide
  image with semi-global matching.
- Guarded zero-dimension planar grids (`nx == 0 || ny == 0`) in
  `synthetic::planar::{grid_points, grid_points_2d}` so empty boards return an
  empty vector cleanly.
- Fixed CLI verbose formatting of tagged `UncertaintyUnavailable` values and
  normalized empty manifest parent paths to `.` in both the CLI and desktop
  app backend.
- Fixed desktop backend lock contention by taking `ExportCache` snapshots
  before blocking compute and offloading I/O and CPU-heavy commands to
  blocking worker threads; added request-sequence guards and unmount cursor
  cleanup in the frontend.
- A solve that meets a residual whose Jacobian is not finite where it
  linearizes (a residual not differentiable there, such as `sqrt` at 0)
  fails with `Error::Numerical`, at the start, after a step or at the
  solution, on both solver backends. Before, both stopped at the first such
  point and reported it as converged: the tiny-solver backend set the
  derivative to 0 (`relative_step`, the parameter reported unobservable),
  the factrs backend found no step (`no_accepted_step`). The other solver
  errors name their cause: a cost that is
  not finite at the initial parameters, or normal equations that cannot be
  allocated.
- PuzzleBoard corners were reported 0.25 px too far right and down (+0.25
  px in x and in y). The detector finds corners on an image upscaled 2×,
  and their positions were divided by 2 to return to the image's pixels,
  which ignores that both images put integer coordinates at pixel
  centres; the corner detector's own upscale stage now maps them back
  exactly. A calibration from PuzzleBoard images moves its principal point
  `cx, cy` by −0.25 px; its reprojection error does not change. Corner
  detection is otherwise the same; decoding now reads the image as given
  rather than the upscaled copy, which in rare frames labels a few fewer
  corners at the image border. Cached PuzzleBoard detections are recomputed
  on the next run.

## [1.0.0-rc.1] - 2026-11-18

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

- **The laserline device input is a dataset object** (breaking).
  `laserline_device::LaserlineDeviceInput` is `LaserlineDataset`, no longer
  a `Vec<LaserlineView>`: build it with `LaserlineDataset::new(views)?`,
  which checks each view, and read the views from its `views` field. Its JSON
  is `{"views": [...]}`, like the other inputs, instead of a bare array
  (Python's `LaserlineDeviceInput.to_payload()` writes the new form). A
  `laserline_device` session saved before this release is not readable: its
  `schema_version` is bumped (3), so it is refused with a version error.

  ```rust
  // before
  session.set_input(views)?;
  // after
  session.set_input(LaserlineDataset::new(views)?)?;
  ```

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

- **Configs, manifests, observation containers and results can grow
  without a breaking release** (breaking). The types below are
  `#[non_exhaustive]`: outside this library they are built with `new(...)`
  (the required fields; optional ones start at their defaults and are set
  on the value) or `Default` (set the fields after), never with a struct
  literal or `..Default::default()`. A pattern that names their fields ends
  in `..`, and a `match` on `NoiseLevel`, `Detection` or `Captures` needs a
  wildcard arm. Their fields stay public and their JSON is unchanged.

  ```rust
  // before
  let config = ChessboardConfig { rows: 17, cols: 28, square_size_m: 0.02, chess_corners: None };
  let view = RigView { obs: RigViewObs { cameras: vec![left, right] }, meta: NoMeta };
  let opts = IterativeIntrinsicsOptions { iterations: 2, ..Default::default() };
  // after
  let config = ChessboardConfig::new(17, 28, 0.02);
  let view = RigView::new(RigViewObs::new(vec![left, right]), NoMeta);
  let mut opts = IterativeIntrinsicsOptions::default();
  opts.iterations = 2;
  ```

  - `core`: `CameraModelSpec::new`, `CameraFix` (`Default`),
    `ParameterCovariance::new`, `ViewCovariance::new`, `ResidualNoise::new`,
    `SecondOrderError::new`, `CombinationTerm::new`, `NoiseLevel`,
    `View::new`, `RigViewObs::new`, `RigView::new`, `RigDataset::new`,
    `PlanarDataset::new`.
  - `detect`: `ChessboardConfig::new`, `CharucoConfig::new`,
    `PuzzleboardConfig::new`, `RinggridConfig::new`, `ChessCornersConfig`
    (`Default`), `CornerCovariance::new`, `Feature::new` (with
    `with_cov_px`), `CacheKey` (`for_detection`, `from_inputs`),
    `CachedFeatures::new`.
  - `dataset`: `DatasetSpec::new` (the current format version),
    `CameraSource::new`, `LaserExtractionSpec`, `DetectorSpec` and
    `ChessCornersDetectorSpec` (`Default`), `CornerCovarianceSpec::new`,
    `RobotPoseSource::new`, `PoseColumnMap::new`, `PoseConvention::new`, and
    the `TargetSpec` variants, built with `TargetSpec::chessboard`,
    `charuco`, `puzzleboard` and `ringgrid`.
  - `device_seed`: `DeviceSpec::new`, `CameraDeviceSpec::new`,
    `ScheimpflugMountSpec::new`, `RigLayoutSpec::new`,
    `CameraMountSpec::new`, `NominalPoseSpec::new`, `LaserPlaneSpec::new`.
  - `common`: `RobotPoseMeta::new`, `RigLaserlineView::new`.
  - `single_cam_handeye`: `SingleCamHandeyeInput::new`.
  - `laserline_device`: `LaserlineMeta::new`.
  - `rig_laserline_device`: `RigLaserlineDeviceInput::new`,
    `RigUpstreamCalibration::new` (no covariance; set `uncertainty` to add
    it), `RigLaserlineDataset::new`.
  - `rig_handeye_laserline`: `RigHandeyeLaserlineInput::new`,
    `RigHandeyeLaserlineView::new`.
  - `session`: `InvalidationPolicy` (its constants and `Default`).
  - `dataset_runner`: `Detection` and the result `PairedViews`.
  - `init`: `DistortionFitOptions` and `IterativeIntrinsicsOptions`
    (`Default`), and the results `LinearPlaneEstimate`,
    `EquidistantIntrinsicsLinearInit` and `ExtrinsicPoses`.
  - `mvg`: `laser::LaserObservation::new`, `pose::PoseObservation::new`,
    `triangulation::RigObservation::new`, `types::PixelMatch::new`,
    `types::Captures`, and the results `laser::LaserPoint` and
    `solvers::CameraMatrixDecomposition`.

- **Every type is reached by one path** (breaking).
  - The seeds and results of the stages both rig workflows run, and the
    hand-eye refinement result both hand-eye workflows share, are in
    `common` only. `rig_extrinsics::RigManualInit`,
    `rig_handeye::RigManualInit` → `common::RigManualInit`;
    `rig_extrinsics::RigPoses`, `rig_handeye::RigPoses` →
    `common::RigPoses`; `rig_extrinsics::RigInitResult`,
    `rig_handeye::RigInitResult` → `common::RigInitResult`;
    `single_cam_handeye::HandeyeOptimizeResult`,
    `rig_handeye::HandeyeOptimizeResult` → `common::HandeyeOptimizeResult`.
    The per-camera intrinsics result of a rig is renamed so it does not
    share a name with the one-camera result:
    `rig_extrinsics::IntrinsicsInitResult`,
    `rig_handeye::IntrinsicsInitResult` → `common::RigIntrinsicsInitResult`
    (`planar_intrinsics::IntrinsicsInitResult` and
    `single_cam_handeye::IntrinsicsInitResult` are unchanged). Python
    follows: the classes are in `vision_calibration.common` and the package
    root, and `rig_extrinsics.IntrinsicsInitResult` /
    `rig_handeye.IntrinsicsInitResult` are `RigIntrinsicsInitResult`.

    ```rust
    // before
    use vision_calibration::rig_extrinsics::{RigManualInit, RigPoses};
    // after
    use vision_calibration::common::{RigManualInit, RigPoses};
    ```

  - `dataset::samples` lists its public items (`Sample`, `FetchError`,
    `CACHE_DIR_ENV`, `catalogue`, `sample`, `sample_dir`, and with the
    `fetch` feature `fetch` and `default_cache_dir`); the undocumented
    download helpers it also exposed are gone from the facade.
  - `init::pnp::PoseEstimate::pose` is `cam_se3_world`, the name of the
    pose it holds.

    ```rust
    // before
    let pose = dlt_ransac(&world, &image, &k, &opts)?.pose;
    // after
    let pose = dlt_ransac(&world, &image, &k, &opts)?.cam_se3_world;
    ```

  - `core::FixedParams`, the fixed-parameter indices `CameraFix::resolve`
    returns, is documented and `#[non_exhaustive]`. The model slots of
    `core::CameraModel` (`core::AnyProjection`, `core::AnyDistortion`,
    `core::AnySensor`) are documented and exported from `core`; they are
    `#[non_exhaustive]`, so a `match` on them needs a wildcard arm.
  - `dataset::DATASET_SPEC_VERSION`, the manifest format version, is
    exported beside `DatasetSpec`.
  - In the lower crates, items the facade does not offer are no longer
    public API: `vision_calibration_linear::planar_pose::PlanarPoseSolver`
    is removed (use `estimate_planar_pose_from_h`), as is the unused
    `vision_calibration_linear::extrinsics::MultiCamExtrinsicsInit`; the
    multi-level reprojection reports of
    `vision_calibration_pipeline::analysis`, `vision_calibration_core`'s
    `TargetPose` and `compute_mean_reproj_error`, and
    `vision_calibration_optim`'s `LaserResidualRecords` and
    `compute_*laserline_feature_residuals` are internal.

### Changed

- The default solver backend runs on `tiny-solver` 0.19, which builds each
  Jacobian with dual numbers on the stack and assembles it without locking.
  Non-linear solves are 1.5–2.9× faster per workflow with all cores (2.1× at
  the median) and 3.0–3.7× faster on one thread. Results are unchanged: bit
  for bit, except some laser workflows with a fisheye camera, which differ
  at the level of floating-point rounding (final cost within 1e-13
  relative).
- The Python package is built on PyO3 0.30 (with `numpy` and `pythonize`
  0.30). It still supports Python 3.10 and later, and its API is unchanged.
- Config validation errors name the full path of the field they reject:
  `solver.max_iters`, `intrinsics_init.iterations` and
  `handeye_init.min_motion_angle_deg`, the same in every problem type. In
  Python, a feature or laser-extractor array with the wrong number of
  columns is reported as `<name> must have shape (N, cols)`, like every
  other array argument, and a feature array with the wrong number of rows
  names that array. The checks themselves are unchanged.
- The chessboard, ChArUco and PuzzleBoard detector is `calib-targets` 0.16.3.
  A grid cell next to a missing corner no longer discards the connected
  patch of the board around it, so partly occluded boards keep more corners.
  The detector version is part of the detection-cache key, so these caches
  are rebuilt on the next run.
- The ring-grid detector is `ringgrid` 0.16.0. Its version is part of the
  detection-cache key, so ring-grid caches are rebuilt on the next run. What
  ring-grid targets now give:
  - Features only from a view whose place on the board is corroborated (by
    agreeing decoded codes, or by the origin dots); a view with a few
    isolated markers or unreadable codes yields none, instead of possibly
    wrong board positions. A coded marker takes the position of the grid
    cell it sits in, so a misread code no longer mislabels it, and more
    markers are labelled on tilted and partial views.
  - Only markers with a perspective-corrected centre, and with a confidence
    of at least 0.03 on the detector's new centre-accuracy score; markers
    whose centre is less certain are left out, so the feature count per view
    can drop, mostly for small, strongly tilted rings.
  - With `center_method = "photometric"`, far more plain markers keep a
    photometric centre, so plain boards return more features.
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
- The ring-grid detector is `ringgrid` 0.15.0. The detector version is part
  of the detection-cache key, so ring-grid caches are rebuilt on the next
  run.

### Added

- The optimization problem builder takes laser planes and the laser
  residual: `Problem::add_plane` (a laser plane, labelled `<name>.nx` …
  `<name>.d`), `fix_plane`, `Solution::plane`, and `Problem::add_laser`, the
  point-to-plane (metres) or line-distance (pixels) laser residual of a
  pixel through one pose or a rig chain (`cam_se3_rig · rig_se3_target`),
  with `LaserNoise::Estimated`. With them a rig without robot poses solves
  its shared per-capture board poses, its laser planes and, as chosen, its
  cameras and extrinsics in one problem, on both solver backends, with the
  covariance. The book's "A Rig Without a Robot" shows the recipe.
- Solve reports say why the solver stopped. `SolveReport` gains an optional
  `termination`, a `SolveTermination`: the stopping rule that ended the
  solve (`relative_step`, `min_error`, `min_abs_decrease`,
  `min_rel_decrease`), `no_accepted_step`, `max_iters` when the iteration
  budget ran out, or `all_parameters_fixed`; `SolveTermination::is_converged`
  is false only for `max_iters`. It reaches every estimate, `problem::Solution`
  and the bundle adjustment through their reports, and the step results that
  run a solve as `termination` (`per_cam_terminations` for the per-camera
  intrinsics stage of the rig workflows and the rig laser planes, whose
  `RigLaserlineEstimate` gains `per_cam_reports`); Python's step results and
  `BundleAdjustment` carry it as a string. A report saved without it reads
  back as `None`, unknown, and is written as before.

- Ring-grid targets on a rectangular lattice and plain ring-grid targets.
  The manifest's `ringgrid` target and `RinggridConfig` take an optional
  `lattice` (`"hex"`, the default, or `"rect"`) and an optional `coding`
  (`"coded"`, the default, or `{ plain = { origin_dots = true } }` for plain
  rings anchored by a triad of origin dots), in the manifest as
  `RinggridLatticeSpec` and `RinggridCodingSpec`, in the detector as
  `RinggridLattice` and `RinggridCoding`. `TargetSpec::ringgrid_rect` and
  `TargetSpec::ringgrid_plain` build them. A plain view is labelled in the
  board frame whenever it shows the origin dots, however small a part of a
  large board it covers, and yields no features otherwise; a plain board
  without origin dots is rejected before detection. Plain boards may leave out `marker_ring_width_m`, which
  they ignore. Photometric centres work on plain markers. Existing manifests,
  configs and their detection-cache entries are unchanged.
- A robust laser-plane fit. `LaserlinePlaneSolver::from_views_ransac` and
  `from_points_3d_ransac` fit the plane by seeded RANSAC over the same
  closed-form fit, with a metric inlier threshold, an iteration cap, a
  confidence, a minimum inlier count and fraction, an optional refit on the
  inliers and a seed; the estimate lists the inlier indices. A fitted plane
  can seed a laser workflow through its manual initialization.
- `RansacOptions::min_inlier_fraction` (with `with_min_inlier_fraction`, and
  `min_inlier_fraction` on the Python `RansacOptions`): the share of all the
  data, in `[0, 1]`, a model's inliers must reach to be accepted, alongside
  `min_inliers`. Every RANSAC-based estimator honours it; the default `0`
  keeps results unchanged.
- **An opt-in correction of the estimated robot error for corner noise
  correlated within views.** With `estimate_sigma`, each robot-pose
  correction carries the pose-like part of its view's corner noise. When the
  residuals are correlated within views (a significant view scatter), that
  part is larger than the independent-noise model says, and the estimate
  counts the excess as the robot's. The new `discount_view_scatter` option
  of the `refine` robot prior (off by default; it requires `estimate_sigma`)
  takes the excess out at the corner noise's own view scatter, as far as the
  corrections resolve it:
  - Where the remaining robot share exceeds two standard errors over the
    views, it is the estimate. On synthetic data with tile-correlated
    corner noise the estimate went from 2.4 and 6.1 times the simulated
    robot error (rotation, translation) to 0.95 and 1.19 times it, and the
    hand-eye translation error from 0.70 to 0.19 mm.
  - Where it does not, the estimate stays at the largest robot error the
    corrections cannot tell from none, never above the plain estimate.
  - Where the subtraction would claim more corner noise than the
    corrections hold (noise that scatters the distortion far more than the
    poses), nothing is taken out.

  The new `robot_refinement.unresolved` (`"rotation"`, `"translation"` or
  `"both"`, absent otherwise) names the parts of `robot_refinement.sigma`
  that are such upper bounds. The option is off by default because the
  scale it takes out is measured on the shared parameters, and how well it
  holds for the poses has been measured only on synthetic single-camera
  scenes; with it off, results are unchanged.

- **The regression suite gates a measurement.** The `opencv_stereo` entry
  also measures its held-out boards: its even views calibrate a stereo rig,
  the corners of each odd view are triangulated with the calibration's
  uncertainty, and the 2070 distances along board rows and columns, 1 to 7
  squares, are compared with the board's size. The reference result stores
  the mean and RMS error, the mean stated σ and the RMS of error over σ per
  held-out view and length, and the suite's check fails on any change (or,
  with a tolerance of `k` σ, on a mean length that moves by `k` of its σ).
  A suite entry asks for it with `measure_board = { square_size_m = ... }`.
  `opencv_fisheye_stereo` measures its held-out boards the same way, through
  the Kannala-Brandt model: 4896 lengths over 17 held-out views.

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
- Ring-grid targets take an optional `center_method` (`RinggridCenterMethod`
  in `RinggridConfig`, `RinggridCenterSpec` in the manifest's `ringgrid`
  target): `"ellipse"` (the default, unchanged) or `"photometric"`, a
  per-marker fit of the marker pattern under a direction-dependent blur, for
  blurred or low-contrast images. It costs about 2.5 times the detection
  time and keeps only the markers the fit refined. Existing manifests are
  unaffected.
- The `schemars` feature also derives `JsonSchema` on the detector configs
  (`detect::ChessboardConfig`, `CharucoConfig`, `PuzzleboardConfig`,
  `RinggridConfig`, `ChessCornersConfig` and their parts), like every other
  config type.

### Fixed

- A solve with every parameter fixed returns `Error::Cancelled` when its
  token is cancelled, like any other cancelled solve; it returned a
  solution on the tiny-solver backend. On the factrs backend such a solve
  now runs no iteration (`num_iters` 0), as on tiny-solver, instead of one
  that could not move anything; the solution is unchanged.
- The book, the READMEs and the API-reference link now match the code: the
  manifold retractions (a pose step is `T·exp(ξ)`, the unit-sphere step is
  its exponential map), the six residual families and their parameter-block
  names, the solver's stopping rules, the per-stage robust-loss defaults and
  the crate graph. The published API reference opens at
  `api/vision_calibration/`. The Python example for selecting the factrs
  backend keeps the workflow's iteration budget.
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

## Older releases

Earlier releases are archived verbatim in the [repository](https://github.com/VitalyVorobyev/calibration-rs/tree/main/docs/changelog), one file per version:

- [`0.17.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.17.0.md) - 2026-11-06
- [`0.16.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.16.0.md) - 2026-11-04
- [`0.15.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.15.0.md) - 2026-11-03
- [`0.14.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.14.0.md) - 2026-11-03
- [`0.13.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.13.0.md) - 2026-11-03
- [`0.12.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.12.0.md) - 2026-11-02
- [`0.11.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.11.0.md) - 2026-11-01
- [`0.10.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.10.0.md) - 2026-10-31
- [`0.9.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.9.0.md) - 2026-09-29
- [`0.8.3`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.8.3.md) - 2026-09-28
- [`0.8.2`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.8.2.md) - 2026-09-27
- [`0.8.1`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.8.1.md) - 2026-09-07
- [`0.8.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.8.0.md) - 2026-08-16
- [`0.7.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.7.0.md) - 2026-07-10
- [`0.6.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.6.0.md) - 2026-06-17
- [`0.5.1`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.5.1.md) - 2026-05-23
- [`0.5.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.5.0.md) - 2026-05-21
- [`0.4.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.4.0.md) - 2026-04-29
- [`0.3.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.3.0.md) - 2026-04-12
- [`0.2.0`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.2.0.md) - 2026-03-07
- [`0.1.2`](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/changelog/0.1.2.md)
