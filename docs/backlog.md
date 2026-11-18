# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §11); the CHANGELOG and git history record what landed. Each
heading is a track of the [ROADMAP](ROADMAP.md), which gives its goal, gate
and phase; the last heading is the release. A parked entry says what reopens
it. Dependencies are named as `Needs: <ID>`.

## Parameter uncertainty (Phase 2)

Design: ADR 0026; evidence: `notes/uncertainty.md`.

- [~] UQ-RINGGRID-COV - The ring-grid detector reports the ellipse-centre
  covariance (cross-repo, ringgrid). Parked: the real-data study advises
  against weighting a calibration by the chessboard and ChArUco
  covariance, so a ring-grid one would only be a report. Reopen if a
  ring-grid weighting study is wanted, or if the view scatter on ring-grid
  data needs the detector's term.
- [ ] UQ-SECOND-ORDER-COMPOSED - A rig laserline device covariance that
  joins its planes to the upstream calibration's
  (`laserline_rig_bundle.rs`, `with_planes`, and
  `handeye_upstream_uncertainty`) has `second_order: None`, so its quality
  report says `covariance_linearity` is not applicable even when the
  upstream warned. The cameras' block is the upstream's marginal, so the
  upstream's `max_scatter_ratio` is a lower bound of the joint's: carry it
  (with the combination's labels mapped through the hand-eye to per-view
  poses) and the planes' own, marked as a bound, or compute the joint
  term from the upstream's residuals where the dataset runner has them.
  Gate: a Monte-Carlo NEES on a rig laser scene.
- [ ] UQ-ROBUST-KAPPA - Under a redescending loss (Cauchy, Arctan) with
  heavy outliers, a one-dimensional residual family can have a mean `ψ′` at
  or below zero (`(1 − u)/(1 + u)²` per Cauchy residual at `u = s/c² > 1`).
  `family_moments` (`backend/covariance.rs`) clamps it to
  `f64::MIN_POSITIVE`, so the sandwich covariance comes back scaled by about
  `1/κ²` instead of unavailable. Return a typed `UncertaintyUnavailable`
  reason there. Gate: a laser scene with a Cauchy loss and 30 % outliers
  reports the reason; no result with `κ` well above zero moves.
- [ ] UQ-ROBOT-SIGMA-SCATTER-DEFAULT - Decide whether the robot prior's
  `discount_view_scatter` (`robot_poses.rs`, `estimated_priors`) becomes the
  default. Today it is measured only on the single-camera eye-in-hand
  synthetic scene of `view_correlated_noise_study`
  (`optim/tests/robot_pose_priors.rs`), at view scatters of 3–9 (tiles) and
  40–80 (radial); on `kuka_1` (view scatter about 4) it is barely resolved
  (c = 46.8 against 2·se = 42.3) and moves `base_se3_target.tz` by 3.6σ with
  no ground truth. Turn it on only after (1) a sweep at `kuka_1`-like view
  scatter (about 4), (2) the same sweep over rig, eye-to-hand and rig laser
  hand-eye scenes, each with (3) an oracle column (the hand-eye solved with
  the true robot error as fixed priors), shows the hand-eye no further from
  the truth than the plain estimate's in any cell and the estimate never
  below the robot's error. Moving the default changes `kuka_1`'s reference.
- [ ] UQ-CROSS-FRAME - Projection uncertainty across the reference-frame
  shift (the mrcal method). Needs the intrinsics–pose cross-covariance,
  which results do not carry. Done when: results carry that
  cross-covariance and the projection σ across the frame shift agrees with
  Monte-Carlo.

## Datasets and the regression suite (alongside)

A versioned public suite for regression and for comparison with other
libraries. Image-free observations live in the repository and gate every PR.
Real-image entries are fetched from their publishers, not archived by us:
`kuka_1` (RWHE-Calib), `stereo` (Kaggle) and the OpenCV fisheye set
(`opencv_extra`) state no licence; DS8 is U.S. public domain. Synthetic
entries need no images. Comparison tools must run on Linux, macOS and
Windows. `stereo_charuco` stays local-only, as do the private sets.

- [ ] DATA-VIEW-IDENTITY - A rig laser run matches its upstream rig
  hand-eye export to its views by position or by a numeric pairing token
  (`dataset_runner/laser.rs`, `rig_se3_target_from_upstream`,
  `robot_delta_for_view`). When both runs kept the same number of views but
  dropped different ones, the positional match pairs the wrong target
  poses with no error. The cause: `calibrate_dataset` never fills the
  export's `diagnostics.image_manifest` (the app builds its own), so an
  export from the CLI or Python carries no view identity. Fill the
  manifest in `calibrate_dataset` (one source with the app), match the
  upstream by its frames' image paths or tokens, and make an ambiguous
  match a typed error. Gate: a rig laser run whose upstream dropped a
  different view than it did fails; the matching cases give bit-identical
  results.
- [~] DATA-LICENCE-KUKA - The `kuka_1` images (RWHE-Calib, published
  without a licence) are committed through LFS. Parked on the maintainer
  asking the RWHE-Calib authors for a licence. Reopens when they answer: on
  one that does not allow redistribution, drop the images from the
  repository and LFS (they are fetchable with `suite fetch --only kuka_1`;
  the `planar_real`/`handeye_session` examples and the LFS-image unit tests
  then name the fetch command).
- [ ] DATA-MORE - More public entries, each after a licence check: the other
  RWHE-Calib sets (`kuka_2`, `kuka_3`) and Tabb's DS1–DS7 (U.S. public
  domain, same record as DS8). Done when: every listed set is a suite entry
  fetched from its publisher with checksums, or is named here as failing
  the licence check.

## Bindings and internals (alongside)

- [ ] PY-MODELS-GEN - `models.py` and the problem modules hand-write the
  config, seed and export dataclasses (87 classes with `to_payload` /
  `from_payload`) beside the TypedDicts `cargo xtask emit-schemas` generates
  from the same Rust types (`_generated_types.py`); 68 names exist twice.
  `tests/test_config_defaults.py` pins the defaults to
  `schemas/*.default.json`, but fields and converters are still kept in
  step by hand. Generate the dataclasses and their converters in
  `xtask/src/py_types.rs`, keeping the class names and the Python-only
  conveniences (`Pose`, the distortion classes) as a thin hand-written
  layer. Gate: no hand-written class for a type the schemas describe; the
  Python suite and the binding parity check pass unchanged.
- [ ] PY-MVG-SCHEMA - The MVG results have no schema:
  `vision-calibration-py/src/mvg.rs` builds JSON by hand (`json!`) and
  `mvg.py` parses it back into hand-written dataclasses. Derive `Serialize`
  and `JsonSchema` on the `vision-mvg` result types (or binding-side wire
  structs), publish their schemas and generate the Python types. Needs:
  PY-MODELS-GEN. Done when: `mvg.rs` builds no JSON by hand, the MVG result
  schemas are under `cargo xtask emit-schemas --check`, and the Python MVG
  types are generated from them.
- [ ] OPTIM-FIXED-ON-BUILDER - The fixed problems of `optim::problems`
  build their IR by hand, beside the public `optim::problem` builder that
  produces the same IR for reprojection (`add_reprojection`, `lower`); only
  `vision-mvg` uses the builder. Give the builder the hand-eye chain and
  the laser residuals, then build the fixed problems through it. Results
  bit-identical on both backends.
- [ ] MATH-ONE-SO3-LOG - Two SO(3) logarithms (`linear/src/handeye.rs`
  `log_so3`: nalgebra angle-axis, zero below 1e-12;
  `optim/src/problem/calibration.rs` `so3_log`: sign-fixed `atan2` with a
  small-angle series) and two pose-from-homography functions
  (`linear/src/planar_pose.rs`, guarded against singular columns and signed
  on `t.z`; `geometry/src/pnp/planar.rs`, unguarded and signed on `h₃.z`)
  differ in arithmetic. Keep one of each in the lowest crate (the guarded
  ones), and measure what moves: the hand-eye initializers and planar PnP.
  Done when: one of each remains and the change in the hand-eye
  initializers and planar PnP is measured on both backends.
- [ ] CI-PRIVATE-EXAMPLES - `vision-calibration-examples-private` is built
  by no CI job, because it path-depends on the sibling `vision-metrology`
  checkout; an example there stopped compiling unnoticed. Check it in CI
  with that repository checked out beside this one, or move the laser
  extraction it needs behind a feature so the rest checks alone. Done when:
  a CI job on every PR builds the crate (all of it, or all but the
  feature-gated laser extraction).

## Camera models (Phase 2)

Each model follows AGENTS.md §10 (a camera model).

- [ ] CAM-EUCM-DS - Enhanced unified (EUCM) and double-sphere projections,
  with their closed-form unprojection and validity regions. Done when: both
  are `ProjectionKind` variants with everything §10 lists, the synthetic
  matrix test passing in every problem type.
- [ ] CAM-TELECENTRIC - Telecentric (affine/orthographic) projection for
  metrology lenses: affine-factorization init, and the gauge of ADR 0005
  (held axial depth, the per-view reflection ambiguity). Done when: it is a
  `ProjectionKind` variant with everything §10 lists, and the matrix test
  covers the held depth and the reflection ambiguity.
- [ ] CAM-SELECT - Model selection: compare candidate models by residual
  structure, an information criterion and held-out views. It extends the
  distortion-model-selection tutorial. Done when: the library ranks
  candidate camera models of one dataset by those three measures, and the
  tutorial uses it.
- [~] CAM-RESIDUAL-FIELD - An optional camera-fixed displacement field on
  top of any camera model, designed in ADR 0030. Parked: on the captures
  that motivated it most of the field is coupled to defocus and depth, so a
  slot fitted on one capture would bake a capture-dependent average into
  the camera (ADR 0030, Consequences; `notes/parametric-residual-term.md`).
  A held-fixed implementation exists under the tag `archive/cam-field-core`,
  unmerged. Reopens with CAM-FIELD-ORIGIN: a field that stays put with the
  aperture and through focus.
- [ ] CAM-FIELD-ORIGIN - Decide whether the defocus-coupled part of the
  Scheimpflug residual is optical or the detector's. What decides it, with
  and without a new capture, is `notes/parametric-residual-term.md`, *What
  is undetermined*; views through focus at one image position do not
  (`notes/tilt-from-focus.md`). The harness is the bench example's `--parts
  field` and `--parts centroid`. The same aperture capture also fixes the
  coupling's image-fixed part, which decides how far it biases the
  distortion-derived tilt (`notes/tilt-bias.md`: 13–51 % of the
  focus–distortion gap, by the assumed law).

## Calibration problems (Phase 2)

- [ ] CAL-TARGET-OFFSETS - A non-ideal target: per-point in-plane offsets of
  the board geometry, opt-in. Evidence (`notes/display-target-residual.md`):
  on a displayed PuzzleBoard, offsets of 34 µm RMS shared by six cameras
  take the held-out residual from 0.315 to 0.269 px; a board shape and a
  cover layer gain nothing and are left out. Design: a target geometry
  value in core (nominal points plus an offset per point id, board frame,
  metres) that every planar workflow reads instead of the bare points,
  so the offsets reach all problem types without solver changes; two
  sources, (a) a table measured once per target and display setting,
  applied as fixed, its σ entering the covariance as consider parameters
  (flatness itself is not worth a term: a shared shape of 0.2 mm
  peak-to-valley changes held-out views by 0.002 px), and
  (b) an estimate: free target points in a joint multi-view, multi-camera
  solve, gauge a planar similarity (hold two points, or four linear
  constraints), a point free only with enough observations, exported with
  its covariance block in the shape (a) reads. Not on by default: it needs
  orientation diversity (several cameras or a board turned in the image),
  else offsets trade with distortion. The separable form (`dx` per column,
  `dy` per row) recovers two thirds of the gain with about 60 parameters
  instead of 1250 and is a possible cheaper mode. Gate: synthetic ground truth
  with known offsets recovered on both backends, intrinsics σ against
  Monte-Carlo with the offsets free, and the held-out gain on
  `scheimpflug_rig_ref_cam0..5`.
- [ ] CAL-TILT-DETERMINE - A Scheimpflug tilt is estimated only when the
  data determine it, the result says why, and a small σ alone does not
  count as determined. Three parts, one design:
  - **The procedure** (ADR 0031, proposed; evidence
    `notes/tilt-determination.md`): `TiltPolicy::{Estimate, Determine {
    max_sigma }}` on `CameraConfig` (Scheimpflug only), the procedure in
    `planar_family::optimize_camera`, the `tilt_determination` record
    (source, frame, statistic) in the export with a schema bump, the
    `TiltDetermination` quality check; opt-in (`Estimate` stays the
    default) until the real-data gate. Needs CAM-PARAM-PRIOR for its prior
    branch. Gate: the study's tables reproduced through the library on both
    backends; on the private Scheimpflug rigs the branch per camera,
    held-out error within 5 % of `Estimate`, the decentring seed reaching
    the seeded route's minimum; `calib-bench accept` unchanged under
    `Estimate`.
  - **The systematics safeguard** (evidence `notes/tilt-bias.md`: a smooth
    unmodelled displacement of 0.10 px biases the tilt by 3.5–15 noise σ,
    more views raising the bias in σ). Design: (1) a sensitivity term, the
    tilt's shift when the camera is refitted with a nuisance displacement
    law (depth `z̄/z − 1` × image monomials of degree 1–4 for any detector;
    unsigned blur `|g|` × degree 0–3 when the detector reports a blur), as
    `σ_sys`, and "determined" read from `√(σ² + σ_sys²)`; a signed-defocus
    law cannot be the nuisance (it leaves the tilt unidentified); (2) the
    nuisance law's held-out gain as a residual-systematics statistic; (3)
    with blur, the focus-direction check of CAL-TILT-FOCUS (no metric
    datum); (4) the export carries the tilt's σ, `σ_sys` with its law, the
    statistic, the view-jackknife scale and a determination status with
    "determined, systematics dominate" beside "determined" and "not
    determined". Gates: the synthetic map of the bench example (`--parts
    tiltsweep`) on both backends, null (no displacement: `σ_sys` within its
    noise), a known displacement (`σ_sys` covers the bias), and the six
    ring-grid cameras.
  - **The quality message** (evidence `notes/tilt-identifiability.md`: the
    tilt is determined only through lens distortion, and
    `ParameterCorrelation` already flags the bad cells pairwise with a
    generic message). Design: in `analysis::quality`, when a flagged pair
    joins a Scheimpflug tilt with a principal-point coordinate, a
    tangential or a thin-prism term, name the cause (weak or unobserved
    distortion: cover the image corners, or hold the tilt at a known mount
    angle; tangential or prism terms free with the tilt: hold them) and add
    the tilt's σ inflation by its correlations, `1/√(1 − R²)` from the
    camera block, as the measured value; additive, no threshold change.
    Gate: the weak-distortion, narrow-field and `p`-free cells of the study
    get the named message, the baseline cell none, on both backends.
- [ ] CAM-PARAM-PRIOR - Gaussian priors on camera parameters, designed in
  ADR 0031 (*Priors on camera parameters*); the tilt is the first user. The
  solve must reach the posterior, which the raw coordinates do not
  (`notes/tilt-determination.md`, C0): run it in the gauge-aligned
  coordinates or with SOLVER-VALLEY-STOP. Gate: posterior σ against the
  analytic `√(1/(σ⁻² + σₚ⁻²))` within 5 %, the C1 table through the
  library, JSON round-trips, typed errors for a prior on a fixed parameter
  or a non-positive σ.
- [ ] CAL-TILT-FOCUS - Per-feature blur as an observation, and a focus
  term that ties the Scheimpflug tilt to the plane of sharp focus
  (`notes/tilt-from-focus.md`). Evidence: on six ring-grid cameras a
  5-parameter focus model (plane `π`, blur scale `κ`, in-focus blur `σ0`)
  explains 91–96 % of the blur variance and the PoF's sign agrees with the
  distortion tilt on all six, but the two routes disagree by ±0.5° per unit,
  which neither model explains (a defocus/depth law in the distortion route
  closes 13–51 % of the `τx` gap and none of the direction gap,
  `notes/tilt-bias.md`). Design: (1) a
  detector-agnostic optional blur σ (px) per feature next to the 2D point
  (ring-grid photometric fits have it; corners could report an edge
  width), carried through the dataset types and the detection cache;
  (2) a diagnostic first: fit `σ² = σ0² + κ²(1/z − π·(x, y, 1))²` after
  the solve in the `K'` frame (the poses' tilt rotation applied) and
  report `π` with its view-jackknife σ, the tilt direction, the tilt
  implied by a stated pitch and pupil magnification, and the difference
  to the fitted tilt — the independent check; (3) only once the ±0.5°
  is explained, an opt-in focus residual (`FactorKind` arm, both
  backends, the covariance family map) linking `π` to the tilt through
  `v0 = f'·pitch` and `m_p` given as priors with σ. Not a replacement for
  the distortion route: it needs `m_p` or a reference unit. Gates: the
  exact thick-lens synthetic of the bench example on both backends,
  Monte-Carlo σ with view-correlated blur noise, null tilt, field
  curvature detected; the six-camera comparison. Reopen the factor step
  with the datasheet values (pitch, focal length, f-number, pupil
  magnification) or a capture at two focus settings.
- [ ] CAL-TILT-SEED-INIT - A tilt seed without an intrinsics seed fits the
  intrinsics at the seeded tilt (`planar_family::bootstrap_camera`, every
  workflow), but only from the candidates the linear Scheimpflug init builds
  around that tilt: the seed with the untilted Zhang `K` as reference, and
  the homography-decomposition fits started there, whose tilt drifts before
  the seed replaces it. When those fit badly the start is degenerate (one
  Monte-Carlo draw in 20 at `k1 = -0.30` and 0.087 rad tilt; one of the 12
  private Scheimpflug cameras ends on the tilt bound) where the full sweep
  found the basin. Evaluate every candidate reference `K` of the
  full sweep (grid, decomposition, refinement) at the seeded tilt and keep
  the best, a fixed-tilt mode of `estimate_scheimpflug_intrinsics_iterative`.
  Gate: those draws and that camera reach the full-seed result, no other
  result moves; measured on both backends.
- [ ] CAL-TILT-BOUNDS - The seeded Scheimpflug route's basin is set by its
  joint-solve bounds, not by the problem (`notes/tilt-identifiability.md`,
  table B: from bad seeds the bounded route reaches the minimum in 4–5 of
  16 starts per cell, the unbounded joint solve from the same linear init
  in 16 of 16). The bounds (`fx` within [0.75, 1.5]× the seed, principal
  point ±0.15·f, tilt ±0.35 rad) were set as a runaway guard.
  Design: solve the joint stage unbounded and fall back to the bounded
  solve only on a runaway (focal length outside [0.3, 3]× the seed, tilt
  beyond 45°, principal point outside the image), or widen the window to
  that guard; opt-in until measured. Gate: table B's bad-seed rows on both
  backends, the real-data `calib-bench basin` (its ×0.5 and ×2 focal
  cells, which fail today) and `calib-bench accept` unchanged; the
  strong-distortion `k1` trap (`scheimpflug-intrinsics.md`) still avoided.
- [ ] CAL-MULTI-TARGET - Several targets with unknown relative poses in one
  scene, estimated jointly. Done when: a synthetic scene with two or more
  targets recovers their relative poses and the cameras on both backends.
- [ ] REPORT-SECTIONS - The HTML report (`analysis::html_report`, shared by
  `vcal` and the app) renders the quality checks and one section per
  camera. A hand-eye export shows no hand-eye (the pose, its σ, the robot
  refinement), and a laser export shows no planes. Add those sections
  (additive). Gate: `vision-calibration-bench/tests/html_report.rs` checks
  them for every problem type.

## Multiple-view geometry (Phase 2)

Design: ADR 0028; what each σ promises: `notes/mvg-measurements.md`.

- [~] MVG-METROLOGY-XREPO - `../vision-metrology` adopts `mvg::metrology`
  (a cross-repo change). Parked by the maintainer (2026-11-06): it would give
  that image-processing library, which has its own f32 camera model and a
  pixel → ray → plane path without σ, a dependency on the calibration facade.
  That is an architecture decision for `vision-metrology`, not part of the
  1.0 path. Reopen when `vision-metrology` needs measurements with σ.
- [~] MVG-HANDEYE-STAGED - A staged laser export (`rig_laserline_device`)
  lists the hand-eye with its cross terms to the planes, so its rig measures
  in the gripper or base frame with the hand-eye's σ: its upstream maps the
  hand-eye into per-view board poses it does not report, and a rig built
  from it with `with_handeye` counts the hand-eye exact. Parked: the joint
  `rig_handeye_laserline` export carries it; reopen when a staged laser
  user needs robot-frame points with the hand-eye's share.
- [ ] MVG-TWO-VIEW - Two-view robustness and uncertainty:
  - optimal two-view triangulation (Lindstrom);
  - covariance of F and H (E and the relative pose have theirs from the
    robust variants' builder refinement, gauge |t| = 1); the all-inlier
    `recover_relative_pose` still returns its linear 5-point pose without a
    refinement or covariance; the calibration's share in the two-view
    covariance (a residual through the camera blocks of
    `add_calibrated_rig`, which needs the unprojection at dual numbers);
  - H-versus-F model selection (GRIC);
  - LO-RANSAC/MSAC with an adaptive iteration count;
  - P3P/EPnP inside RANSAC.

  Done when: each item is in `vision-mvg` with a synthetic ground-truth
  test, and every new covariance agrees with Monte-Carlo.

## Book (alongside)

- [ ] BOOK-WASM - A `vision-calibration-wasm` crate (wasm-bindgen over the
  facade, `publish = false` at first), built by `publish-docs.yml` with a
  bun bundle loaded through mdBook `additional-js`. The app can use the
  same build for per-frame math. Done when: the published book loads the
  wasm build and a page calls into it.
- [ ] BOOK-DEMOS - Interactive pages running the library in the browser,
  first:
  - triangulation σ against baseline, from a calibration's covariance;
  - planar calibration on synthetic views, with a noise slider,
    covariance ellipses and the projection-σ map;
  - a distortion and camera-model explorer.

  Then: how view placement changes σ, DLT with and without normalization,
  epipolar geometry, Scheimpflug tilt, hand-eye rotation diversity, robust
  losses. Needs: BOOK-WASM. Done when: the first three pages run in the
  published book.
- [ ] BOOK-RESULTS - A results page generated from the regression suite:
  the OpenCV comparison and solver timings. Done when: the book has the
  page and it is regenerated from the suite's results by a command, not
  edited by hand.

## App (alongside)

Design: ADR 0014; developer guide: `app/DEVELOPING.md`.

- [ ] APP-IA - Organize the app as Project → Dataset → Detect → Calibrate →
  Diagnose → Use.
  - **Dataset:** a per-camera × pose image grid.
  - **Detect:** detection overlays, a board coverage map and per-view
    statistics.
  - **Diagnose:** residuals, the quality report, uncertainty and run
    comparison.
  - **Use:** undistort and rectify, epipolar, depth, and measurement
    with σ (from `mvg::metrology`, on the stereo of `StereoRectification`).
  - Also covers empty states and the fail-fast error surfaces. Gate: a
    frontend review with no high-severity findings.
- [ ] APP-RUN-DISPATCH - The app's `run.rs` keeps its own topology dispatch
  (`run_*_topology`) beside `dataset_runner::calibrate_dataset`, which
  Python and `vcal` share. It keeps it for per-stage progress events and
  the image manifest it splices into each export. Move it onto
  `calibrate_dataset` with a `RunControl` built from its cancel token and
  observer, and surface the returned `rejected`, `usable_views` and
  `image_sizes`. Needs: DATA-VIEW-IDENTITY (which puts the manifest in
  every export). Gate: the app's run tests, and no `run_*_topology` left.
- [ ] APP-WIRE-TYPES - `app/src/types.ts` hand-writes `FrameRef`,
  `ImageManifest`, `PerFeatureResiduals` and `ExportDiagnostics` stricter
  than the generated wire types (`types/generated/diagnose-wire.ts`): the
  wire allows a null `roi` and `image_manifest` and an absent `target` and
  `per_feature_residuals`, the hand-written types do not. `AnyExport`
  (`store/types.ts`), `RunResponse` (`lib/runCalibration.ts`) and `RunRecord`
  (`project/projectApi.ts`) mirror Rust types that are not in the wire
  schema. Use the generated types, handle the absent cases where the
  components read them, and emit the last three from
  `src-tauri/src/bin/emit_schemas.rs`. Gate: no hand-written mirror of a
  Rust type in `app/src`.
- [ ] VIEW-ERRMAP - A continuous reprojection-error colour map in the
  report viewer (desktop and `calib-viewer`): over each image, interpolated
  from its features' residuals, and per camera, accumulated over all views
  in sensor coordinates, so a distortion or tilt misfit shows as a field.
  Gate: on a synthetic run with a known unmodelled radial term the
  per-camera map shows the radial pattern; a Vitest for the binning.
- [ ] VIEW-QUALITY - A verdict and checks panel from the run's
  `quality.json` (`vcal run -o`; the app computes the same report for its
  own runs). Gate: a fixture with a warn and a fail check renders both,
  each linking to its camera.
- [ ] VIEW-COMPARE - Two runs side by side in the report viewer: per-camera
  parameters with their σ and the difference in σ units, residual
  statistics, and the per-view matrix of differences. `calib-viewer` takes
  two runs. Gate: two fixture runs whose difference is known.
- [ ] VIEW-FEEDBACK - Notes attached to a camera, a view or a feature in
  the report viewer, saved as `feedback.json` in the run folder (a typed,
  versioned shape with a JSON Schema) for an agent to read. `calib-viewer`
  writes only that file, inside the run folder. Gate: a round-trip test of
  the file and a Playwright test that adds and reloads a note.

## Solver backends and performance (Phase 2)

- [ ] SOLVER-VALLEY-STOP - A stopping rule in units of the estimate's σ. The
  loop stops on a 1e-5 relative cost decrease, which in a long, flat valley
  is reached before the minimum: on a tilted camera with weak distortion the
  default stops up to 0.66σ short, and where the valley is short nothing
  moves (`notes/tilt-identifiability.md`). Design: stop when the undamped
  Gauss-Newton step's predicted decrease `½·ΔᵀJᵀJΔ` is below `ε²·σ̂²/2`,
  `σ̂² = 2·cost/(m − n)`, i.e. the remaining step is under `ε` standard
  deviations in the Fisher metric (`ε = 0.01`), next to the existing rules;
  opt-in, then the default once measured. Gate: the weak-distortion
  Monte-Carlo ratio within 10 % of 1; `calib-bench suite check` and the
  solver benchmark's iteration counts, both backends.
- [ ] TS-UPSTREAM - Propose the remaining tiny-solver fork branches upstream
  (0.19.0 carries the rest): a sphere manifold; the nalgebra 0.35 / faer
  0.24 bump; the sort-free symbolic structure (Jacobian pattern by counting
  sort, values scattered in place — `Problem::build_symbolic_structure`
  still argsorts the triplets); and a true SE(3) exponential map. Its
  `SE3Manifold` retracts with a "fake exp" (rotation and translation
  stepped separately, `x·(Exp(ω), v)`, no `V(ω)`), while factrs 0.3 uses
  the true SE(3) exp, so the two backends agree only to first order; the
  true exp makes them agree to second order. On release: replace
  `backend/s2.rs` with the upstream sphere manifold, rerun `calib-bench
  solver` for the symbolic structure, and see NALGEBRA-035; adopting the
  true SE(3) exp changes results slightly, so measure it on both backends
  (`calib-bench suite check`, the solver benchmark) first.
- [~] O-FACTRS-PARALLEL - Parallel linearization for factrs: rpl-cmu/fact-rs
  #56 is merged but unreleased (the release is breaking, so 0.4.0). Expected
  ~3.9× over serial factrs on 12 threads, bit-identical.
  **Trigger**: that release. Then, in one commit: require it with
  `features = ["rayon"]` (off, nothing runs in parallel); take its Jacobian
  as a faer-0.23 view (NALGEBRA-035); rerun `calib-bench solver` for both
  backends.
- [ ] PERF-ROBOT-REFINE - A single-camera hand-eye refinement with robot-pose
  refinement on takes about 350 ms for 10 views × 63 points, about 20× the
  same solve without it (two weighted solves plus covariance; measured in
  `vision-calibration/tests/handeye_init_comparison.rs`). Profile it and cut
  the repeated work; results stay bit-identical.
- [ ] PERF-ANALYTIC - Close the gap to Ceres. Planar intrinsics are an
  estimated 3–4× (12 threads) and ~7× (one thread) slower than Ceres 2.2 at
  matched tolerances (measured against tiny-solver 0.18, scaled by 0.19's
  speed-up in `calib-bench solver`, not re-measured). What remains is
  linearization: per-block library overhead, num-dual against
  Ceres' `Jet`, and differentiation through the SE(3) retraction. Next
  lever: an analytic `ReprojPoint` Jacobian, checked against autodiff in
  tests. Done when: the analytic Jacobian matches autodiff in tests, and
  the gap is re-measured against Ceres with the new `calib-bench solver`
  numbers committed.
- [ ] BENCH-COMPARE - Extend the external comparisons: the OpenCV harness to
  stereo rectification; `camera-intrinsic-calibration`
  (Rust, cross-platform) for the KB4 and EUCM models on the same
  observations; PoseLib and pycolmap for MVG; a Ceres timing harness where
  it builds on all three platforms. Done when: each comparison runs from
  the bench on Linux, macOS and Windows and its numbers are published
  (BOOK-RESULTS).
- [~] P2-BA-DENSITY - Corner budget for the joint rig + hand-eye BA
  (spatially distributed subsample, or per-stage decimation). Parked:
  reopen when the acceptance run of the six-camera `scheimpflug_rig`
  dataset exceeds about 5 min.
- [~] NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 / `faer-ext`
  0.8, a major release (AGENTS.md §7). Parked: the bump waits for a
  tiny-solver release (the bump is a fork branch, TS-UPSTREAM) and a factrs
  release on them. factrs `main` is on faer 0.24 / faer-ext 0.8 but still
  nalgebra 0.34; the shared LM reads its Jacobian only as pattern and
  values (`backend/normal_equations.rs`), so that side needs a zero-copy
  faer view, not a redesign. Batch it with other breaks. Reopen when both
  backends release on nalgebra 0.35.

## v1.0 release

- [ ] D4-RELEASE - Tag and publish `1.0.0-rc.1` by `docs/RELEASE-RUNBOOK.md`.
  The version sources already say `1.0.0-rc.1`; fold `[Unreleased]` into
  the rc.1 section of the CHANGELOG. `vision-calibration-cli` has never been
  published, so `release.yml` skips it: publish it by hand right after the
  tag and register its Trusted Publisher (runbook §4), so the next release
  publishes it. Then 1.0.0 after an rc series with no breaking change; the
  `semver-rc` CI job checks every PR from the first `v1.0.0-rc.*` tag on.
  Done when: `v1.0.0` is tagged and on crates.io and PyPI.
