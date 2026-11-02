# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §11); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the phases and gates these serve. Track headings
match the ROADMAP; each says its phase. Dependencies are named as
`Needs: <ID>`. Phase 1b tracks are listed in the order they run.

## Extension API — EXT (Phase 1b)

A user builds their own pipeline from the facade. Today the optimization IR,
the factors and the backends are private and `ProblemType` is sealed, so a
new residual or problem needs a fork.

- [ ] EXT-ADR - ADR for the public optimization extension:
  - a user residual trait with a generic `eval<T: RealField>`, bridged to an
    object-safe trait behind `FactorKind::Custom`;
  - a public IR builder subset (parameter blocks, manifolds, `FixedMask`,
    robust losses);
  - a public solve returning the solution and the covariance over labelled
    blocks, reusing the existing covariance estimator;
  - user residuals run on tiny-solver; on factrs they are a typed error;
  - `ProblemType` stays sealed: composition goes through public building
    blocks, with the closed-form initializers re-exported on the facade.
- [ ] EXT-FACTOR - Implement it, with a downstream test crate that uses only
  the facade, Python where it applies, and a book chapter "Your own
  residual" (worked example: a known-distance constraint). Needs: EXT-ADR.
- [ ] EXT-COMPOSE - A book chapter "Compose your own pipeline": closed-form
  init → IR → solve → σ from the public pieces. The contributor chapters on
  new problem types say they are for contributors. Needs: EXT-FACTOR.

## Multiple-view geometry — MVG (Phase 1b)

Measurements from a calibrated system with error bars. Today MVG takes a
bare `K` (no distortion, fisheye or Scheimpflug), takes and returns no
covariance, runs BA on tiny-solver outside the optimization IR, and has no
Python binding and no real-data demonstration.

- [ ] MVG-ADR - ADR for the MVG layering and uncertainty:
  - BA and covariance-bearing refinement move to `optim` on the EXT API;
    the facade keeps the `mvg` paths; `vision-mvg` keeps the deterministic
    pipelines and drops its tiny-solver dependency;
  - MVG takes calibrated cameras (`CameraParams`, rig exports);
  - σ propagation from measurement noise and the calibration's covariance,
    keeping the cross-covariance a shared calibration induces;
  - gauge and scale anchoring for BA.
  Needs: EXT-ADR.
- [ ] MVG-CAMERA-IN - MVG APIs take the composable camera and rig exports
  instead of a bare `K`, so every camera model works (bearings through the
  camera's own unprojection). Remove the duplicate triangulator and
  `refine_point` between `vision-geometry` and `vision-mvg`; align names
  with ADR 0024. Needs: MVG-ADR.
- [ ] MVG-OBS - Observations carry a 2×2 covariance. Add Mahalanobis
  reprojection and covariance-weighted Sampson residuals.
- [ ] MVG-TRI-COV - N-view triangulation returns each point's covariance,
  from two sources:
  - measurement noise;
  - calibration noise: the export's covariance through the projection
    Jacobian.

  Derived quantities (distances, angles, plane fits) propagate σ jointly
  and keep the cross-covariance the shared calibration induces between
  points. Gate: Monte-Carlo over pixel noise and calibration samples.
  Needs: MVG-OBS.
- [ ] MVG-BA - Bundle adjustment on the optimization IR through the EXT API:
  - the composable camera model, not `K` only;
  - robust losses and optional intrinsics refinement;
  - gauge options (fixed datum or inner constraints);
  - covariance through the existing sandwich.

  It replaces the current `bundle_adjust`. Gate: synthetic-GT matrix and
  Monte-Carlo σ. Needs: EXT-FACTOR, MVG-CAMERA-IN.
- [ ] MVG-STEREO - Stereo from a calibration:
  - rectification straight from a rig export, any supported camera model;
  - the disparity-to-depth matrix;
  - depth σ from disparity σ and the calibration;
  - a point cloud with per-point covariance.
  The app's Depth workspace then uses it instead of its own conversion.
- [ ] MVG-METROLOGY - A measurement API (point, distance, angle, plane fit)
  with joint σ, used by `../vision-metrology` (cross-repo). Needs:
  MVG-TRI-COV.
- [ ] MVG-PNP-COV - Object pose from PnP with its covariance.
- [ ] MVG-LASER-COV - Laser-triangulated 3D points with σ from the camera
  and the laser-plane covariance. Needs: MVG-TRI-COV.
- [ ] MVG-PY - Python bindings for the MVG surface.
- [ ] MVG-DEMO - A book walk-through plus Rust and Python examples on the
  fetchable samples (`opencv_fisheye_stereo`, a pinhole stereo pair once
  CAL-CHESSBOARD-TURN lands): calibrate, triangulate the board corners of
  held-out views, measure known square lengths with σ, and compare the
  predicted σ with the actual error on the known geometry.
  Needs: MVG-METROLOGY.

## Correctness — CORRECT (Phase 1b)

- [ ] CAL-LASERPLANE-UNIT - `LaserPlane`'s normal is a `Unit<Vector3>`
  that nalgebra deserializes without renormalizing, and `LaserPlane::new`
  normalizes the normal without rescaling the distance, which moves a plane
  given with a non-unit normal. Only the joint solve's seed check (#256)
  rejects such a plane; `RigLaserlineDeviceInput::initial_planes_cam`,
  `PlaneManualInit` and the laser exports read back from JSON do not. Make
  `LaserPlane` validate at construction and on deserialize (one fallible
  constructor that divides both by the norm, or rejects), and drop the
  per-call checks. Gate: a non-unit plane cannot be built or deserialized.
- [ ] CAL-INIT-DEGENERATE - Planar initialization fails on a degenerate
  capture with `numerical failure: invalid sign for lambda; check
  homographies`, which tells the user nothing. The `capture_planning`
  example reproduces it: views all fronto-parallel, or all tilted the same
  way (parallel board planes). Tilting about one axis only passes init and
  returns σ(fx) ≈ 96 px instead. Detect the degeneracy from the homographies
  before Zhang's solve (rank of the constraint system, or the spread of the
  board normals) and return a typed error that names the fix: tilt the board
  about both of its axes. Gate: both cases above return it, a good capture
  is unaffected.
- [ ] CAL-CHESSBOARD-TURN - A plain chessboard in a rig loses most views of
  a hand-held capture. The dataset runner rejects every view whose board is
  seen turned by 90° (`TurnedChessboard`), and cannot tell a 180° turn at
  all, because the detector labels each image from its own top-left corner.
  On OpenCV's stereo sample (`samples/data/left*.jpg`, `right*.jpg`, a 6×9
  board turned in most views) `rig_extrinsics` keeps 4 of 13 views; with
  rows and cols swapped it keeps 9 and diverges (σ(fx) ≈ 65 px), while
  `planar_intrinsics` uses all 13. A rig needs labels that agree across the
  cameras of one view, not across views: relabel each view's detections to
  a common orientation per view (from the cameras' relative geometry or the
  board's asymmetry) instead of rejecting them. Gate: the OpenCV pair
  calibrates as a rig from all 13 views; then add it to the samples.

- [ ] CAL-RIG-GRAPH - Rig initialization chained through a camera
  co-visibility graph. Today `linear/src/extrinsics.rs` errors when a
  camera never shares a view with the reference camera. Gate: a synthetic
  four-camera ring where only neighbours overlap.
- [ ] CAL-HANDEYE-SOLVERS - Linear hand-eye initializers beyond Tsai-Lenz
  (`linear/src/handeye.rs`): dual-quaternion (Daniilidis), Park-Martin, and
  a robot-world AX=ZB solver (Shah). Gate: a synthetic comparison of basin
  and accuracy under rotation-poor motion.
- [ ] LIB-PROGRESS - An iteration observer and a cancellation token through
  the optimization loop and the pipeline steps. Today the app reports
  three coarse stages and cannot stop a running solve. Python maps
  KeyboardInterrupt onto the token.

## Lean code — LEAN (Phase 1b)

- [ ] LEAN-PIPELINE - Factor the step logic the rig, hand-eye and laser
  problem types share (`rig_handeye/steps.rs` alone is ~1.5k lines) into
  the existing `planar_family` / `rig_family` helpers. Gate: library code
  lines down, every workflow's results bit-identical, regression suite
  unchanged.

## Parameter uncertainty (Phase 2)

Every result carries the covariance of its estimated parameters, computed
from measured feature noise, and a quality verdict built on it. Design:
ADR 0026; evidence: `notes/uncertainty.md`. Items that change results
need their measurement first and stay opt-in until it is in.

- [~] UQ-RINGGRID-COV - The ring-grid detector reports the ellipse-centre
  covariance (cross-repo, ringgrid). Parked: the real-data study advises
  against weighting a calibration by the chessboard and ChArUco
  covariance, so a ring-grid one would only be a report. Reopen if a
  ring-grid weighting study is wanted, or if the view scatter on ring-grid
  data needs the detector's term.
- [ ] UQ-CROSS-FRAME - Projection uncertainty across the reference-frame
  shift (the mrcal method). Needs the intrinsics–pose cross-covariance,
  which results do not carry.
- [ ] UQ-TARGET-FLATNESS - Target manufacturing and flatness error in the
  covariance.

## Regression suite (alongside)

A versioned public suite for regression and for comparison with other
libraries. Image-free observations live in the repository and gate every PR.
Real-image entries are fetched from their publishers, not archived by us:
`kuka_1` (RWHE-Calib), `stereo` (Kaggle) and the OpenCV fisheye set
(`opencv_extra`) state no licence; DS8 is U.S. public domain. Synthetic
entries need no images. Comparison tools must run on Linux, macOS and
Windows. `stereo_charuco` stays local-only, as do the private sets.

- [~] DATA-LICENCE-KUKA - The `kuka_1` images (RWHE-Calib, published
  without a licence) are committed through LFS. Parked on the maintainer
  asking the RWHE-Calib authors for a licence. Reopens when they answer: on
  one that does not allow redistribution, drop the images from the
  repository and LFS (they are fetchable with `suite fetch --only kuka_1`;
  the `planar_real`/`handeye_session` examples and the LFS-image unit tests
  then name the fetch command).
- [ ] DATA-MORE - More public entries, each after a licence check: the other
  RWHE-Calib sets (`kuka_2`, `kuka_3`), Tabb's DS1–DS7 (U.S. public domain,
  same record as DS8) and OpenCV's sample stereo chessboards (Apache-2.0).

## Camera models (Phase 2)

Each model ships with:
- a math note;
- a synthetic matrix test across every problem type;
- the Python class;
- a book chapter with a demo.

Each slots into the camera spec every workflow takes (ADRs 0005, 0013).

- [ ] CAM-EUCM-DS - Enhanced unified (EUCM) and double-sphere projections,
  with their closed-form unprojection and validity regions.
- [ ] CAM-TELECENTRIC - Telecentric (affine/orthographic) projection for
  metrology lenses: affine-factorization init, and the gauge of ADR 0005
  (held axial depth, the per-view reflection ambiguity).
- [ ] CAM-SELECT - Model selection: compare candidate models by residual
  structure, an information criterion and held-out views. It extends the
  distortion-model-selection tutorial.

## Calibration problems (Phase 2)

- [ ] CAL-MULTI-TARGET - Several targets with unknown relative poses in one
  scene, estimated jointly.

## Multiple-view geometry (Phase 2)

- [ ] MVG-TWO-VIEW - Two-view robustness and uncertainty:
  - optimal two-view triangulation (Lindstrom);
  - covariance of F, E, H and the relative pose (gauge |t| = 1);
  - H-versus-F model selection (GRIC);
  - LO-RANSAC/MSAC with an adaptive iteration count;
  - P3P/EPnP inside RANSAC.

## Book (alongside)

- [ ] BOOK-WASM - A `vision-calibration-wasm` crate (wasm-bindgen over the
  facade, `publish = false` at first), built by `publish-docs.yml` with a
  bun bundle loaded through mdBook `additional-js`. The app can use the
  same build for per-frame math. Needs: MVG-TRI-COV (the first demos show
  it).
- [ ] BOOK-DEMOS - Interactive pages running the library in the browser,
  first:
  - triangulation σ against baseline, from a calibration's covariance;
  - planar calibration on synthetic views, with a noise slider,
    covariance ellipses and the projection-σ map;
  - a distortion and camera-model explorer.

  Then: how view placement changes σ, DLT with and without normalization,
  epipolar geometry, Scheimpflug tilt, hand-eye rotation diversity, robust
  losses. Needs: BOOK-WASM.
- [ ] BOOK-RESULTS - A results page generated from the regression suite:
  the OpenCV comparison and solver timings.

## App (alongside)

The app runs on the `@vitavision/workbench` shell with project folders and
asset-protocol images (ADR 0014); the five workspaces are ported as they
were. What is left reorganizes them around the project.

- [ ] APP-IA - Organize the app as Project → Dataset → Detect → Calibrate →
  Diagnose → Use.
  - **Dataset:** a per-camera × pose image grid.
  - **Detect:** detection overlays, a board coverage map and per-view
    statistics.
  - **Diagnose:** residuals, the quality report, uncertainty and run
    comparison.
  - **Use:** undistort and rectify, epipolar, depth, and measurement
    with σ (from MVG-STEREO and MVG-METROLOGY).
  - Also covers empty states and the fail-fast error surfaces. Gate: a
    frontend review with no high-severity findings.
- [ ] APP-DIST - Bundle the presets, so installers run without a source
  checkout (`repo_root_cmd` uses `CARGO_MANIFEST_DIR` today).
- [ ] B-QUAL-TS7 - Move to TypeScript 7. **Blocked**: `typescript-eslint`
  rejects TS 7 (typescript-eslint#10940), and adopting it would drop the
  type-aware lint config. Revisit when typescript-eslint supports TS 7.

## Solver backends and performance (Phase 2)

- [ ] TS-UPSTREAM - Propose the remaining tiny-solver fork branches upstream:
  variables no residual uses left out of the solve; a sphere manifold; the
  nalgebra 0.35 / faer 0.24 bump; the internal perf changes (Mutex-free
  assembly, sort-free symbolic structure). Keep the backend's layout for a
  fully fixed manifold block (no manifold, so no columns): upstream keeps its
  tangent columns with a zero Jacobian, which would make `JᵀJ` singular for
  the covariance. On release: replace
  `backend/s2.rs` with the upstream sphere manifold, and see
  O-TINYSOLVER-PERF and NALGEBRA-035.
- [ ] O-TINYSOLVER-PERF - Adopt the tiny-solver speed-ups once released.
  Results stay bit-identical and no code changes here; expected 3.9× faster
  solves on 12 threads, which `calib-bench solver` should confirm.
  - Mutex-free residual and Jacobian assembly (fits 0.18.x).
  - Jacobian pattern by counting sort, values scattered in place (fits
    0.18.x).
  - Stack duals in passes of 4 directions, as Ceres'
    `DynamicAutoDiffCostFunction` (adds supertraits to `FactorImpl` and
    `Manifold`, so a 0.19 change).

  **Trigger**: the release that carries them. Bump the floor, rerun
  `calib-bench solver`.
- [~] O-FACTRS-PARALLEL - Parallel linearization for factrs: rpl-cmu/fact-rs
  #56 is merged but unreleased (the release is breaking, so 0.4.0). Expected
  ~3.9× over serial factrs on 12 threads, bit-identical.
  **Trigger**: that release. Then, in one commit: require it with
  `features = ["rayon"]` (off, nothing runs in parallel); take its Jacobian
  as a faer-0.23 view (NALGEBRA-035); rerun `calib-bench solver` for both
  backends.
- [ ] PERF-ANALYTIC - Close the gap to Ceres. Planar intrinsics against Ceres
  2.2 at matched tolerances is 7–9× slower on 12 threads and about 21× on
  one (2.6× and 4.5× with the O-TINYSOLVER-PERF changes). What remains is
  linearization: per-block library overhead, num-dual against Ceres' `Jet`,
  and differentiation through the SE(3) retraction. Next lever: an analytic
  `ReprojPoint` Jacobian, checked against autodiff in tests.
- [ ] BENCH-COMPARE - Extend the external comparisons: the OpenCV harness to
  stereo rectification; `camera-intrinsic-calibration`
  (Rust, cross-platform) for the KB4 and EUCM models on the same
  observations; PoseLib and pycolmap for MVG; a Ceres timing harness where
  it builds on all three platforms.
- [~] NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 / `faer-ext`
  0.8. Parked: optim passes these types across both backend boundaries
  (AGENTS.md §7), so the bump waits for a tiny-solver release (the bump is
  a fork branch, TS-UPSTREAM) and a factrs release on them. factrs `main` is
  on faer 0.24 / faer-ext 0.8 but still nalgebra 0.34; the shared LM reads
  its Jacobian only as pattern and values (`backend/normal_equations.rs`),
  so that side needs a zero-copy faer view, not a redesign. Landing it
  before 1.0 keeps it a minor; after 1.0 it waits for 2.0. Decide before
  the 1.0 rc whether 1.0 waits for it.

## v1.0

- [ ] D4-RELEASE - Cut `1.0.0-rc.1` once the ROADMAP exit criteria hold,
  then 1.0 after an rc series with no breaking change. Needs: every Phase
  1b track.

## Deferred and parked

- [ ] P2-BA-DENSITY - Corner budget for the joint rig + hand-eye BA
  (spatially distributed subsample, or per-stage decimation). Schedule only
  if the acceptance run for the six-camera `scheimpflug_rig` dataset exceeds ~5 min.
- [~] V7-SCHEIMPFLUG-RIG-INTRINSICS-FLOOR - Drive the scheimpflug_rig from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by the
  maintainer; reopen only on their request.
