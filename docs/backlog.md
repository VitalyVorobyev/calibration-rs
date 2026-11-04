# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §11); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the phases and gates these serve. Track headings
match the ROADMAP; each says its phase. Dependencies are named as
`Needs: <ID>`. Phase 1b tracks are listed in the order they run.

## Multiple-view geometry — MVG (Phase 1b)

Measurements from a calibrated system with error bars, designed in ADR
0028: MVG takes a `CalibratedRig`, refines through `optim::problem` only,
and propagates the calibration's covariance as consider parameters. Today
MVG takes a `CalibratedRig` and pixels in its closed-form steps, and the
two-view functions take matches with their stated noise (`PixelMatch`), and
triangulation refines the points and returns their covariance, the
calibration's share included, and bundle adjustment refines capture poses
and points through the builder (Consider or Prior, robust losses, an explicit
gauge, covariance). The robust two-view functions refine through the builder
and return the covariance of `E` and of the relative pose from the stated
pixel noise (the cameras as exact). The rectification still runs on a bare `K`.

- [ ] MVG-DENSE-COV - The calibration's share in a dense cloud's per-point
  covariance (today: disparity share; the calibrated σ goes through
  `StereoRectification::triangulate` for selected matches).
- [ ] MVG-METROLOGY-XREPO - `../vision-metrology` adopts `mvg::metrology`
  (a cross-repo change).
- [~] MVG-HANDEYE-STAGED - A staged laser export (`rig_laserline_device`)
  lists the hand-eye with its cross terms to the planes, so its rig measures
  in the gripper or base frame with the hand-eye's σ: its upstream maps the
  hand-eye into per-view board poses it does not report, and a rig built
  from it with `with_handeye` counts the hand-eye exact. Parked: the joint
  `rig_handeye_laserline` export carries it; reopen when a staged laser
  user needs robot-frame points with the hand-eye's share.

- [ ] HANDEYE-EXPORT-ENUM - The hand-eye exports (`single_cam_handeye`,
  `rig_handeye`, `rig_handeye_laserline`) carry a mode field and two optional
  poses (`gripper_se3_rig` / `rig_se3_base`, `base_se3_target` /
  `gripper_se3_target`), so an export with the wrong pose for its mode is
  representable. Carry core's `Handeye` enum (and the matching target pose)
  instead: a schema break, migration notes, Python and app types.

## Lean code — LEAN (Phase 1b)

- [ ] LEAN-PLANAR-INIT - `planar_intrinsics` initializes its pinhole
  cameras with its own `init_untilted` / `init_tilted` (~300 lines), a
  second copy of `planar_family::bootstrap_camera` that every other
  workflow uses (fisheye already goes through it). Route both through it.
  The copies differ in results: a tilt seed with auto-fitted intrinsics
  seeds the linear tilt sweep in `bootstrap_camera` but only replaces the
  swept tilt here; a zero tilt seed of an untilted camera is accepted
  there and rejected here; and here the fitted Brown-Conrady coefficients
  sit in state unconverted and the homographies are kept. Measure the
  first on the Scheimpflug suite entries and seed tests before switching.
  Gate: library lines down, results unchanged or the change measured and
  in the CHANGELOG.
- [ ] LEAN-RIG-TILTED - The rig stages run the wide tilt multi-start and
  the nominal recovery of a badly fitted tilted camera
  (`rig_family::tilted`) for `rig_handeye` only
  (`RigStages::RECOVERS_TILTED_CAMERAS`); `rig_extrinsics` gives an
  auto-initialized tilted camera the per-camera staged solve. Nothing
  records why. Measure both on tilted rigs (the synthetic recovery fixture
  of `tests/rig_tilted_recovery.rs`, the Scheimpflug rig tests) and either
  give both the recovery or state the reason in the trait item. Gate: one
  behaviour or a documented reason; results measured first.

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
- [ ] UQ-NONLINEAR - A weak capture leaves a calibration where the
  first-order covariance holds per parameter but not jointly: the stereo rig
  of the facade's Monte-Carlo tests (centred boards, tilts up to 0.2 rad)
  states every parameter's σ within 10 % yet scatters 2.5-3× its σ along two
  combinations (camera 1's yaw with the principal points; NEES/22 1.64,
  falling to 1.00 at a tenth of the pixel noise). Find a cheap indicator of
  that regime from the result (e.g. the curvature of the cost along the
  covariance's weakest directions, or a second-order term of the
  estimate), add it to the quality report as a check, and recommend views
  that leave it. Then move the five Monte-Carlo tests still calibrating on
  the weak views (pose, triangulation, stereo, metrology, board measure) to
  the spread views and update the book's numbers.
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
  RWHE-Calib sets (`kuka_2`, `kuka_3`) and Tabb's DS1–DS7 (U.S. public
  domain, same record as DS8).

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
  - covariance of F and H (E and the relative pose have theirs from the
    robust variants' builder refinement, gauge |t| = 1); the all-inlier
    `recover_relative_pose` still returns its linear 5-point pose without a
    refinement or covariance; the calibration's share in the two-view
    covariance (a residual through the camera blocks of
    `add_calibrated_rig`, which needs the unprojection at dual numbers);
  - H-versus-F model selection (GRIC);
  - LO-RANSAC/MSAC with an adaptive iteration count;
  - P3P/EPnP inside RANSAC.

## Book (alongside)

- [ ] BOOK-WASM - A `vision-calibration-wasm` crate (wasm-bindgen over the
  facade, `publish = false` at first), built by `publish-docs.yml` with a
  bun bundle loaded through mdBook `additional-js`. The app can use the
  same build for per-frame math.
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
    with σ (from `mvg::metrology`, on the stereo of `StereoRectification`).
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
- [ ] PERF-ROBOT-REFINE - A single-camera hand-eye refinement with robot-pose
  refinement on takes about 350 ms for 10 views × 63 points, about 20× the
  same solve without it (two weighted solves plus covariance; measured in
  `vision-calibration/tests/handeye_init_comparison.rs`). Profile it and cut
  the repeated work; results stay bit-identical.
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
