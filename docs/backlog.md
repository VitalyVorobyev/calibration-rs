# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §9); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the phases and gates these serve. Track headings
match the ROADMAP; each says its phase. Dependencies are named as
`Needs: <ID>`.

## Parameter uncertainty (Phase 1, then Phase 2)

Every result carries the covariance of its estimated parameters, computed
from measured feature noise, and a quality verdict built on it. Design:
ADR 0026; evidence: `notes/uncertainty.md`. Items that change results
need their measurement first and stay opt-in until it is in.

- [ ] UQ-REAL-SCALE - The stated projection σ is 2–5 times smaller than
  the real scatter on every real camera. In the feature-covariance study
  (`notes/feature-covariance-study.md`, E3), two calibrations from disjoint
  halves of the views differ by `r` = 4.9–26 times the predicted `z²/2` on
  the public cameras (pooled 7.8). Only 16 % of grid nodes fall inside the
  68 % contour. The detector's per-corner covariance does not change this.
  The missing error is systematic: the lens model, the target (see
  UQ-TARGET-FLATNESS) or detector bias. First measure which one dominates
  on `kuka_1`, `ds8` and `stereo` (residual structure, a richer distortion
  model, a flatness term). Then either model it, or state the real-data
  scale beside the covariance. Exit criterion 3 asks for the real-data
  check. Gate: split-half `r` within 0.85–1.15 on the public cameras, or a
  documented, measured inflation.
- [~] UQ-RINGGRID-COV - The ring-grid detector reports the ellipse-centre
  covariance (cross-repo, ringgrid). Parked: the real-data study advises
  against weighting a calibration by the chessboard and ChArUco
  covariance, so a ring-grid one would only be a report. Reopen if a
  ring-grid weighting study is wanted, or if UQ-REAL-SCALE needs the
  detector's term.
- [ ] UQ-CROSS-FRAME - (Phase 2) Projection uncertainty across the
  reference-frame shift (the mrcal method). Needs the intrinsics–pose
  cross-covariance, which results do not carry.
- [ ] UQ-ROBOT-NOISE - (Phase 2) Robot-pose noise in hand-eye as a measured
  quantity rather than a fixed prior. The robot-pose priors are weighed
  against the estimated corner noise, but their configured σ is taken as
  the robot's true accuracy: on `kuka_1` the corrections are about ten
  times smaller than the 0.5° default, so the hand-eye and target σ are
  pessimistic, and priors tighter than the robot's error make them too
  small. Estimate the prior family's variance factor from the corrections
  (a variance component with its redundancy), and gate it on the
  `robot_pose_priors` study's mismatched cells.
- [ ] UQ-ROBOT-FRAME - (Phase 1: it changes what `robot_deltas` mean, a
  breaking change.) Robot-pose corrections perturb the measured pose in
  the base frame (`exp(δ) · base_se3_gripper`, a rotation about the base
  origin), so `trans_sigma` is not the gripper's position error: a 0.5°
  rotation prior lets a gripper 1.5 m from the base move 13 mm, and on
  `kuka_1` the largest translation correction is 3.7 mm while the rotation
  ones stay below 0.11°. Measure a gripper-frame correction
  (`base_se3_gripper · exp(δ)`), whose σ are the tool's orientation and
  position errors, on `kuka_1`, `ds8` and the `robot_pose_priors` study. It
  changes what `robot_deltas` mean (`apply_robot_delta`, the app's
  overlays, the laser upstream in `dataset_runner`). Pairs with
  UQ-ROBOT-NOISE.
- [ ] UQ-TARGET-FLATNESS - (Phase 2) Target manufacturing and flatness error
  in the covariance.

## Regression suite (Phase 1)

A versioned public suite for regression and for comparison with other
libraries. Image-free observations live in the repository and gate every PR.
Real-image entries are fetched from their publishers, not archived by us:
`kuka_1` (RWHE-Calib) and `stereo` (Kaggle) state no licence; DS8 is U.S.
public domain. Synthetic entries need no images. Comparison tools must run
on Linux, macOS and Windows. `stereo_charuco` stays local-only, as do the
private sets.

- [ ] DATA-FETCH - `suite fetch` and the weekly `Suite images` workflow
  (fetch → `suite freeze --images` → `SHA256SUMS` unchanged) are in, and the
  workflow has run green on `main`. Left: retire
  `crates/vision-calibration-bench/registry/public.json` (its `ds8` entry
  still globs the old grayscale PNG conversion, as does
  `tools/convert_ds8_to_gray_png.py`; both go with it) and the acceptance
  job's image run. Ask the RWHE-Calib authors for a licence (maintainer
  action); without one that allows redistribution, drop the `kuka_1`
  images from the repository and LFS (the `planar_real`/`handeye_session`
  examples and the LFS-image unit tests then name the fetch command).
- [ ] DATA-MORE - More public entries, each after a licence check: the other
  RWHE-Calib sets (`kuka_2`, `kuka_3`), Tabb's DS1–DS7 (U.S. public domain,
  same record as DS8), OpenCV's sample stereo chessboards (Apache-2.0), and
  a fisheye set for CAM-KB.

## Calibration quality (Phase 1)

Found by `calib-bench solver` (well-conditioned synthetic scenes; the named
scenes are the gate) and the regression suite.

- [ ] Q-SCHEIMPFLUG-STAGED-ROBUST - The staged Scheimpflug solve
  (`optimize_scheimpflug_intrinsics_staged`, the cold-start and rig
  per-camera route) fails on contaminated data: its last stage, which frees
  the principal point, solves without the caller's robust loss and is
  accepted only if the mean reprojection error, outliers included, does not
  grow. With 5 % outliers it is rejected and the principal point keeps its
  linear-init value: `rig_extrinsics/scheimpflug/*/{huber,cauchy}` stay
  20–45 px and 2–3° off. Solving that stage under the caller's loss and
  accepting it on the objective brings `small/n0.1/{huber,cauchy}` to
  ≤ 4.7 px and ≤ 0.3° (about 1σ). It changes every robust-loss cold-start
  and rig Scheimpflug result, so check the acceptance runs.
  Gate: `rig_extrinsics/scheimpflug/*/{huber,cauchy}` within 3σ of the truth.
- [ ] Q-TANGENTIAL-ONE-SOURCE - Whether the tangential terms are estimated
  has two settings: `camera.fix.distortion` (the non-linear solves) and
  `intrinsics.fix_tangential` / `init.fix_tangential` (the linear stage).
  A camera whose `camera.fix` holds `p1`, `p2` while `fix_tangential` is
  unset gets them fitted in the linear stage and then frozen at that value,
  the fault that held tilted rig cameras off their minimum. Derive the
  linear stage's choice from the camera's fix list and drop
  `fix_tangential` (breaking, with a migration note). Gate: a camera holding
  `p1`, `p2` keeps them at zero through every stage, in every workflow.

- [ ] Q-SCHEIMPFLUG-TILE-BOUND - On every `scheimpflug_rig` tile, the
  single-camera seeded planar solve ends with `tilt_x` at its ±0.10 rad
  bound around the seed (`step_optimize`'s seeded path), so the covariance
  is unavailable (`at_bound`). The feature-covariance study found this on
  all six tiles, for every arm. Check whether the rig route's per-camera
  stage does the same. Then establish whether the seed is off or the tilt
  is not determined by one tile's views, and fix the seed or the bound.
  Gate: each tile's own solve ends inside its bounds, with a covariance.

## Camera models (Phase 2)

Each model ships with:
- a math note;
- a synthetic matrix test across every problem type;
- the Python class;
- a book chapter with a demo.

Each slots into the camera spec every workflow takes (ADRs 0005, 0013).

- [ ] CAM-KB - Kannala-Brandt equidistant fisheye (k1–k4) as a projection
  model. Gate: OpenCV `fisheye` comparison on a public fisheye set
  (DATA-MORE).
- [ ] CAM-EUCM-DS - Enhanced unified (EUCM) and double-sphere projections,
  with their closed-form unprojection and validity regions.
- [ ] CAM-TELECENTRIC - Telecentric (affine/orthographic) projection for
  metrology lenses: affine-factorization init, and the gauge of ADR 0005
  (held axial depth, the per-view reflection ambiguity).
- [ ] CAM-SELECT - Model selection: compare candidate models by residual
  structure, an information criterion and held-out views. It extends the
  distortion-model-selection tutorial.

## Calibration problems (Phase 2)

- [ ] CAL-RIG-GRAPH - Rig initialization chained through a camera
  co-visibility graph. Today `linear/src/extrinsics.rs` errors when a
  camera never shares a view with the reference camera. Gate: a synthetic
  four-camera ring where only neighbours overlap.
- [ ] CAL-MULTI-TARGET - Several targets with unknown relative poses in one
  scene, estimated jointly.
- [ ] CAL-HANDEYE-SOLVERS - Linear hand-eye initializers beyond Tsai-Lenz
  (`linear/src/handeye.rs`): dual-quaternion (Daniilidis), Park-Martin, and
  a robot-world AX=ZB solver (Shah). Gate: a synthetic comparison of basin
  and accuracy under rotation-poor motion.

## Multiple-view geometry (Phase 2)

Measurements from a calibrated system with error bars. Today no MVG API
takes or returns a covariance. Open with an ADR on the BA layering:
`vision-mvg` runs tiny-solver directly, outside the optimization IR.

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
- [ ] MVG-LASER-COV - Laser-triangulated 3D points with σ from the camera
  and the laser-plane covariance. Needs: MVG-TRI-COV.
- [ ] MVG-BA - Bundle adjustment on the optimization IR:
  - the composable camera model, not K only;
  - robust losses and optional intrinsics refinement;
  - gauge options (fixed datum or inner constraints);
  - covariance through the existing sandwich.

  It replaces `vision_mvg::bundle_adjust`. Gate: synthetic-GT matrix and
  Monte-Carlo σ.
- [ ] MVG-TWO-VIEW - Two-view robustness and uncertainty:
  - optimal two-view triangulation (Lindstrom);
  - covariance of F, E, H and the relative pose (gauge |t| = 1);
  - H-versus-F model selection (GRIC);
  - LO-RANSAC/MSAC with an adaptive iteration count;
  - P3P/EPnP inside RANSAC.
- [ ] MVG-PNP-COV - Object pose from PnP with its covariance.
- [ ] MVG-STEREO - Stereo from a calibration:
  - rectification straight from a rig export;
  - the disparity-to-depth matrix;
  - depth σ from disparity σ;
  - a point cloud with per-point covariance.
- [ ] MVG-PY - Python bindings for the MVG surface.
- [ ] MVG-METROLOGY - A pixel-to-millimetre measurement API with σ, used by
  `../vision-metrology` (cross-repo). Needs: MVG-TRI-COV.

## Book (alongside)

- [~] BOOK-WASM - Parked until after v1.0: large, and not the adoption lever
  (BOOK-RESULTS and CAM-KB are). Reopen after D4-RELEASE. A
  `vision-calibration-wasm` crate (wasm-bindgen over the
  facade, `publish = false` at first), built by `publish-docs.yml` with a
  bun bundle loaded through mdBook `additional-js`. The app can use the same
  build for per-frame math.
- [~] BOOK-DEMOS - Parked with BOOK-WASM. Interactive pages running the
  library in the browser:
  - a distortion and camera-model explorer;
  - planar calibration on synthetic views, with a noise slider,
    covariance ellipses and the projection-σ map;
  - how view placement changes σ;
  - DLT with and without normalization;
  - epipolar geometry;
  - triangulation σ against baseline;
  - Scheimpflug tilt;
  - hand-eye rotation diversity;
  - robust losses.

  Needs: BOOK-WASM.
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
    with σ.
  - Also covers empty states and the fail-fast error surfaces. Gate: a
    frontend review with no high-severity findings.
- [ ] APP-SAVE-REPORT - Save the export and a self-contained HTML report
  (parameters, σ, quality verdicts, residual plots) from a run.
- [ ] LIB-PROGRESS - An iteration observer and a cancellation token through
  the optimization loop and the pipeline steps. Today the app reports
  three coarse stages and cannot stop a running solve. Python maps
  KeyboardInterrupt onto the token.
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
  O-TINYSOLVER-PERF and D4-NALGEBRA-035.
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
  as a faer-0.23 view (D4-NALGEBRA-035); rerun `calib-bench solver` for both
  backends.
- [ ] PERF-ANALYTIC - Close the gap to Ceres. Planar intrinsics against Ceres
  2.2 at matched tolerances is 7–9× slower on 12 threads and about 21× on
  one (2.6× and 4.5× with the O-TINYSOLVER-PERF changes). What remains is
  linearization: per-block library overhead, num-dual against Ceres' `Jet`,
  and differentiation through the SE(3) retraction. Next lever: an analytic
  `ReprojPoint` Jacobian, checked against autodiff in tests.
- [ ] BENCH-COMPARE - Extend the external comparisons: the OpenCV harness to
  fisheye (CAM-KB) and stereo rectification; `camera-intrinsic-calibration`
  (Rust, cross-platform) for the KB4 and EUCM models on the same
  observations; PoseLib and pycolmap for MVG; a Ceres timing harness where
  it builds on all three platforms.

## v1.0

- [ ] D4-RELEASE - Cut v1.0 once the ROADMAP exit criteria hold.
- [ ] D4-NALGEBRA-035 - **Decide before D4-RELEASE** (ROADMAP criterion 9):
  ship 1.0 on `nalgebra` 0.34 and bump in 2.0, or wait for the backends.
  Move to `nalgebra` 0.35 / `faer` 0.24 /
  `faer-ext` 0.8. **Blocked on both backends**, built against 0.34 / 0.23 /
  0.7: `vision-calibration-optim` passes nalgebra and faer types across the
  tiny-solver boundary (`Factor<T: nalgebra::RealField>`,
  `faer::sparse::SparseColMat`, `faer_ext::IntoNalgebra`) and the factrs one
  (its variables and residual traits), so a bump fails to compile.
  `vision-calibration-detect` stays nalgebra-free at its boundary, which is
  what lets `calib-targets`' nalgebra 0.35 coexist.
  - tiny-solver: the bump is a fork branch (TS-UPSTREAM).
  - factrs `main` is on faer 0.24 / faer-ext 0.8 (nalgebra 0.34). The shared
    LM owns its sparse solve and reads a Jacobian only as pattern and values
    (`backend/normal_equations.rs`), so its next release needs a zero-copy
    faer-0.23 view of those slices, not a blocker.

## Deferred and parked

- [ ] P2-BA-DENSITY - Corner budget for the joint rig + hand-eye BA
  (spatially distributed subsample, or per-stage decimation). Schedule only
  if the acceptance run for the six-camera `scheimpflug_rig` dataset exceeds ~5 min.
- [~] V7-SCHEIMPFLUG-RIG-INTRINSICS-FLOOR - Drive the scheimpflug_rig from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by the
  maintainer; reopen only on their request.
