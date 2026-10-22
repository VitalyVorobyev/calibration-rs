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

- [ ] UQ-JOINT-LASER-TRUTH - `synth_rig_handeye_laserline`, the joint rig
  hand-eye laserline solve (no robust loss), lands `cam1.cam_se3_rig.rx`
  4.5σ from its truth, where the other synthetic entries stay within 2.2σ.
  Find the cause (model mismatch between the line-distance residual and the
  generator's pixel noise, family weighting, or a σ that misses a
  correlation) before trusting the joint covariance.
- [ ] UQ-PRIOR-WEIGHTS - Robot-pose refinement weighs pixel residuals at
  unit σ against the priors' `sqrt_info`, so with sub-pixel corners the
  priors are far too weak: on `kuka_1` (σ̂ 0.11 px) the robot-pose deltas
  absorb signal and σ(fx) is 0.95 px against 0.30 px with exact poses.
  Rescale the reprojection family by its σ̂ (one re-solve) or the priors by
  σ̂², and measure hand-eye accuracy against ground truth on synthetic
  scenes; results change, so it needs a migration note. Today σ reflects
  the priors, not the data: on `kuka_1` our intrinsics σ is 1.4–2.8×
  OpenCV's, which takes the robot poses as exact.
- [ ] UQ-RIG-INTRINSICS - Rig and rig hand-eye results hold the per-camera
  intrinsics fixed in their final solve by default, so their covariance has
  no intrinsics block and the extrinsics' σ is conditional on them. Decide:
  refine intrinsics in the final solve by default (results change), or
  report the per-camera intrinsics stage covariance alongside (loses the
  intrinsics–extrinsics correlation, which MVG-TRI-COV needs). Measure both
  on `stereo_rig` and `ds8`. How optimistic the conditional σ is: on
  `stereo_rig` the backends' intrinsics differ by 0.006σ(fx), which moves
  `cam1.cam_se3_rig.tz` by 0.4 of its conditional σ, so
  `suite check --backend factrs --sigma 1` passes only by that margin; with
  the intrinsics in the covariance it should tighten to about 0.05σ. Also
  drop `fix_poses: vec![0]` from the per-camera Scheimpflug stage of
  `rig_extrinsics/steps.rs` and `rig_handeye/steps.rs` (the single-camera
  Scheimpflug default does not fix a view).
- [ ] UQ-FEATURE-COV - Per-feature 2×2 covariance from the detector to the
  covariance of the result. σ̂ today is one pooled, isotropic value per
  residual family. Steps:
  1. **chess-rs (cross-repo):** the saddle-point refine
     (`chess-corners-core/src/refine/saddle_point.rs`) reports
     Σ = s²H⁻¹ from its quadratic fit.
  2. **calib-targets (cross-repo):** passes it through.
  3. **ringgrid (cross-repo):** reports the ellipse-centre covariance.
  4. **Here:**
     - `Feature` gains an optional `cov_px`;
     - reprojection factors whiten with a 2×2 square-root information
       matrix in place of the scalar weight;
     - σ̂ becomes the a-posteriori variance factor σ̂₀, ≈1 when the
       detector's noise model holds;
     - opt-in `weighting: Uniform | DetectorCovariance`.
  5. **Evidence:** etendue and calib-synth renders with exact corners,
     across blur, noise and scale. The predicted corner scatter must
     match the empirical scatter, and the predicted parameter σ must
     match the Monte-Carlo σ. Then flip the default, with a migration
     note.

  Gate: a proof-pack note with both comparisons.
- [ ] UQ-NBV-CORE - Predicted covariance after adding a candidate view
  (rank-update of the information matrix), and a ranking of candidate
  target poses by the reduction in maximum projection σ. Library only; the
  app's Diagnose view and the book demo consume it. Gate:
  prediction matches the re-solve on synthetic scenes.
- [ ] UQ-CROSS-FRAME - (Phase 2) Projection uncertainty across the
  reference-frame shift (the mrcal method). Needs the intrinsics–pose
  cross-covariance, which results do not carry.
- [ ] UQ-ROBOT-NOISE - (Phase 2) Robot-pose noise in hand-eye as a measured
  quantity rather than a fixed prior. Needs: UQ-PRIOR-WEIGHTS.
- [ ] UQ-TARGET-FLATNESS - (Phase 2) Target manufacturing and flatness error
  in the covariance.

## Python (Phase 1)

- [ ] PY-TYPES - Generate the Python `TypedDict`s and `__init__.pyi` stubs
  from the JSON schemas `cargo xtask emit-schemas` already writes for the
  app, and delete the hand-written duplicates in `types.py`. Gate: a schema
  drift check in CI covers the generated Python.
- [ ] PY-PARITY - Bring the Python package to the facade:
  - the session and step API, including manual init (Rust-only today);
  - manifest-driven runs (`dataset_runner`) and the detectors on numpy
    images;
  - camera objects with vectorized `project`, `unproject` and
    `undistort`;
  - typed per-feature residuals.

  `scripts/check_binding_parity.py` becomes a per-item map of the facade
  that CI enforces; today it only matches `run_*` functions. Gate: the
  map lists no unmapped item.

## Regression suite (Phase 1)

A versioned public suite for regression and for comparison with other
libraries. Image-free observations live in the repository and gate every PR.
Real-image entries are fetched from their publishers, not archived by us:
`kuka_1` (RWHE-Calib) and `stereo` (Kaggle) state no licence; DS8 is U.S.
public domain. Synthetic entries need no images. Comparison tools must run
on Linux, macOS and Windows. `stereo_charuco` stays local-only, as do the
private sets.

- [ ] DATA-FETCH - `calib-bench suite fetch` downloads an entry's images
  from its publisher into a cache and verifies a per-file SHA-256 list:
  `kuka_1` from RWHE-Calib at commit `610a412` (raw files), DS8 from the Ag
  Data Commons (`DS8.zip`). Kaggle needs an account, so `stereo` stays a
  manual download. A scheduled CI job runs `suite freeze` from the fetched
  images and checks the observations still match `SHA256SUMS`; then retire
  `crates/vision-calibration-bench/registry/public.json` and the acceptance
  job's image run. Ask the RWHE-Calib authors for a licence; without one
  that allows redistribution, drop the `kuka_1` images from the repository
  and LFS (the `planar_real`/`handeye_session` examples then name the fetch
  command).
- [ ] DATA-MORE - More public entries, each after a licence check: the other
  RWHE-Calib sets (`kuka_2`, `kuka_3`), Tabb's DS1–DS7 (U.S. public domain,
  same record as DS8), OpenCV's sample stereo chessboards (Apache-2.0), and
  a fisheye set for CAM-KB.

## Calibration quality (Phase 1)

Found by `calib-bench solver` (well-conditioned synthetic scenes; the named
scenes are the gate) and the regression suite.

- [ ] Q-LASERLINE-OUTLIERS - With 5 % of target corners displaced 10–30 px,
  `laserline_device` under Huber or Cauchy (scale 1 px) settles off the
  minimum: inlier RMS 0.16–0.30 px at σ 0.1 px (floor 0.14), focal error
  0.4–1.8 %, laser plane up to 0.29° / 7.6 mm. The other problems (the
  Scheimpflug rig aside, below) reach the floor on the same contamination,
  and the robust-cost LM does not change it, so the init is the suspect. Gate:
  `laserline_device/pinhole/*/n0.1/{huber,cauchy}` at the noise floor.
- [ ] Q-SCHEIMPFLUG-RIG-PERCAM - The Scheimpflug rig's per-camera stage
  (staged init, radial-only BA) lands off the minimum on clean data, even
  seeded with each camera's nominal tilt: inlier RMS 0.21–0.26 px at
  σ 0.1 px, principal point 26–44 px and tilt about 2.5° off. The rig BA
  keeps intrinsics fixed, so it cannot recover; the single-camera Scheimpflug
  problem reaches the floor on comparable data (principal point 1.4 px).
  Gate: `rig_extrinsics/scheimpflug/*/clean` at the noise floor.
- [ ] Q-CHESS-SHARED-FRAME - A plain chessboard detection labels corners
  from the detection's own top-left corner, and accepts a board seen turned
  by 90° with its labels rotated. Where views share one target frame (rig
  and hand-eye topologies) a partial or turned board therefore moves the
  target frame between views or cameras: DS8 with partial boards kept
  reprojects at 4.5 px instead of 0.26 px. Manifests can keep complete boards
  only (`detector.min_features_per_view` = corner count); make the shared-
  frame topologies reject inconsistent plain-chessboard labelings by default
  (complete and in the declared orientation) and say so in the manifest docs.

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
  Needs: MVG-OBS, UQ-RIG-INTRINSICS.
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
- [ ] MVG-RANSAC-THRESH - `RansacOptions::default()` sets `thresh: 2.0`, a
  pixel-scale value, but the robust two-view estimators in `mvg::robust`
  measure residuals in normalized coordinates, where 2.0 accepts nearly
  everything (the tests use 0.001). Give each robust estimator a threshold
  in its own unit (pixels, converted through the intrinsics) or a required
  argument, so no default silently disables outlier rejection.
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

- [ ] BOOK-WASM - A `vision-calibration-wasm` crate (wasm-bindgen over the
  facade, `publish = false` at first), built by `publish-docs.yml` with a
  bun bundle loaded through mdBook `additional-js`. The app can use the same
  build for per-frame math.
- [ ] BOOK-DEMOS - Interactive pages running the library in the browser:
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

The frontend is rebuilt on the `@vitavision/workbench` shell. The Rust
backend and the generated IPC types stay, and the existing views are ported
into the new structure. ADR 0014 is rewritten with APP-SHELL.

- [ ] APP-SHELL - Workbench app shell and current `@vitavision/*` versions:
  - a project-folder model (manifest plus `runs/<name>/`, saved and
    reopened);
  - images over Tauri's asset protocol instead of base64 over IPC;
  - workspace modules split to a reviewable size (`RunWorkspace` is
    1,265 lines today).
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
    frontend review with no high-severity findings. Needs: APP-SHELL.
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
- [ ] D4-NALGEBRA-035 - Move to `nalgebra` 0.35 / `faer` 0.24 /
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
  if the acceptance run for the six-camera rtv3d rig exceeds ~5 min.
- [~] V7-RTV3D-INTRINSICS-FLOOR - Drive the rtv3d from-scratch reprojection
  floor below 0.4 px. The blocking term is in the detector/target/model,
  not the rig chain; seeded init is the official route. Parked by the
  maintainer; reopen only on their request.
