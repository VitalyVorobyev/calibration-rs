# Backlog

Open (`[ ]`) and parked (`[~]`) tasks only. Finishing a task deletes its entry
(AGENTS.md §11); the CHANGELOG and git history record what landed. The
[ROADMAP](ROADMAP.md) gives the phases and gates these serve. Track headings
match the ROADMAP; each says its phase. Dependencies are named as
`Needs: <ID>`.

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
- [ ] UQ-CROSS-FRAME - Projection uncertainty across the reference-frame
  shift (the mrcal method). Needs the intrinsics–pose cross-covariance,
  which results do not carry.
- [ ] UQ-TARGET-FLATNESS - Target manufacturing and flatness error in the
  covariance.

## Datasets and the runner (alongside)

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
  PY-MODELS-GEN.
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
- [ ] CI-PRIVATE-EXAMPLES - `vision-calibration-examples-private` is built
  by no CI job, because it path-depends on the sibling `vision-metrology`
  checkout; an example there stopped compiling unnoticed. Check it in CI
  with that repository checked out beside this one, or move the laser
  extraction it needs behind a feature so the rest checks alone.

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
- [ ] CAL-TILT-SEED-INIT - A tilt seed without an intrinsics seed fits the
  intrinsics at the seeded tilt (`planar_family::bootstrap_camera`, every
  workflow), but only from the candidates the linear Scheimpflug init builds
  around that tilt: the seed with the untilted Zhang `K` as reference, and
  the homography-decomposition fits started there, whose tilt drifts before
  the seed replaces it. When those fit badly the start is degenerate: one
  Monte-Carlo draw in 20 at `k1 = -0.30` and 0.087 rad tilt (linear `fx`
  half the truth), and one of the 12 private Scheimpflug cameras (linear
  reprojection 20 px, the solve ends on the tilt bound), where the full
  sweep found the basin. Evaluate every candidate reference `K` of the
  full sweep (grid, decomposition, refinement) at the seeded tilt and keep
  the best, a fixed-tilt mode of `estimate_scheimpflug_intrinsics_iterative`.
  Gate: those draws and that camera reach the full-seed result, no other
  result moves; measured on both backends.
- [ ] REPORT-SECTIONS - The HTML report (`analysis::html_report`, shared by
  `vcal` and the app) renders the quality checks and one section per
  camera. A hand-eye export shows no hand-eye (the pose, its σ, the robot
  refinement), and a laser export shows no planes. Add those sections
  (additive). Gate: `vision-calibration-bench/tests/html_report.rs` checks
  them for every problem type.

## Multiple-view geometry (Phase 2)

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
pixel noise (the cameras as exact). Rectification takes a `CalibratedRig`;
`StereoRectification::triangulate` and `StereoRectification::dense_cloud`
carry the calibration's share.

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
- [ ] B-QUAL-TS7 - Move to TypeScript 7. **Blocked**: `typescript-eslint`
  rejects TS 7 (typescript-eslint#10940), and adopting it would drop the
  type-aware lint config. Revisit when typescript-eslint supports TS 7.
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
  so that side needs a zero-copy faer view, not a redesign. Decided: 1.0
  ships on nalgebra 0.34 and does not wait for the backends; the bump is a
  2.0 change, to batch with other breaks. Reopen when both backends release
  on nalgebra 0.35.

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
