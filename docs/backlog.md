# Backlog

Execution status for agent-driven implementation tasks. Each completed task gets
a short report under `docs/report/` and a task-scoped commit.

Open tasks below derive from the
[2026-06-11 workspace review](report/2026-06-11-workspace-review.md) and the
V/O/M/B tracks in the [ROADMAP](ROADMAP.md). Findings F1–F6 reference that
review.

## V — rtv3d validation

- [x] V1-EXAMPLE - `rtv3d_rig` example: ChArUco rig hand-eye.
  Completed 2026-06-11. `detect_charuco` added to `examples-private/src/lib.rs`
  (calib-targets bumped 0.8 → 0.9); `examples/rtv3d_rig.rs` runs detection →
  Scheimpflug rig hand-eye → laser → joint BA with oracle comparison tables.
  Findings: hand-eye is **EyeToHand** (dataset.json's EyeInHand is wrong —
  3× residual evidence), cell size **5.2 mm**, rtv3d_2 is byte-identical to
  rtv3d minus laser images.
- [x] V2-LASER - rtv3d full pipeline. Completed 2026-06-11. Joint BA:
  1.16 px mean reproj, laser point-to-plane σ 0.017–0.031 mm over 1672–1868
  points/camera.
- [x] V3-REPORT - Beat-the-oracle report. Completed 2026-06-11 —
  [report](report/2026-06-11-rtv3d-validation.md). All criteria PASS on
  rtv3d (reproj < oracle on all cams incl. a sane cam 5; σ < oracle on all
  6 planes; extrinsic scale within 10 %). Plane-parameter and pose-delta
  comparisons demoted to informational (parameterization valley + legacy
  frame convention).
- [x] V4-BENCH - Registry entries. Completed 2026-06-11. `rtv3d` (+laser
  extraction profile) in `bench/registry/private.json`,
  smoke-tested (unseeded bench reproduces 1.86 px). 2026-06-12: the redundant
  rtv3d_2 dataset (byte-identical minus laser) and its registry entry were
  deleted; rtv3d_1 renamed to plain `rtv3d`.
- [x] V5-BENCH-LASER - Full `RigLaserlineDevice` + joint BA runner in
  `bench/src/run.rs`. Completed 2026-06-13 —
  [report](report/2026-06-13-V5-BENCH-LASER-rtv3d-calibration-quality.md).
  The bench now runs target detection once, then `RigHandeye` →
  `RigLaserlineDevice` → joint hand-eye/laser BA, with rtv3d detector
  threshold override, manual Scheimpflug seeds, app parity hooks, and
  diagnostic sweeps. Local V5 floor: 1.19 px mean reprojection, laser
  point-to-plane RMS 0.018–0.035 mm over 10,797 points. Laser criterion passes;
  reprojection remains above the 0.4 px target.
- [x] RTV3D-FROZEN-LASER-POSES - Preserve optimized upstream hand-eye target
  poses in the frozen rig-laserline app path. Completed 2026-06-13 —
  [report](report/2026-06-13-RTV3D-FROZEN-LASER-POSES-frozen-rig-laserline-poses.md).
  `RigLaserlineDevice` now prefers upstream `rig_se3_target` by view token and
  applies upstream robot deltas in the legacy chain fallback, fixing the
  coherent 54 px reprojection drift seen in the rtv3d laser preset.
- [x] RTV3D-JOINT-LASERLINE-APP - Add app/pipeline joint rig hand-eye
  laserline topology. Completed 2026-06-13 —
  [report](report/2026-06-13-RTV3D-JOINT-LASERLINE-APP-joint-app-topology.md).
  The rtv3d laser preset now runs `RigHandeye -> RigLaserlineDevice ->
  optimize_rig_handeye_laserline` directly from `dataset_laser.toml`.
  The shipped preset fixes `cx/cy` in joint BA to avoid the nonphysical
  principal-point valley observed in the all-20-pose app run.
- [x] RTV3D-LASER-CUTS - Render active-pose target-plane intersections for
  the six laser planes in the 3D viewer. Completed 2026-06-13 —
  [report](report/2026-06-13-RTV3D-LASER-CUTS-viewer-laser-cuts.md).
  The viewer now draws clipped colored line segments on the selected board,
  while the translucent full laser-plane quads remain optional.
- [x] RTV3D-INTRINSICS-FOCUS - Isolate per-camera rtv3d Scheimpflug intrinsic
  calibration and repair Scheimpflug-aware intrinsic-floor reporting.
  Completed 2026-06-14 —
  [report](report/2026-06-14-RTV3D-INTRINSICS-FOCUS-scheimpflug-intrinsics.md).
  `calib-bench diagnose intrinsics` now runs target detection only, then a
  staged/multistart Scheimpflug intrinsic solve with centered `cx/cy`,
  `p1/p2/k3` fixed, and diagnostic-only model variants. Threshold 30 improves
  the rtv3d floor, but all six cameras still fail the raw `<0.4 px` gate
  (best centered means: 0.747–1.199 px), pointing to detector/target/model
  floor rather than rig-chain error.
- [ ] V6-SCALE - Settle rtv3d absolute scale: get the mechanical camera
  spacing of the head (our hexagon: 90.1 mm at 5.2 mm cells; oracle implies
  ~98.5 mm). If 98.5 mm is right the true cell is ≈5.69 mm and both shipped
  board specs are wrong.
- [ ] V7-RTV3D-INTRINSICS-FLOOR - Drive the rtv3d reprojection floor below
  0.4 px or prove the blocking model/data term. Follow up from
  RTV3D-INTRINSICS-FOCUS: inspect ChArUco corner localization quality and
  target/print/blur effects first, then test richer lens terms (M2 thin-prism
  or rational) only if detector residual vector fields remain structured after
  cleaning detections.

## O — apex-solver backend (O1/O2 PARKED 2026-06-15; O3 DONE)

Pre-verify failed: apex-solver 1.3 is a hand-Jacobian factor-graph library, not
autodiff-capable — fundamentally mismatched to our autodiff-first IR (ADR 0008 /
M0 ADR 0020). Full findings:
`docs/report/2026-06-14-O1-apex-solver-preverify.md`. Reviving Track O is a user
call (pick an autodiff-capable optimizer, or keep tiny-solver as the sole
backend).

- [~] O1-BACKEND - **PARKED.** `ApexSolverBackend` blocked on the autodiff API
  mismatch (no generic scalar / dual numbers; `Factor::linearize` takes a
  caller-supplied Jacobian). Also missing: S2 manifold (we use one for
  laser-plane normals), documented robust losses, documented SE3 quaternion
  order. A numeric-difference bridge is possible but slower / less accurate and
  needs hand-derived manifold Jacobians — not recommended unsupervised.
- [~] O2-AB - **PARKED** (depends on O1).
- [x] O3-CERES - **DONE 2026-06-15.** Removed the `BackendKind::Ceres` stub from
  `optim/src/backend/mod.rs` plus the now-orphaned `Error::numerical` helper;
  `BackendKind` is a single-variant enum and the dispatch has no unreachable
  arm. Report: `docs/report/2026-06-15-O3-CERES-drop-ceres-stub.md`.

## P — Performance & profiling

Opened 2026-06-16 after the from-scratch Scheimpflug rig calibration
(`rtv3d_ref_rig`) took 30+ min on the dense `puzzle_board` dataset (~200
corners/view). Root-caused to dense linear-algebra hot paths, not the
algorithms. Full profiling:
`docs/report/2026-06-16-perf-from-scratch-rig-profiling.md`.

Systemic causes:
1. `nalgebra::svd(true, true)` is used pervasively (~20 sites) for both
   null-space extraction and least-squares. On a tall/dense design matrix it
   accumulates the U factor across thousands of rows — pathologically slow
   (the homography DLT hung >15 min; the distortion fit hung >11 min).
2. The `tiny-solver` backend recomputes an autodiff (dual-number) Jacobian over
   every residual each LM iteration, re-evaluates all residuals on each of up to
   32 damping retries, and is single-threaded.
3. No data-density control for the joint rig/hand-eye BA — per-iteration cost
   scales linearly with corner count, but extrinsics/hand-eye do not need full
   corner density.

- [x] P1-SVD-SWEEP - Replace `svd(true, true)` on large matrices with a method
  that cannot hang on nalgebra's unbounded QR iteration: `AᵀA` + symmetric-eigen
  (null-space) or ridge-regularized normal equations / QR (least-squares).
  Centralize behind `math::null_space` / `ridge_lstsq` / `project_to_so3`
  helpers; guard non-finite inputs and reject geometrically-bad results.
  **First pass 2026-06-16:** homography DLT and distortion fit (the two confirmed
  hangs) + view-tolerant iterative init. **Sweep finished 2026-06-16** —
  [report](report/2026-06-16-P1-SVD-SWEEP-finish-centralize.md): the 7 remaining
  hang-risk null-space sites (`camera_matrix` + `pnp/dlt` + `pnp/epnp` +
  `epipolar/{fundamental,essential}` across `linear` and `vision-geometry`,
  including vision-geometry's insufficient `svd(false,true)` homography) and the
  3 moderate sites (`handeye` rotation + `ridge_llsq`, `zhang_intrinsics`) now
  route through `math::null_space` / `ridge_lstsq`; `project_to_so3` deduped
  across 5 sites. All linear/geometry/mvg + downstream optim/pipeline (golden
  pins) tests green; clippy + doc clean. **Closed 2026-06-17:** the last
  `triangulation.rs` `2N×4` site now routes through `core::linalg::null_space`
  (done as part of C2); the shared-helper de-dup landed via C1-FOLLOWUP.
- [ ] P2-BA-DENSITY - Principled corner budget for the joint rig + hand-eye BA
  (spatially-distributed subsample preserving coverage, or per-stage decimation
  knobs). Extrinsics/hand-eye converge on a fraction of the corners; the
  per-camera intrinsics stage already uses a cheap subsampled tilt sweep + a
  full-data refine (`optimize_scheimpflug_intrinsics_staged`).
- [ ] P3-BACKEND-COST - Profile the tiny-solver split (autodiff Jacobian vs
  `JᵀJ` assembly vs linear solve vs per-retry residual re-eval). Evaluate
  analytic Jacobians for the hot `ReprojPoint` factor, Jacobian/residual caching,
  and parallel (rayon) residual + Jacobian evaluation.
- [x] P4-CRITERION - criterion benchmarks for the hot paths to guard against
  regressions and quantify P1–P3 gains. **Done 2026-06-16** —
  [report](report/2026-06-16-P4-CRITERION-hot-path-benches.md): `criterion`
  workspace dev-dep + `[[bench]]` targets; `linear/benches/linear_init.rs`
  (homography DLT 225pts ~8.6µs — was a >15min hang, zhang-from-homographies,
  distortion fit) and `optim/benches/ba_iter.rs` (one per-camera planar BA solve
  ~16.9ms). Deterministic synthetic data; `cargo bench --no-run` is the
  CI-friendly guard. The joint rig/hand-eye-BA iteration bench needs the rig
  fixtures and is deferred (per-camera BA is the proxy until then).
- [x] P5-STAGE-TIMING - Per-stage timing instrumentation so future regressions
  surface without ad-hoc harnesses. **Done 2026-06-16** —
  [report](report/2026-06-16-P5-STAGE-TIMING-bench-per-stage.md): additive
  `StageTiming` (6 optional per-stage `*_ms` fields) on the bench `Timing` struct
  (`serde(default)` + `skip_serializing_if` → back-compat with v3 records, no
  schema bump); `run_rig_extrinsics` (3 stages) and `run_rig_handeye` (5 stages)
  now time each optimize sub-stage instead of lumping them into `optimize_ms`,
  via a reusable `ms_since` helper. Serde back-compat/roundtrip test added.
- [x] P6-PERCAM-CONVERGENCE - From-scratch Scheimpflug per-camera convergence
  on the private `rtv3d_ref` rig. **Done 2026-06-16** —
  [report](report/2026-06-16-P6-PERCAM-CONVERGENCE-tilt-aware-init.md);
  supersedes the diagnosis
  [report](report/2026-06-16-P6-PERCAM-CONVERGENCE-diagnosis.md). Implemented a
  tilt-aware linear Scheimpflug initializer plus a rig-handeye auto-recovery pass
  that forms a shared nominal seed from good cameras and retries bad cameras
  against good-camera rig poses. Private validation:
  `rtv3d_ref_rig` from scratch reaches per-camera intrinsics BA reprojection
  `[0.3802, 0.2677, 0.2833, 0.3522, 0.4725, 0.3268]`, final mean reprojection
  `0.4057px`, and all `tau_x` values stay within about `1.5°` of oracle. Remaining
  risks: the shared-nominal retry is currently private to `rig_handeye` rather
  than factored into `rig_family` for `rig_extrinsics`, and the final joint
  hand-eye per-camera reprojection still has cam 0 at ~`0.528px` despite
  sub-`0.5px` intrinsics solves.
- [x] P7-SCHEIMPFLUG-SEEDED-DEFAULT - Make **user-seeded** Scheimpflug *intrinsics*
  the supported default and demote from-scratch to experimental (ADR 0022).
  **Done 2026-06-17.** The seeded path now (a) trusts a user-provided mount-tilt
  seed instead of the cold multi-start sweep, (b) frees the (non-existent) pose
  gauge that previously pinned a distortion-biased homography pose off the optimum,
  and (c) escapes the spurious `k1≈0` local minimum via a `k1` multi-start with
  tilt fixed, then a bounded joint refine. New private harness `rtv3d_ref_intrinsics`
  calibrates all 6 `rtv3d_ref` cameras from one coarse shared seed
  (`fx=fy=1150, pp=(360,270), tilt_x=−0.087, distortion=0`) and **all pass the hard
  ≤ 0.5 px gate** (`[0.373, 0.267, 0.282, 0.473, 0.342, 0.321]` px; cam 3 is the
  tightest). Public CI guard: synthetic
  `seeded_coarse_prior_converges_on_strong_tilt_distortion`. From-scratch
  `step_init` now logs an experimental warning. **A reprojection error > 0.5 px is
  never accepted as success** — the harness exits non-zero on any miss.

## M — camera models (gated on M0)

- [x] M0-GENERIFY - Factor generification (F1). Completed 2026-06-12
  (ADR 0020). FactorKind = 4 families (ReprojPoint, LaserPointToPlane,
  LaserLineDistance, Se3TangentPrior) with CameraModelDesc + chain as data;
  layout-derived validation; ZST-kernel monomorphization via one
  dispatch_camera_model! table; net ~-1.9k LoC in optim. Also fixed F3
  (export path now uses the generic helper) and F4 (pinhole rig laserline
  upstream accepted, end-to-end test). Numerics bit-identical on all
  production paths (golden-value pins). Deferred follow-up: collapse the
  near-duplicate problem-builder pairs (`handeye`/`handeye_scheimpflug`,
  `rig_extrinsics`/`rig_extrinsics_scheimpflug`) into one builder
  parameterized by `CameraModelDesc` — out of M0 scope.
- [x] M1-RATIONAL / M2-THINPRISM / M3-DIVISION — **additive layer done
  2026-06-14** —
  [report](report/2026-06-14-M-distortion-models.md). `RationalPolynomial`
  (k1–k6,p1,p2), `ThinPrism` (BC5+s1–s4), and `Division` (Fitzgibbon `lambda`,
  closed-form inverse) added at the core runtime model + optim IR/backend
  layers: new `DistortionParams`/`AnyDistortion` variants, `DistortionKind`
  variants, `CameraModelDesc::PINHOLE4_{RATIONAL8,THINPRISM9,DIVISION1}`
  (+`_SCHEIMPFLUG2`), ZST kernels, 6 dispatch rows, synthetic-GT + roundtrip
  tests. **Strictly additive** — no pipeline/builder/export change, BC5
  production paths byte-identical. Usable via `CameraParams` and hand-built
  `ProblemIR`.
- [~] M-WIRE - Pipeline-selection plumbing for the new distortion models.
  **PlanarIntrinsics vertical slice DONE 2026-06-21** (user-supervised; chose
  vertical slice + "Zhang shared terms, zero extras, refine" init + a
  model-agnostic export contract). `PlanarIntrinsicsConfig.distortion_model:
  DistortionKind` (Serde, `#[serde(default)]` → BC5) selects BC5 / Rational8 /
  ThinPrism9 / Division1. `PlanarIntrinsicsParams.camera` is now the serializable
  `CameraParams` (was the concrete `PinholeCamera`) — `build_camera()` →
  `CameraModel` for residuals (generic via `CameraProject`), `from_pinhole()` /
  `pinhole_camera()` (Result; `Err` for extended models) bridge the rig family
  which stays BC5. Model-aware `pack/unpack_distortion_params` (variable-dim,
  IR-order-exact); the IR builder maps the kind → `CameraModelDesc::PINHOLE4_*`;
  init embeds the BC5 linear seed into the target model and zeros the extras.
  7 new E2E tests (per-model noiseless round-trip — Rational8 4.2e-6 px,
  ThinPrism9 2.9e-4 px, Division1 3.2e-6 px from λ=0, BC5 3.0e-3 px regression
  + JSON roundtrip + back-compat default). The export `params.camera` JSON shape
  changed to the tagged `CameraParams` form (app/diagnose deserialization updates
  when the app selector lands).
  - [ ] **Remaining (supervised):** the other intrinsics-bearing problem types
    (`ScheimpflugIntrinsics`, rig family), the app Run-form model selector +
    Diagnose display, Python binding, and manual-init seeding of extended models
    (`PlanarManualInit::distortion` is still `Option<BrownConrady5>`). Also: the
    BC5-shaped `DistortionFixMask` is currently ignored for non-BC5 models (no
    user-visible error) — generalize or warn.
  - **Robust wide-FOV inverse** (sub-item): the rational/thin-prism runtime +
    kernel `undistort` use a radial-division fixed point that is contracting
    only within the calibrated FOV (radius ≲ ~1.2), matching OpenCV
    `undistortPoints`; it oscillates for extreme wide-FOV inputs (codex P2 on
    PR #61). Replace with a Newton / 1D-radial solve when these models are wired
    (their required FOV + tangential handling become concrete then).
- [ ] M4-FISHEYE - Kannala-Brandt equidistant k1–k4: new `ProjectionModel`
  impl (first beyond `Pinhole`), linear-init changes (Zhang assumptions
  don't hold at large FOV), synthetic wide-FOV tests.

## C — MVG (multiple-view geometry)

- [x] C1-CRATES - Land `vision-geometry` + `vision-mvg`. Completed 2026-06-14 —
  [report](report/2026-06-14-C1-mvg-crates.md). Ported fresh and additively from
  the stale `mvg` branch source (NOT a branch merge): `vision-geometry`
  (deterministic solvers: epipolar/homography/triangulation/camera-matrix, 20
  tests) + `vision-mvg` (pipelines/robust/pose-recovery, optional `refine`
  feature, 31 tests), both `publish = false`. `vision-calibration-linear`
  untouched (zero regression). [ADR 0015](adrs/0015-mvg-ceiling.md) caps the
  ceiling (no dense matcher, no full SfM).
- [~] C1-FOLLOWUP - De-duplicate the geometric solvers (have the calibration
  and MVG crates share one source of truth instead of parallel copies).
  **Partially done 2026-06-17** — stale `mvg` branch / PR #28 and the other
  merged feature branches pruned; the shared **low-level `math` primitives** are
  now deduped onto the shared foundation crate. Rather than make the published
  `vision-calibration-linear` depend on the (private) `vision-geometry` for a
  handful of numeric helpers — a layering smell that would also force geometry's
  publication — the primitives moved **down into `vision-calibration-core`** as a
  new `vision_calibration_core::linalg` module (with a typed `MathError` replacing
  stringly `anyhow`, advancing D1): `normalize_points_2d/3d`,
  `solve_{quadratic,cubic,quartic}_real`, `null_space` + `NullSpaceSolution`,
  `mat3/mat34_{from_vec,from_svd_row}`. **Both** `linear::math` and
  `geometry::math` now re-export from `core::linalg`; `geometry` keeps only its
  DLT-specific `dlt_rank_ok`. No new cross-crate dependency edge; full suite green
  (zero drift). The linear-only helpers `ridge_lstsq` / `project_to_so3` stay in
  `linear`.
  - [ ] **Remaining (supervised):** the higher-level solvers
    (`homography`, `epipolar/{fundamental,essential,decomposition}`,
    `camera_matrix`, `triangulation`) are still duplicated between `linear` and
    `geometry`. The two implementations have **diverged** (geometry was a fresh
    port — richer essential/RANSAC, different normalization/error paths), so
    substituting one for the other risks **calibration numeric drift**. Reconcile
    numeric equivalence (golden pins on both sides as the gate) before
    collapsing — not a safe unsupervised swap. Whether the shared home is
    `core`, `geometry`, or a new crate is part of that design.
  - [x] Promote `vision-geometry` / `vision-mvg` to the crates.io publish set.
    **Done 2026-06-17** (user call): `publish = false` removed from both; the
    `[workspace.dependencies]` version pins were already in place; release
    version-lockstep doc updated to nine publishable crates with the publish
    order. The actual first `cargo publish` is a manual step the user drives.
  - [ ] PyO3 bindings for the MVG surface — **deferred** (A5 Python parity was
    dropped: no Python consumer, and the py crate binds the calibration facade
    only). Revisit if a consumer appears.
- [x] C2-TRIANGULATION - N-view triangulation + nonlinear refinement. **Done
  2026-06-17.** `vision-geometry` already had N-view linear DLT; added
  `triangulate_point` (linear init + self-contained Gauss-Newton reprojection
  refinement, no external solver) and `refine_point`, and migrated
  `triangulate_point_linear` off the raw `svd(true,true)` onto `core::linalg::null_space`
  — closing the last P1 SVD-hang leftover (the `triangulation.rs` `2N×4` site).
  `vision-mvg` gains `triangulate_nview` (refined N-view + RMS reprojection,
  widest-baseline parallax, all-views cheirality diagnostics). Synthetic-GT
  tests: 4-view noiseless recovery, refinement-improves-noisy-estimate,
  degenerate-camera safety, count-mismatch guard.
- [x] C3-BA - Bundle adjustment with frozen intrinsics, free poses, free
  structure. **Done 2026-06-21** (PR #73). New
  `vision-mvg::bundle_adjust` (behind `refine`): tiny-solver LM with an analytic
  reprojection `Factor<T>`, SE3 pose blocks on `SE3Manifold` + 3D point blocks,
  per-camera intrinsics baked into each factor (frozen). `fix_first_camera`
  (default) anchors the gauge on the first *observed* camera (no-manifold + fix
  all 7 raw indices → 0 columns), removing the 6-DOF rigid gauge; global scale
  is documented as an inherent 1-DOF gauge that free-structure reprojection
  cannot observe. Returns refined poses/points + initial/final/per-camera RMS;
  unobserved cameras/points pass through. 9 synthetic-GT tests (recovery up to
  scale, perfect-init, pixel-noise, no-gauge-fix, unobserved-camera-0 anchor,
  three input-validation guards).
- [x] C4-RECTIFY - Scheimpflug-aware stereo rectification (the D4 gate).
  **Done 2026-06-21** (PR #74). New `vision-mvg::rectification`:
  `rectify_stereo_pair(left, right, cam1_se3_cam0, opts) -> StereoRectification`.
  Because a pixel is `K·H_tilt·x_n`, the sensor tilt is a homography on the
  normalized plane; pre-multiplying each camera's unprojection by `H_tilt⁻¹`
  collapses a Scheimpflug camera to a frontal pinhole, after which standard
  Fusiello/Bouguet applies (`H_left = K_rect·R_rect·H_tilt0⁻¹·K0⁻¹`,
  `H_right = K_rect·(R_rect·Rᵀ)·H_tilt1⁻¹·K1⁻¹`). Zero tilt reduces exactly to
  pinhole rectification; inputs are undistorted pixels (distortion handled
  separately, as OpenCV splits `initUndistortRectifyMap`). 6 synthetic tests
  through the real core Scheimpflug model (rows align <1e-6). D4 gate closed by
  the `rtv3d_ref_rectify` example: worst rectified row disagreement 3.4e-13 px
  across all oracle camera pairs (real K, asymmetric per-cam ~-5° tilts, ring
  extrinsics). Also repointed a pre-existing #72 regression in examples-private
  (`rtv3d_ref_reproj` imported the removed `linear::homography`).
- [x] C-FACADE-MVG - Re-export the `vision-mvg` surface through the
  `vision-calibration` facade. **Done 2026-06-21**. New
  `vision_calibration::mvg` module (mirrors the existing `geometry` module
  style) re-exporting `pose_recovery`, `robust`, `cheirality`, `degeneracy`,
  `triangulation`, `homography`, `rectification`, `residuals`, `types`, `error`
  + `MvgError`. Frozen-intrinsics `bundle_adjust` is surfaced behind a new
  facade `refine` feature (`refine = ["vision-mvg/refine"]`). Closes the gap
  that the MVG pipelines (C2 triangulation, C3 BA, C4 rectification) were
  reachable only via a direct `vision-mvg` dependency; also unblocks future
  Python parity for the MVG surface. Surface locked by
  `tests/facade_compile_surface.rs` (default + `refine`).
- [x] C-MVG-TUTORIAL - Tutorial + runnable example for the MVG surface.
  **Done 2026-06-21**. `docs/tutorials/multiple-view-geometry.md` (pose recovery
  → triangulation → bundle adjustment → Scheimpflug-aware rectification via the
  facade `mvg` module) with the runnable companion
  `crates/vision-calibration/examples/mvg_two_view.rs` — a synthetic-GT
  end-to-end demo (pose recovery exact, BA 18→0.03 px, rectification rows align
  to ~1e-13 px). BA section is `#[cfg(feature = "refine")]`-gated so the example
  builds in both configs. Tutorials README index updated. Closes the
  ship-a-tutorial-with-new-features gap for C2/C3/C4.
- [~] C5-DENSE - Dense stereo matcher. **Direction reset + harness DONE
  2026-06-21** (user-supervised). ADR 0015's original ceiling ("no in-house
  matcher; wrap `opencv-rust` SGBM behind a feature flag") is **amended**: the
  matcher ships **pure-Rust in `vision-mvg`**; `opencv-rust` SGBM is a
  **benchmark-only** quality baseline confined to the unpublished
  `vision-calibration-bench` crate (never a published crate, never the shipped
  impl). Validation uses existing calibration-target data — a C4-rectified pair
  has known target-plane depth.
  - [x] **Benchmark harness** (`vision-calibration-bench::dense`): `DenseMatcher`
    trait, `GrayBuffer` / `DisparityMap`, deterministic `synthetic_rectified_pair`
    (slanted-plane GT, right = bilinear warp of left), `evaluate` metrics
    (RMS / MAE / bad-pixel-rate / density), `OracleMatcher`. 8 tests; zero new
    deps; `--all-features` workspace stays green (no OpenCV feature added — would
    break the `--all-features` gate without a system OpenCV).
  - [ ] **OpenCV SGBM baseline** — needs an OpenCV-equipped environment to add +
    verify (not installable in the authoring env; would break `--all-features`
    if added as a normal cargo feature → put it in a workspace-EXCLUDED crate or
    a dedicated CI job with OpenCV). Implements `DenseMatcher`, scored by
    `evaluate`.
  - [x] **Pure-Rust matcher** — the block-matching MVP. **Done 2026-06-21.** New
    `vision_mvg::dense` (always-on, no new deps): `GrayImage` / `DisparityMap` /
    `BlockMatchOptions` + `match_block` — ZNCC over a square window aggregated in
    `O(1)/px` via summed-area tables (so the search is `O(W·H·D)` for any block
    size), winner-take-all with parabolic sub-pixel refinement, and three
    independent invalidation filters (min-correlation, uniqueness margin,
    left-right consistency). Typed `MvgError`; surfaced through the facade as
    `vision_calibration::mvg::dense` (surface-locked). The bench `BlockMatcher`
    (`impl DenseMatcher`, reached via the facade — no new bench dep) scores it
    through the harness: synthetic slanted-plane recovery hits **94% density at
    0.18 px RMS**. Two visual-evidence demos write inspectable PNGs to
    `target/fixtures/`: `dense_synth` (bench, `--features tier-b`) tiles
    left | right | GT | estimate | error; `dense_stereo_real` (facade) rectifies
    the committed `data/stereo` chessboard rig (undistort → C4 rectify → match)
    and recovers the board plane at **0.44 px planarity-fit RMS** over ~13.5k
    inlier pixels (no GT needed — the planar target is the reference). 8 matcher
    unit tests + 1 harness integration gate.
  - [x] **Semi-global (SGM) aggregation** — **Done 2026-06-21.** Opt-in
    `BlockMatchOptions::semi_global` (+ `sgm_p1`/`sgm_p2`): the raw ZNCC cost is
    aggregated along 8 paths with Hirschmüller `P1`/`P2` smoothness penalties
    before disparity selection, propagating disparity into low-texture regions
    that block matching leaves blank. Strictly additive — the raw cost volume and
    block-mode behaviour are unchanged (block stays the default). On the real
    `data/stereo` rig SGM **doubles board coverage (21% → 49% density)** —
    `dense_stereo_real` now writes both `disparity_block.png` (grid only) and
    `disparity.png` (filled board) for the before/after. 1 new unit test
    (stays accurate + never recovers fewer pixels than block). Remaining C5 work:
    the OpenCV SGBM baseline above (needs an OpenCV-equipped env).

- [x] C-UI-DEPTH - Dense-matching **Depth workspace** in the Tauri app (first
  C-UI slice). **Done 2026-06-21.** New server-side `compute_disparity` Tauri
  command (`app/src-tauri/src/disparity.rs`): reads the loaded rig export
  (cameras `k`/`dist`, `cam_se3_rig`), composes `T_Cb_Ca`, rectifies the chosen
  synchronized pair (C4), undistort+rectify remaps, dense-matches
  (`mvg::dense::match_block`, block or SGM), and returns rectified-pair /
  disparity / overlay PNG data URLs + metrics (density, disparity range,
  robust planarity RMS, baseline). New React `DepthWorkspace` (pose stepper,
  left/right camera selectors, SGM toggle, Compute button, view-mode switch,
  metrics strip) + `/depth` route + rail nav. End-to-end Rust test on the
  committed `data/stereo` rig (board recovered, valid PNGs); `bun run build` +
  18 vitest tests green; app `cargo fmt`/clippy clean. Matches at downscale 4 for
  dev responsiveness. Remaining C-UI: point clouds (triangulation), depth maps.

## D — Earn v1.0

- [x] D2-DOCS - `missing_docs = warn` enforced workspace-wide + all public items
  documented (PR #69).
- [x] D1-TYPED-ERRORS - Drop `anyhow` from the published library crates' public
  surfaces onto `thiserror` enums. **Done across PR #72 (geometry/mvg) + PR-1
  (optim) + PR-2 (detect/pipeline).** `vision_calibration_core::linalg` carries a
  typed `MathError` (C1-FOLLOWUP); `vision-calibration-linear` bridges geometry via
  `#[from]`. No `vision-calibration*` library crate carries `anyhow` in its
  `[dependencies]` any more (only `[dev-dependencies]` for tests/doctests).
  - [x] **optim** (PR-1). Converted every internal `anyhow!` / `ensure!` /
    `AnyhowResult` straggler to the existing typed `crate::Error`
    (`invalid_input` for structural/precondition checks, a new `pub(crate)
    numerical()` constructor for post-solve / decode failures), retyped the
    *private* `OptimBackend::solve` (zero external blast radius), deleted the
    `impl From<anyhow::Error> for Error` escape hatch, and dropped the `anyhow`
    dependency entirely. 45 optim tests green, full workspace builds, zero
    numeric drift (mechanical type change only).
  - [x] **detect + pipeline** (PR-2). detect gained a typed `DetectError`
    (`Config { detector, serde_json::Error }` + `InvalidConfig(String)`); the
    *sealed* `Detector::detect_json` + the public `validate_*` fns retyped (no
    downstream blast radius). pipeline: `RunError::Detection.source` is now the
    typed `DetectError`; only the *open* `LaserPixelExtractor::extract` trait +
    `RunError::LaserExtraction.source` move onto `Box<dyn Error + Send + Sync>`
    (std-only, preserves the opaque injected source — no `anyhow` in the public
    type). `planar_family`'s `anyhow::Context` chains → `Error::numerical`; the
    `From<anyhow::Error>` escape hatch removed. The app's `VmLaserExtractor` impl
    + the laser-manifest doc example updated to the new trait signature; `core` /
    facade / pipeline `anyhow` moved to `[dev-dependencies]`. Caught a key
    asymmetry: a *sealed* extension trait can be fully typed, an *open* one needs
    `Box<dyn Error>`.
  - **NaN-rejection regression (PR-1 review, codex P2):** the `ensure!(x > 0.0)`
    → `if x <= 0.0` rewrite silently accepted NaN (`NaN <= 0.0` is false). Fixed
    across 10 sites with `x.is_nan() || x <= 0.0` (the `!(x > 0.0)` form trips
    `clippy::neg_cmp_op_on_partial_ord`); added a `validate_rejects_nan_bound`
    regression test. Lesson: convert `ensure!(c)` as `if !c`, never by
    hand-negating a float comparison operator.
- [~] D3-PY-PARITY - Audit the PyO3 binding surface against the Rust facade
  (incl. the new `mvg` surface); fill gaps; add parity tests.
  - [x] **Audit DONE 2026-06-21** — [`docs/python-parity-audit.md`]. Findings:
    **seven of the eight** facade calibration workflows + `robust_*` /
    `pixel_to_gripper_point` / `library_version` are bound (JSON/`pythonize`
    style — serde across the boundary, dataclass wrappers). Gaps: **G0** the
    EIGHTH workflow `rig_handeye_laserline` (`RigHandeyeLaserlineProblem`,
    facade lib.rs:446) has NO `run_rig_handeye_laserline` binding — cheap (same
    `run_problem::<P>` pattern), highest priority (caught by codex on the audit
    PR — my first draft mis-counted seven as "all eight"); **G1** the entire MVG
    surface (`geometry` + `mvg`: pose recovery, N-view triangulation,
    rectification, robust, bundle adjust) is Rust-only — medium effort (MVG API
    uses raw nalgebra types → needs serde DTOs per entry point; `bundle_adjust`
    is `refine`-gated); **G2** the M-WIRE `distortion_model` config field isn't
    in the Python wrapper (cheap; was Rust-core-only by scope); **G3** low-level
    modules unbound, most by design. No binding-coverage test exists.
  - [ ] **Fill (sequenced):** G0 `run_rig_handeye_laserline` (cheap, completes
    the workflow surface) → G2 distortion-model field (cheap) → G1 MVG bindings
    (DTOs → triangulation + rectification + pose recovery → robust → BA, + `.pyi`
    + round-trip parity tests) → a binding-coverage parity test (the guard that
    would have caught G0). G3 deferred pending a consumer.
- [ ] D4-RELEASE - v1.0 gate: puzzle rig green via app + C4 landed (done) + API
  stable across two minor releases.

## B — app (extend; sequencing serves V-track)

- [x] B3C-PUZZLEBOARD - PuzzleBoard detector. Completed 2026-06-14 —
  [report](report/2026-06-14-B3C-PUZZLEBOARD-puzzleboard-detector.md).
  `PuzzleboardDetector` wraps `calib-targets` `detect_puzzleboard` behind the
  sealed `Detector` trait + `puzzleboard` feature; `dataset_runner` resolves
  the `"puzzle_<R>x<C>"` layout name to dimensions and dispatches it (was
  `UnsupportedTarget`). Synthetic-board detection + dispatch tests green.
- [x] B3C-RINGGRID - Coded ring-grid detector. Completed 2026-06-14 —
  [report](report/2026-06-14-B3C-RINGGRID-ringgrid-detector.md).
  `RinggridDetector` wraps `ringgrid` 0.6 (crates.io) behind the sealed
  `Detector` trait + `ringgrid` feature; `dataset_runner` dispatches
  `TargetSpec::Ringgrid`. `TargetSpec::Ringgrid` realigned (breaking) to the
  real hex-lattice `BoardLayout` model (`pitch`/`rows`/`long_row_cols`/radii/
  ring-width); schema regenerated. All four target detectors now calibrate
  end-to-end. Synthetic-board detection + dispatch tests green.
- [ ] B3C-CHARUCO-DEDUP - **Deferred 2026-06-14** (blocked on verification).
  Consolidate the three charuco detection paths (`detect/src/charuco.rs`
  canonical, `examples-private::detect_charuco`, `bench::detect_charuco_view`)
  onto one shared detection call. The clean design (bench/examples delegate to
  the detect crate and adapt `Vec<Feature>` → `CorrespondenceView`) is known,
  but its gate — bench + examples residual numbers unchanged — cannot be
  verified without the private golden datasets (`/data/*`, registry
  `private.json`), which are absent from CI and dev checkouts. Resume on a
  machine with the private data, or after a committed charuco fixture exists.
  `detect/src/charuco.rs` now documents the canonical-authority boundary.
- [x] B3C-RIG - Run-workspace coverage for `RigExtrinsics`, `RigHandeye`,
  `RigLaserlineDevice` + charuco detector wiring. **Already shipped** in
  B3c-1 + B3c-3 (2026-06-12); checkbox was stale. Verified 2026-06-14:
  `app/src-tauri/src/run.rs` dispatches all 8 topologies via an exhaustive
  `match` (no stub), `dataset_runner` exposes all seven `build_*_input`
  converters + all four detectors, and `RunWorkspace/topologies.ts` exposes
  every topology with `supported: true`. No code change needed.
- [x] B3D-SNIFF - Heuristic dataset folder → `DatasetSpec` sniffer (B3d-1).
  Completed 2026-06-14 —
  [report](report/2026-06-14-B3D-SNIFF-heuristic-manifest-sniffer.md).
  `vision_calibration_dataset::sniff_folder` walks a dataset directory and
  infers only structurally-unambiguous fields (camera dirs/globs, robot-pose
  file format, `by_index` pairing), leaving board geometry / target kind /
  frame convention / ambiguous topology at placeholders with their dotted
  paths in `_unresolved` (ADR 0019). New `generate-manifest` CLI (`cli`
  feature → TOML) and Tauri `sniff_folder` command share the one inference.
  Acceptance: round-trips `data/kuka_1` (single_cam_handeye + rowmajor4x4
  poses) and `data/stereo` (rig_extrinsics, topology flagged).
- [x] B3D-UX - Frontend manifest UX (B3d-2). Completed 2026-06-14 —
  [report](report/2026-06-14-B3D-UX-manifest-sniff-unresolved-askuser.md).
  "Sniff folder" button (calls the `sniff_folder` command), `UnresolvedNotice`
  strip with vendor-aware field hints + per-field "mark resolved", red
  `N unresolved` badge on the Manifest section (new `badgeVariant`), Run
  blocked while `_unresolved` non-empty, and `AskUserModal` replacing the
  inline AskUser banner (click-to-apply suggestion buttons + free-text).
  Vendor hints live front-end-side (`FIELD_HINTS`) so runner suggestions stay
  raw click-to-apply values; no pipeline change. `bun run build` + `tsc -b`
  green.
- [ ] B-LASER - Laserline visualization: laser-pixel overlay in Diagnose,
  point-to-plane residuals (mm) panel, laser planes in the 3D rig viewer.
- [ ] B-EXPLORE - Pre-calibration dataset exploration: per-camera/pose image
  grid, detection-cache overlay, board coverage map.
- [ ] B-INFRA - ts-rs (or specta) codegen for wire types + export
  discriminator tag (F6); `resource_dir` presets
  (`RunWorkspace/presets.ts:16`); Vitest unit + Playwright smoke tests.
  - [x] Vitest unit slice (2026-06-15) - Vitest scaffold (`vitest@4`,
    `vitest.config.ts` node env, `test` / `test:watch` scripts) + pure-logic
    coverage of `inferExportKind` (every probe branch + probe-order precedence),
    `exportKindLabel` (exhaustive), and `mergeConfig` (deep-merge / array-replace
    / type-disagreement / null-override / base-immutability). 18 tests. Safe
    subset — no wire change. Report:
    `docs/report/2026-06-15-B-INFRA-vitest-unit-slice.md`.
  - [ ] ts-rs/specta codegen + export discriminator tag (F6) - DEFERRED: a
    risky wire change that replaces the shape-probe in `inferExportKind` with a
    Rust-emitted tag; needs supervised design (which exports gain the tag, how
    `AnyExport` narrows). Tracked in the report's follow-ups.
  - [ ] `resource_dir` preset resolution (`RunWorkspace/presets.ts:16`) -
    replace the hard-coded `REPO_ROOT` with Tauri bundle-asset resolution.
  - [ ] Playwright smoke tests - needs a Tauri/webview harness; out of the
    pure-logic Vitest scope.

## Benchmark

- [x] BENCH-W2C - Compact benchmark reports, puzzle rig wiring, laser extraction, and hand-eye diagnostics.
  Completed 2026-05-31. This task keeps compact JSON records at schema v3,
  writes full residuals only through sidecars, wires the private 130x130 puzzle
  rig and optional laser extraction, and adds deterministic hand-eye diagnostic
  sweeps. Puzzle calibration remains an open benchmark finding: the unseeded
  validation run was stopped after roughly 12 minutes with no solve output.
- [x] BENCH-W2D - Interactive benchmark dashboard, robot-correction visibility, and stage profiling.
  Completed 2026-05-31. Adds compact BenchRecord dashboard mode to the
  calibration viewer, reports robot-pose correction magnitudes in mm/degrees,
  exposes `diagnose stages` for target/laser timing, and records current
  hand-eye and private ChArUco quality findings.
- [x] BENCH-W2E - Fix DS8 hand-eye mode and known-grid checkerboard handling.
  Completed 2026-05-31, superseded by BENCH-W2F on the mode interpretation.
  Confirms DS8 uses a 10x14 checkerboard with 52 mm cells, rejects partial /
  local-grid checkerboard detections for this dataset, and extends hand-eye
  diagnostics with alternate-mode comparison.
- [x] BENCH-W2F - Benchmark viewer script, progress, artifacts, topological chessboard, and DS8 pose convention.
  Completed 2026-05-31. Adds `scripts/bench-viewer.sh`, stderr progress during
  dataset runs, calibration artifact output for the dashboard, topological
  chessboard dispatch for simple checkerboards, and corrects DS8 to physical
  EyeInHand with `gripper_se3_base` robot-pose convention.
- [x] BENCH-W2G - Fix viewer temp output path and validate KUKA topological chessboard run.
  Completed 2026-05-31. Writes script-generated benchmark JSON to `/tmp` so
  Vite can serve it through `/@fs`, and verifies `kuka_1` succeeds through the
  plain chessboard topological detector path.
- [x] BENCH-W2H - ChArUco rig-hand-eye correction.
  Completed 2026-05-31. Adds typed ChESS detector threshold overrides, wires
  the private ChArUco rig to EyeToHand Scheimpflug staged BA, reports
  Intrinsic/RigExtrinsic/HandEye levels for rig hand-eye, and flags robot-pose
  corrections that exceed configured priors.
