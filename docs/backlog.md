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
  pins) tests green; clippy + doc clean. **Left:** the small/bounded
  `triangulation.rs` `2N×4` sites (`N` = view count); `linear`→`vision-geometry`
  helper de-dup is C1-FOLLOWUP.
- [ ] P2-BA-DENSITY - Principled corner budget for the joint rig + hand-eye BA
  (spatially-distributed subsample preserving coverage, or per-stage decimation
  knobs). Extrinsics/hand-eye converge on a fraction of the corners; the
  per-camera intrinsics stage already uses a cheap subsampled tilt sweep + a
  full-data refine (`optimize_scheimpflug_intrinsics_staged`).
- [ ] P3-BACKEND-COST - Profile the tiny-solver split (autodiff Jacobian vs
  `JᵀJ` assembly vs linear solve vs per-retry residual re-eval). Evaluate
  analytic Jacobians for the hot `ReprojPoint` factor, Jacobian/residual caching,
  and parallel (rayon) residual + Jacobian evaluation.
- [ ] P4-CRITERION - criterion benchmarks for the hot paths (homography DLT,
  distortion fit, one per-camera BA iteration, one joint-BA iteration) to guard
  against regressions and quantify P1–P3 gains. (`criterion-bench` skill.)
- [ ] P5-STAGE-TIMING - Per-stage timing instrumentation in the pipeline (behind
  `verbosity`) so future regressions surface without ad-hoc harnesses.
- [ ] P6-PERCAM-CONVERGENCE - From-scratch per-camera Scheimpflug init diverges
  on the 2 harder `rtv3d_ref` cameras (cam 3 ~248 px, cam 4 ~52 px — the same
  pair that ran `fx→0` pre-staging); the Phase 3 guard correctly rejects them.
  Robustness, not performance: better linear seeds for ill-conditioned cameras
  (cam 3 currently falls back to the distortion-free iteration-0 estimate), a
  wider tilt sweep, or the gated Euclidean L2 prior (degeneracy "Rung 4").
  **Blocks measuring the joint rig/hand-eye BA stages** (P2/P3), which the guard
  stops short of. Report: `docs/report/2026-06-16-perf-from-scratch-rig-profiling.md`.

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
- [ ] M-WIRE - Pipeline-selection plumbing for the new distortion models:
  user-facing distortion-model choice through `PlanarIntrinsicsConfig` /
  `ScheimpflugIntrinsicsConfig` into the problem builders (which hardcode
  `CameraModelDesc::PINHOLE4_DIST5*`). Requires: generalize the BC5-hardwired
  `DistortionFixMask`, model-aware init seeds (note Division's `lambda=0`
  degeneracy + Rational k1↔k4 correlation), variable-dim pack/unpack, export
  reconstruction per model, schema regen, app selector. Touches validated
  calibration paths — do supervised, not in an autonomous batch.
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
- [ ] C1-FOLLOWUP - De-duplicate `vision-calibration-linear` onto
  `vision-geometry` (have `linear` depend on the new crate, delete the moved
  solvers). Deferred from C1 to keep landing additive; do supervised with the
  numeric pins on both sides as the gate. Also: promote both crates to the
  crates.io publish set + release version-lockstep when the API stabilises;
  PyO3 bindings (deferred per A5); close the stale `mvg` branch / PR #28.
- [ ] C2-TRIANGULATION - N-view triangulation + nonlinear refinement.
- [ ] C3-BA - Bundle adjustment with frozen intrinsics, free poses, free
  structure.

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
