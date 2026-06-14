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

## O — apex-solver backend

- [ ] O1-BACKEND - `ApexSolverBackend` in
  `optim/src/backend/apex_solver_backend.rs` behind an `apex-solver` cargo
  feature; `BackendKind::ApexSolver` + dispatch arm; mirror `compile_factor`
  from `tiny_solver_backend.rs`. Pre-verify before coding: does apex-solver
  1.3 accept generic `Factor` impls (its API is graph-flavored)?; SE3
  quaternion order vs IR `[qx,qy,qz,qw,tx,ty,tz]` (write a round-trip unit
  test); S2 manifold availability (fallback: Euclidean R3 + renormalize);
  Huber/Cauchy/Arctan loss coverage.
- [ ] O2-AB - Backend A/B validation: synthetic IR param parity < 1e-4;
  bench datasets final cost within 0.1 %, reproj within 0.01 px; timing
  comparison on rtv3d; backend selector surfaced in `calib-bench`.
- [ ] O3-CERES - Remove the `BackendKind::Ceres` stub (F2,
  `optim/src/backend/mod.rs:119`).

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
- [ ] M1-RATIONAL - Rational distortion k4–k6 (OpenCV rational model):
  new `DistortionModel` impl + `AnyDistortion`/`DistortionParams` variants,
  fix-mask plumbing, undistort fixed-point check, synthetic GT tests.
- [ ] M2-THINPRISM - Thin-prism s1–s4; compose with Scheimpflug sensor;
  metrology-lens synthetic tests.
- [ ] M3-DIVISION - Division model (1–2 params, analytically invertible).
- [ ] M4-FISHEYE - Kannala-Brandt equidistant k1–k4: new `ProjectionModel`
  impl (first beyond `Pinhole`), linear-init changes (Zhang assumptions
  don't hold at large FOV), synthetic wide-FOV tests.

## B — app (extend; sequencing serves V-track)

- [ ] B3C-RIG - Run-workspace coverage for `RigExtrinsics`, `RigHandeye`,
  `RigLaserlineDevice` + charuco detector wiring
  (`app/src-tauri/src/run.rs:131` topology dispatch).
- [ ] B-LASER - Laserline visualization: laser-pixel overlay in Diagnose,
  point-to-plane residuals (mm) panel, laser planes in the 3D rig viewer.
- [ ] B-EXPLORE - Pre-calibration dataset exploration: per-camera/pose image
  grid, detection-cache overlay, board coverage map.
- [ ] B-INFRA - ts-rs (or specta) codegen for wire types + export
  discriminator tag (F6); `resource_dir` presets
  (`RunWorkspace/presets.ts:16`); Vitest unit + Playwright smoke tests.

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
