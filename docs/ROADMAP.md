# calibration-rs Roadmap

Canonical short-form summary of the multi-quarter direction. Detailed reasoning per track
lives in ADRs (`docs/adrs/`); work-in-flight lives in open PRs.

## Status (as of 2026-06-11)

- **Version line:** 0.x (latest release 0.5.1). v1.0 (= stable public API) is deferred
  until the API has been stable across two minor releases without breaking changes.
  Pre-1.0 means breaking changes are acceptable.
- **Active branch:** `main`.
- **Track A — Calibration core: COMPLETE.** A1 (manual init), A2 (per-feature
  residuals), A4 (Scheimpflug EyeToHand) shipped. A3 closed (false premise — the
  reported Zhang failure was a fixed puzzleboard-detector bug). A5 dropped (no real
  Python consumer; revisit after the Rust API stabilises). A6 (`rig_family` sensor-
  axis refactor) shipped via PRs #36 + #37 + #38; see
  [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md).
- **Track B — Tauri viewer:** B0 (PR #40), B0.5/B0.6 (PR #42), B1 (PR #43), B2
  (PR #44), **B3a + B3b (PR #45) SHIPPED**. The `app/` shell hosts four workspaces
  (Diagnose, 3D, Epipolar, Run); the Run workspace covers PlanarIntrinsics +
  chessboard end-to-end. Bench crate + multi-level reprojection report shipped
  (PR #49). A 2026-06-11 review (see
  [workspace review](report/2026-06-11-workspace-review.md)) confirmed the
  extend-don't-rebuild verdict for the app.
- **New tracks (2026-06-11):** V (real-data validation on the rtv3d datasets),
  O (apex-solver optimization backend), M (camera-model expansion — supersedes the
  former "new camera models out of scope" line).
- **In-flight PRs:**
  [#28 mvg](https://github.com/VitalyVorobyev/calibration-rs/pull/28) — multiple-view
  geometry crate split (Track C, deferred until the B3 series stabilises).

## Tracks

### Track A — Calibration core (DONE)

Eight problem types across four workflow modules: `planar_intrinsics`,
`scheimpflug_intrinsics`, `single_cam_handeye`, `laserline_device`,
`rig_extrinsics` (pinhole + Scheimpflug via `SensorMode`), `rig_handeye`
(pinhole + Scheimpflug via `SensorMode`), and `rig_laserline_device`. All carry
manual init (ADR 0011) and per-feature residuals on export (ADR 0012).

| Item | Status | Notes |
|------|--------|-------|
| **A1** Manual init | **SHIPPED** (PR #32) | ADR 0011, all problem types, `manual_init_proof` example, tutorial. |
| **A2** Per-feature residuals | **SHIPPED** (PR #33 + #35) | ADR 0012, every `*Export` carries `per_feature_residuals`, tutorial. |
| **A3** Zhang lambda-sign fallback | **CLOSED — false premise** | The reported failure was a puzzleboard-detector bug, since fixed. No real-data Zhang failure to defend against. |
| **A4** Scheimpflug EyeToHand | **SHIPPED** | `RigHandeyeProblem` (Scheimpflug variant) supports both `EyeInHand` and `EyeToHand` via `RigHandeyeInitConfig::handeye_mode`. |
| **A5** Python parity | **DROPPED** | No real Python consumer; revisit *after* the Rust API stabilises, as one coherent build, not parity patches. |
| **A6** `rig_family` refactor | **SHIPPED** (PRs #36 + #37 + this PR) | Sensor-axis-only collapse. Five rig sibling modules → three. Net ~−2,300 LoC. See [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md). |

**Track A exit criterion (met):** the `vision-calibration` facade is stable enough
that B0 (Tauri scaffold) can compile against it with no breaking churn back into core.

### Track B — Tauri 2 + React + TypeScript desktop app

A production-grade internal tool wrapping the calibration library.
[ADR 0014](adrs/0014-tauri-desktop-app.md) records the framework choice
(Tauri 2 + React + TS over `rerun.io` and `egui`) and the diagnose-first
v0 scope. After B0–B2 shipped, a 2026-05-02 grill session committed the
track to a much larger goal: make the app the **primary calibration
tool**, not a passive viewer.

**End-state vision (settled 2026-05-02):** point the app at any
foreign dataset → AI inspects the layout → emits a canonical
[`DatasetSpec`](adrs/0016-dataset-manifest.md) manifest with fields
the AI couldn't determine listed under `_unresolved` (no silent
guessing, [ADR 0019](adrs/0019-fail-fast-on-ambiguity.md)) → schema-
driven forms ([ADR 0018](adrs/0018-schema-driven-ui.md)) let the user
edit any of the manifest's or the per-problem-type config's fields →
Run dispatches detection (cached, [ADR 0017](adrs/0017-detection-cache.md))
+ calibration in-process and routes the export into `/diagnose`. All
8 problem types and 4 target detectors (chessboard / charuco /
puzzleboard / ringgrid) are supported.

**Phase 1 — passive viewer (DONE, 2026-05-02).**

- **B0 — diagnose viewer v0** (PR #40). Passive viewer of one
  `PlanarIntrinsicsExport`. `ImageManifest` Export-side contract.
- **B0.5/B0.6 — real-data acceptance + viewer UX** (PR #42). Manifest
  extended to `RigHandeyeExport`; puzzle 130×130 Scheimpflug rig
  rendered correctly; design tokens, navigation, theme toggle.
- **B1 — 3D rig viewer** (PR #43). React-Three-Fiber scene with rig
  origin, per-camera frustums, target boards. Multi-workspace shell.
- **B2 — epipolar workspace** (PR #44). Server-side
  `compute_epipolar_overlay` Tauri command via canonical camera
  models. Two-pane viewer with click-to-pick.

**Phase 2 — self-contained calibration app (in flight).**

- **B3a — foundation** (PR #45). `schemars` derives across every
  `*Config` + shared option types; `cargo xtask emit-schemas` →
  `app/src/schemas/`. New crates `vision-calibration-dataset`
  (`DatasetSpec` + validator) and `vision-calibration-detect`
  (`Detector` trait, `ChessboardDetector`, `DetectionCache` trait
  + filesystem impl). `pipeline::dataset_runner::build_planar_input`
  wires manifest → cache → detect-on-miss → `PlanarDataset` IR.
  ADRs 0016–0019.
- **B3b — Tauri runner + Run workspace.** Tauri `run_calibration`
  command (Planar+Chessboard end-to-end), schema-driven
  `<ConfigForm/>` React component, Run workspace replaces the stub.
  Vertical slice ships first; coverage to all 8 topologies + 4
  detectors follows in B3c.
- **B3c — coverage.** Wire the remaining 7 problem types through
  dispatch; wire charuco / puzzleboard / ringgrid detectors;
  per-problem-type `DatasetSpec → *Input` converters; manifest
  sweep finish on `SingleCamHandeyeExport`,
  `ScheimpflugIntrinsicsExport`, and `LaserlineDeviceExport`.
  **Sequencing (2026-06-11): RigHandeye + RigLaserlineDevice + charuco
  detector first** — they serve the V-track (rtv3d) directly.
- **B-laser — laserline visualization.** Laser-pixel overlay in
  Diagnose; laser plane-fit residuals (point-to-plane mm) alongside
  reprojection residuals; laser planes rendered in the 3D rig viewer.
  No laser data is visible anywhere in the app today.
- **B-explore — dataset exploration.** Browse a dataset *before*
  calibrating: image grid per camera/pose, detection overlay from the
  cache, board coverage map. Today the app only visualizes exports.
- **B3d — manifest UX.** AI-driven `generate-manifest` CLI binary
  (heuristic-only v0: regex / file-extension / vendor signature /
  README scraping). Tauri "Sniff folder" command. `_unresolved` UX
  (red badges, blocked Run button). `AskUser` modal component.
  Frame-convention validator with vendor-aware error messages.
- **B3e — iteration polish.** Cancellability for long solves;
  progress event streaming; multi-pose residual stats panel;
  cross-camera residual matrix; experiments directory storing
  `(dataset.toml, config.json, export.json)` tuples for
  reproducibility.

**Phase 3 — deferred until B3 stabilises.**

- LLM-backed manifest inference (separate ADR; opt-in, behind API
  key configuration).
- Init-failure diagnosis sweeps (perturbed re-runs).
- Signed installers per OS.
- Infra ratchet: ts-rs (or specta) TS codegen replacing hand-written
  wire types + an export discriminator tag; `resource_dir`-based
  presets; Vitest unit + Playwright smoke tests.

### Track V — Real-data validation: rtv3d (NEW 2026-06-11, top priority)

Prove the library functional on the rtv3d sensor — 6 laser-plane-triangulation
devices (Scheimpflug camera + laser projector), `privatedata/rtv3d_1` (target +
laser) and `rtv3d_2` (target only). Each ships a legacy-system `artifacts.json`
oracle to beat (cams 0–4: 1.6–2.8 px reproj, 0.03–0.05 mm plane σ; cam 5 is
degenerate in the oracle — fx=51, 127 px). Full inventory and oracle schema in
the [2026-06-11 workspace review](report/2026-06-11-workspace-review.md).

- **V1** `rtv3d_rig` example in `examples-private` (clone of
  `puzzle_130x130_rig`): ChArUco 22×22 detection, EyeInHand
  `RigHandeye(Scheimpflug)`, oracle comparison tables, empirical 4.8 vs 5.2 mm
  cell-size resolution. Validate on rtv3d_2 (no laser).
- **V2** rtv3d_1 full pipeline: + laser detection on the 4 `double_snap` poses,
  `RigLaserlineDevice`, joint BA; laser-plane comparison vs oracle
  (origin/xaxis/yaxis → normal+distance conversion).
- **V3** Beat-the-oracle report (`docs/report/`): per-camera reproj < oracle on
  cams 0–4; sane cam 5 (fx ∈ [1500, 2500], reproj < 10 px); extrinsics within
  2° / 5 mm; plane normals within 0.5°, distances within 1 mm, σ < oracle.
- **V4** Bench registry entries (`registry/private.json`) for both datasets →
  regression tracking; full RigLaserlineDevice-in-bench as follow-up.

### Track O — Optimization backends (NEW 2026-06-11)

`OptimBackend` (ADR 0008) gets a second real implementation:
[apex-solver](https://crates.io/crates/apex-solver) 1.3 (LM/GN/DogLeg, Lie-group
support), behind an `apex-solver` cargo feature in `vision-calibration-optim`.

- **O1** `ApexSolverBackend` implementing `OptimBackend`, mirroring
  `compile_factor` from the tiny-solver backend. Pre-verify: whether apex-solver
  accepts generic factors (its API is graph-flavored), SE3 quaternion order vs
  our `[qx,qy,qz,qw,tx,ty,tz]` (round-trip unit test), S2 manifold availability
  (fallback: R3 + renormalize), robust-loss coverage.
- **O2** Backend A/B in bench: param parity < 1e-4 on synthetic IR, final cost
  within 0.1 % on bench datasets, timing comparison on rtv3d_1.
- **O3** Drop the `BackendKind::Ceres` stub.

### Track M — Camera-model expansion (NEW 2026-06-11)

Supersedes the former "new camera models out of scope" rule — all four models
below are user-requested. Gated on M0: the `FactorKind` IR currently enumerates
projection × distortion × sensor × chain combinations, so new models multiply
variants (workspace review finding F1).

- **M0** Factor generification: one reprojection-factor family per *chain*,
  camera model as data (seed: the dyn-safe `CameraProject` trait in core). Also
  resolves the `PinholeCamera`-typed residual helper (finding F3) and unblocks
  pinhole rig laserline (finding F4).
- **M1** Rational distortion k4–k6 (distortion slot only; OpenCV rational model).
- **M2** Thin-prism s1–s4 (composes with Scheimpflug; metrology lenses).
- **M3** Division model (cheap, invertible; self-calibration friendly).
- **M4** Kannala-Brandt fisheye (new projection slot + linear-init changes —
  the biggest lift; last).

### Track C — MVG (postponed; depends on diagnose viewer done)

PR #28 splits two-view geometry into `vision-geometry` (deterministic solvers) and
`vision-mvg` (pipelines, robust estimation). Post-merge the track extends to multi-view
geometry over already-calibrated rigs. ADR 0015 will cap the ceiling explicitly: no
in-house dense matcher, no full SfM.

- **C1** Land PR #28.
- **C2** N-view triangulation + nonlinear refinement.
- **C3** Bundle adjustment with frozen intrinsics, free poses, free structure.
- **C4** Stereo rectification — including **Scheimpflug-aware rectification** (genuinely
  novel for this project).
- **C5** Dense matcher integration behind a `dense-opencv` feature flag, wrapping
  `opencv-rust` SGBM.
- **C-UI** MVG visualizations layered into the Tauri app (point clouds, depth maps,
  rectified pairs).

### Track D — Earn v1.0 (continuous ratchet)

- **D1** Typed errors only — no `String`-typed escape hatches in public APIs.
- **D2** Doc-warning-free, MSRV pinned at v1.0 cut (currently 1.93).
- **D3** Python binding parity audited at every minor version bump.
- **D4** v1.0 release once the puzzle rig runs green end-to-end via the Tauri app,
  PR #28 + the diagnose viewer + C4 have all landed, and the API has been stable across two minor
  releases.

## Load-bearing path

**V1 → V2 → V3 (prove the library on rtv3d) → V4 + B3c coverage (rig +
laserline + charuco first) → B-laser visualization → B3d manifest UX →
B3e polish.** O1/O2 (apex-solver) and M0 (factor generification) are
parallelizable with the V-track; M1–M4 follow M0. C resumes once B3c
lands; D is a continuous ratchet.

## Out of scope (explicit)

- Camera models beyond the M-track set (omnidirectional / MEI, double-sphere,
  telecentric, spline). Defer until a concrete project demands one.
- In-house dense stereo matching. External (`opencv-rust` SGBM) only.
- Full structure-from-motion (incremental SfM, pose graph, loop closure).
- `rerun.io` / `egui` as the UI.
- OSS-grade community surface and multi-platform install docs (internal-first; defer until
  the tool earns it).
- What-if interactive re-optimize (B6 stretch at earliest).

## See also

- [ADR index](adrs/README.md) — design records.
- [Tutorials](tutorials/README.md) — hands-on walkthroughs for new users.
- [MSRV notes](MSRV.md) — why the lockfile is frozen below latest releases.
- Per-track ADRs:
  [`0011-manual-initialization-workflow.md`](adrs/0011-manual-initialization-workflow.md) (A1, landed in PR #32);
  [`0012-per-feature-reprojection-residuals.md`](adrs/0012-per-feature-reprojection-residuals.md) (A2, landed in PR #33 + #35);
  [`0013-rig-family-sensor-axis-refactor.md`](adrs/0013-rig-family-sensor-axis-refactor.md) (A6, landed in PRs #36 + #37 + #38);
  [`0014-tauri-desktop-app.md`](adrs/0014-tauri-desktop-app.md) (B0, landed in PR #40);
  [`0016-dataset-manifest.md`](adrs/0016-dataset-manifest.md) (B3a, PR #45);
  [`0017-detection-cache.md`](adrs/0017-detection-cache.md) (B3a, PR #45);
  [`0018-schema-driven-ui.md`](adrs/0018-schema-driven-ui.md) (B3a, PR #45);
  [`0019-fail-fast-on-ambiguity.md`](adrs/0019-fail-fast-on-ambiguity.md) (B3a, PR #45);
  `0015-mvg-ceiling.md` (C1, pending).
