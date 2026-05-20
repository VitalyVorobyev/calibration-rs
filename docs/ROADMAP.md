# calibration-rs Roadmap

Canonical short-form summary of the multi-quarter direction. Detailed reasoning per track
lives in ADRs (`docs/adrs/`); work-in-flight lives in open PRs.

## Status (as of 2026-05-02)

- **Version line:** 0.x. v1.0 (= stable public API) is deferred until the API has been
  stable across two minor releases without breaking changes. Pre-1.0 means breaking changes
  are acceptable.
- **Active branch:** `main`.
- **Track A — Calibration core: COMPLETE.** A1 (manual init), A2 (per-feature
  residuals), A4 (Scheimpflug EyeToHand) shipped. A3 closed (false premise — the
  reported Zhang failure was a fixed puzzleboard-detector bug). A5 dropped (no real
  Python consumer; revisit after the Rust API stabilises). A6 (`rig_family` sensor-
  axis refactor) shipped via PRs #36 + #37 + #38; see
  [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md).
- **Track B — Tauri viewer:** B0 (PR #40), B0.5/B0.6 (PR #42), B1 (PR #43), B2
  (PR #44) **SHIPPED**. The `app/` shell now hosts four workspaces (Diagnose, 3D,
  Epipolar, Run-stub). After a 2026-05-02 grill the user committed to evolving the
  app into the **primary calibration tool** (Track B is no longer "post-B0 enrichments
  TBD" — see the B-track section below for the committed sub-phasing).
- **In-flight PRs:**
  [#28 mvg](https://github.com/VitalyVorobyev/calibration-rs/pull/28) — multiple-view
  geometry crate split (Track C, deferred until the B3 series stabilises);
  [#45 B3a foundation](https://github.com/VitalyVorobyev/calibration-rs/pull/45) —
  schema-driven configs, `DatasetSpec`, detection cache, fail-fast contract
  (ADRs 0016–0019).

## Four tracks

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
- **D2** Doc-warning-free, MSRV 1.88 frozen.
- **D3** Python binding parity audited at every minor version bump.
- **D4** v1.0 release once the puzzle rig runs green end-to-end via the Tauri app,
  PR #28 + the diagnose viewer + C4 have all landed, and the API has been stable across two minor
  releases.

## Load-bearing path

**B0 → B0.5 → B1 → B2 (DONE) → B3a foundation (in flight) → B3b runner +
Run workspace → B3c coverage (all 8 problem types + 4 detectors) → B3d
manifest UX → B3e iteration polish → C-track resumes.** Track A is done;
the B-track is now committed to making the app self-contained rather than
treating post-B0 work as discretionary. C is parallelizable once B3c
lands (the viewer can render arbitrary exports the runner produces); D
is a continuous ratchet.

## Out of scope (explicit)

- New camera models (fisheye / Kannala-Brandt, omnidirectional, double-sphere, telecentric,
  spline). Defer until a concrete project demands one.
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
