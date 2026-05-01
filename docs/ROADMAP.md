# calibration-rs Roadmap

Canonical short-form summary of the multi-quarter direction. Detailed reasoning per track
lives in ADRs (`docs/adrs/`); work-in-flight lives in open PRs.

## Status (as of 2026-05-01)

- **Version line:** 0.x. v1.0 (= stable public API) is deferred until the API has been
  stable across two minor releases without breaking changes. Pre-1.0 means breaking changes
  are acceptable.
- **Active branch:** `main`.
- **Track A — Calibration core: COMPLETE.** A1 (manual init), A2 (per-feature
  residuals), A4 (Scheimpflug EyeToHand) shipped. A3 closed (false premise — the
  reported Zhang failure was a fixed puzzleboard-detector bug). A5 dropped (no real
  Python consumer; revisit after the Rust API stabilises). A6 (`rig_family` sensor-
  axis refactor) shipped across PRs #36 + #37 + this PR; see
  [ADR 0013](adrs/0013-rig-family-sensor-axis-refactor.md).
- **In-flight PRs:**
  [#28 mvg](https://github.com/VitalyVorobyev/calibration-rs/pull/28) — multiple-view
  geometry crate split (Track C, deferred until B5 ships).

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

A production-grade internal tool wrapping the calibration library. ~4–6 months
end-to-end at one-engineer cadence. ADR 0014 will record the choice over
`rerun.io` and `egui` (rejected: `rerun`'s developer-tool aesthetic + gRPC
clashes with this project's zenoh-Rust stack; `egui` has less ecosystem leverage
for production polish).

- **B0** Scaffolding — new `app/` directory with conventional Tauri 2 layout
  (`app/src-tauri/` Rust + `app/src/` React + TS). **Next up.**
- **B1** Project model + file loading (images + calibration config in).
- **B2** Feature detection pipeline wrapping `chess-corners` and `calib-targets`.
- **B3** Calibration runner with live progress streaming via Tauri events.
- **B4** 3D rig viewer (Three.js / React Three Fiber: camera frustums, target poses,
  laser plane).
- **B5** **Diagnose mode — the MVP.** Per-feature reprojection arrows, per-image
  residual heatmaps, drill-down, coordinated highlighting between 2D image, 3D
  scene, and sidebar. Consumes A2's per-feature residuals.
- **B6** Production polish — wizard, settings persistence, signed installers per OS.

### Track C — MVG (postponed; depends on B5 done)

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
  PR #28 + B5 + C4 have all landed, and the API has been stable across two minor
  releases.

## Load-bearing path

**B0 → … → B5.** Track A is done; the diagnose UI (B5) is the next user-facing
milestone and consumes the A2 per-feature-residuals foundation that already
ships on every export. C is parallelizable once B5 lands; D is a continuous
ratchet.

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
  [`0013-rig-family-sensor-axis-refactor.md`](adrs/0013-rig-family-sensor-axis-refactor.md) (A6, landed in PRs #36 + #37 + this PR);
  `0014-tauri-desktop-app.md` (B0, pending);
  `0015-mvg-ceiling.md` (C1, pending).
