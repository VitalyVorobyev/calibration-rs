# calibration-rs Roadmap

Canonical short-form summary of the multi-quarter direction. Detailed reasoning per track
lives in ADRs (`docs/adrs/`); work-in-flight lives in open PRs.

## Status (as of 2026-04-29)

- **Version line:** 0.x. v1.0 (= stable public API) is deferred until the API has been
  stable across two minor releases without breaking changes. Pre-1.0 means breaking changes
  are acceptable.
- **Active branch:** `main`.
- **In-flight PRs:**
  [#27 Manual init](https://github.com/VitalyVorobyev/calibration-rs/pull/27) — calibration
  warm-start across all problem types.
  [#28 mvg](https://github.com/VitalyVorobyev/calibration-rs/pull/28) — multiple-view
  geometry crate split.

## Four tracks

### Track A — Calibration core

Anchored to the **puzzle 130x130 rig** — the workspace's primary internal use case. The rig
is structurally complete; its only blocker is Zhang's intrinsics init failing on a specific
real-data camera ("invalid sign for lambda"). PR #27 is the literal unblocker.

- **A1** Land PR #27 — manual init / warm-start across all nine problem types. See
  [ADR 0011](adrs/0011-manual-initialization-workflow.md).
- **A2** Per-feature residuals on every `*Export` type (gates the diagnose UI). New
  ADR 0012 pending.
- **A3** Zhang's init robustness: detect the lambda-sign failure and fall back to a
  constant-K estimate.
- **A4** EyeToHand mode for the Scheimpflug hand-eye family (currently EyeInHand only).
- **A5** Python bindings parity for `RigScheimpflugExtrinsics`, `RigScheimpflugHandeye`,
  `RigLaserlineDevice`.
- **A6** `rig_family` shared-helpers refactor (defer until a third sibling problem type
  exists).

### Track B — Tauri 2 + React + TypeScript desktop app

A production-grade internal tool wrapping the calibration library. ~4–6 months end-to-end
at one-engineer cadence. ADR 0013 pending — records the choice over `rerun.io` and `egui`
(rejected: `rerun`'s developer-tool aesthetic + gRPC clashes with this project's zenoh-Rust
stack; `egui` has less ecosystem leverage for production polish).

- **B0** Scaffolding — new `app/` directory with conventional Tauri 2 layout
  (`app/src-tauri/` Rust + `app/src/` React + TS).
- **B1** Project model + file loading (images + calibration config in).
- **B2** Feature detection pipeline wrapping `chess-corners` and `calib-targets`.
- **B3** Calibration runner with live progress streaming via Tauri events.
- **B4** 3D rig viewer (Three.js / React Three Fiber: camera frustums, target poses,
  laser plane).
- **B5** **Diagnose mode — the MVP.** Per-feature reprojection arrows, per-image residual
  heatmaps, drill-down, coordinated highlighting between 2D image, 3D scene, and sidebar.
  Depends on A2.
- **B6** Production polish — wizard, settings persistence, signed installers per OS.

### Track C — MVG (postponed; depends on B5 done)

PR #28 splits two-view geometry into `vision-geometry` (deterministic solvers) and
`vision-mvg` (pipelines, robust estimation). Post-merge the track extends to multi-view
geometry over already-calibrated rigs. ADR 0014 will cap the ceiling explicitly: no
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
- **D4** v1.0 release once all puzzle rig phases run green via the Tauri app, PRs #27 and
  #28 + A2 + B5 + C4 have all landed, and the API has been stable across two minor releases.

## Sequencing

```
Weeks 1–2:    A1 (PR #27 rebase + land)   +  B0 (Tauri scaffolding)
Weeks 3–6:    A2, A3                      +  B1 (file load) → B2 (detection)
Weeks 7–12:   A4, A5, A6 cleanup          +  B3 (runner) → B4 (3D viewer)
Weeks 13–24:  (A done)                    +  B5 — the MVP
Weeks 25+:                                   B6 polish + C1 (PR #28 land)
Weeks 28–40:                                                C2 → C3 → C4 → C5
Weeks 40+:                                                                   D-ratchets, v1.0
```

Load-bearing path: **A1 → A2 → B5**. A1 unblocks the rig immediately. A2 is the small
schema extension that gates the diagnose UI. Everything else is parallelizable cleanup or
future ambition.

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
  [`0012-per-feature-reprojection-residuals.md`](adrs/0012-per-feature-reprojection-residuals.md) (A2, landed in PR #33 + follow-ups);
  `0013-tauri-desktop-app.md` (B0, pending);
  `0014-mvg-ceiling.md` (C1, pending).
