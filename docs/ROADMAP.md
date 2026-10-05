# calibration-rs Roadmap

What stands between the 0.x line and v1.0, and what is deliberately out of
scope. Design reasoning lives in [ADRs](adrs/README.md), tasks in the
[backlog](backlog.md), released changes in the [CHANGELOG](../CHANGELOG.md).

## State

Version line 0.x: nine crates on crates.io plus the `vision-calibration`
Python package on PyPI. Pre-1.0, breaking changes are allowed; each is listed
in the CHANGELOG with migration notes.

## v1.0 exit criteria

1. **Acceptance** — `calib-bench accept` passes every on-disk registered
   dataset through the seeded official route (hard gates per entry).
2. **Soundness** — every shipped algorithm family has a proof pack
   ([`notes/`](notes/README.md)): math note, synthetic-GT matrix test,
   property tests where an invariant exists, and a committed regression
   baseline with a drift gate.
3. **Frozen API** — facade-only consumers, normalized configs
   ([ADR 0024](adrs/0024-config-vocabulary.md)), typed errors, no duplicate
   public type names, no deprecated shims; and the API unchanged across two
   consecutive minor releases.
4. **App CI green** — lint (zero warnings), format, typecheck, unit,
   component and smoke tests, generated IPC types in sync.
5. **Docs current** — README, crate READMEs, the book, tutorials, ADRs and
   this file describe the code as it is, and user-facing docs carry no
   internal references (`cargo xtask check-docs`).

## Open work

Tracked in the [backlog](backlog.md); each entry names its blocker or
trigger.

## Standing decisions

- **Seeded initialization is the official route** for Scheimpflug and rig
  calibration ([ADR 0022](adrs/0022-scheimpflug-intrinsics-seeded-default.md),
  [ADR 0023](adrs/0023-device-spec-seed-derivation.md)).
- **MVG stops short of SfM** ([ADR 0015](adrs/0015-mvg-ceiling.md)): two-view
  solvers, N-view pose recovery, triangulation, bundle adjustment,
  rectification and a pure-Rust dense matcher, nothing beyond.
- **Two solver backends, one LM loop.** tiny-solver (default) and factrs
  compile the optimization IR
  ([ADR 0008](adrs/0008-backend-agnostic-optimization-ir.md)) and both drive
  the shared Levenberg–Marquardt loop
  ([ADR 0025](adrs/0025-factrs-backend-shared-lm.md)). The default changes
  only on a measured comparison and the user's decision.

## Out of scope

- Camera models beyond the shipped set (pinhole with Brown-Conrady, rational,
  thin-prism and division distortion; Scheimpflug sensor): no
  omnidirectional/MEI, double-sphere, telecentric or spline models until a
  project needs one.
- Full structure-from-motion: incremental SfM, pose graphs, loop closure.
- `rerun.io` / `egui` as the UI.
- A community-grade contribution surface and multi-platform install docs.
