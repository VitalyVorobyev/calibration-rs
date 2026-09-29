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
   this file describe the code as it is.

## Open work

| Item | Status | Blocker / trigger |
|---|---|---|
| `B-UX2-ELEVATION` — app empty states, error surfaces, image grid, detection overlay, coverage map | open | gate: frontend review with no high-severity findings |
| `B-QUAL-TS7` — TypeScript 7 | blocked | `typescript-eslint` support for TS 7 |
| `D4-NALGEBRA-035` — nalgebra 0.35 / faer 0.24 | blocked | a `tiny-solver` release built on them |
| `D4-RELEASE` — cut v1.0 | open | the exit criteria above |
| `C-PYO3-MVG` — Python bindings for the MVG surface | deferred | a Python consumer |
| `P2-BA-DENSITY` — corner budget for joint rig BA | conditional | acceptance wall time > ~5 min |
| `P3-BACKEND-COST` — solver hot-path profiling | parked | post-1.0 |
| `M4-FISHEYE` — Kannala-Brandt projection | parked | post-1.0, a fisheye dataset |
| `V7-RTV3D-INTRINSICS-FLOOR` | parked | do not reopen unprompted |

## Standing decisions

- **Seeded initialization is the official route** for Scheimpflug and rig
  calibration ([ADR 0022](adrs/0022-scheimpflug-intrinsics-seeded-default.md),
  [ADR 0023](adrs/0023-device-spec-seed-derivation.md)).
- **MVG stops short of SfM** ([ADR 0015](adrs/0015-mvg-ceiling.md)): two-view
  solvers, N-view pose recovery, triangulation, bundle adjustment,
  rectification and a pure-Rust dense matcher, nothing beyond.
- **One solver backend.** tiny-solver LM behind the optimization IR
  ([ADR 0008](adrs/0008-backend-agnostic-optimization-ir.md)). A second
  backend is worth adding only if it brings autodiff, an S2 manifold and
  documented robust losses.

## Out of scope

- Camera models beyond the shipped set (pinhole with Brown-Conrady, rational,
  thin-prism and division distortion; Scheimpflug sensor): no
  omnidirectional/MEI, double-sphere, telecentric or spline models until a
  project needs one.
- Full structure-from-motion: incremental SfM, pose graphs, loop closure.
- `rerun.io` / `egui` as the UI.
- A community-grade contribution surface and multi-platform install docs.
