# calibration-rs Roadmap

What stands between the 0.x line and v1.0, and what is deliberately out of
scope. Design reasoning lives in [ADRs](adrs/README.md), tasks in the
[backlog](backlog.md), released changes in the [CHANGELOG](../CHANGELOG.md).

## State

Pre-1.0 (0.x): breaking changes are allowed; each is listed in the CHANGELOG
with migration notes. The published packages are listed in AGENTS.md.

## v1.0 exit criteria

1. **Acceptance** — `calib-bench accept` passes every on-disk registered
   dataset through the seeded official route (hard gates per entry).
2. **Public regression suite** — a versioned public suite (observations in
   the repository, images fetched from their publishers and verified by
   checksum) gates every PR, with reference results compared against OpenCV.
3. **Uncertainty** — every calibration result carries its parameter
   covariance and projection uncertainty, validated against Monte-Carlo
   scatter in the proof pack.
4. **Soundness** — every shipped algorithm family has a proof pack
   ([`notes/`](notes/README.md)): math note, synthetic-GT matrix test,
   property tests where an invariant exists, and a committed regression
   baseline with a drift gate.
5. **Frozen API** — facade-only consumers, normalized configs
   ([ADR 0024](adrs/0024-api-vocabulary.md)), typed errors, no duplicate
   public type names, no deprecated shims; and the API unchanged across two
   consecutive minor releases.
6. **App CI green** — lint (zero warnings), format, typecheck, unit,
   component and smoke tests, generated IPC types in sync.
7. **Docs current** — README, crate READMEs, the book, tutorials, ADRs and
   this file describe the code as it is, and user-facing docs carry no
   internal references (`cargo xtask check-docs`).

## Open work

Tracked in the [backlog](backlog.md); each entry names its blocker or
trigger.

## Standing decisions

The binding design decisions are the [ADRs](adrs/README.md). Two policies
that are not design:

- The default solver backend changes only on a measured comparison and the
  maintainer's decision.
- Cross-platform only: the library, its comparison harnesses and its CI
  jobs run on Linux, macOS and Windows; tools that do not stay out of the
  suite.

## Out of scope

- Camera models beyond the shipped set (pinhole with Brown-Conrady, rational,
  thin-prism and division distortion; Scheimpflug sensor): no
  omnidirectional/MEI, double-sphere, telecentric or spline models until a
  project needs one.
- Full structure-from-motion: incremental SfM, pose graphs, loop closure
  (ADR 0006).
- `rerun.io` / `egui` as the UI.
- A community-grade contribution surface and multi-platform install docs.
