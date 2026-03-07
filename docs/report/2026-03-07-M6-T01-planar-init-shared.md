# M6-T01: Shared Planar Initialization Helpers in Pipeline

Date: 2026-03-07
Commit: pending

## Scope

- Added internal `planar_family` helper module in `vision-calibration-pipeline` to consolidate shared planar bootstrap logic.
- Implemented shared bootstrap flow that computes:
  - view homographies
  - iterative planar intrinsics/distortion initialization (Zhang + iterative distortion)
  - per-view planar pose recovery from homographies
- Refactored `planar_intrinsics::step_init` to use the shared helper and keep writing homographies/initial state as before.
- Refactored `scheimpflug_intrinsics::step_init` to reuse the same shared bootstrap for intrinsics + pose initialization (then applies Scheimpflug-specific sensor/tangential handling).
- Added unit test coverage for the new shared helper on deterministic synthetic planar data.

## Files changed

- `crates/vision-calibration-pipeline/src/lib.rs`
- `crates/vision-calibration-pipeline/src/planar_family.rs`
- `crates/vision-calibration-pipeline/src/planar_intrinsics/steps.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M6-T01-planar-init-shared.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- `M6-T02` remains open: optimization setup is still duplicated between planar and Scheimpflug flows.
