# M1-T01: Scheimpflug Pipeline ProblemType Refactor (Bundled)

Date: 2026-03-07
Commit: pending

## Scope

- Refactored Scheimpflug pipeline from a single direct-function module into standard pipeline shape:
  `scheimpflug_intrinsics/{mod.rs,problem.rs,state.rs,steps.rs}`.
- Added `ScheimpflugIntrinsicsProblem` implementing `ProblemType` with schema identity and validation.
- Moved initialization/optimization logic into session step functions (`step_init`, `step_optimize`) and added session runner.
- Added compatibility direct wrapper (`run_calibration_direct`) and preserved facade-level
  `vision_calibration::scheimpflug_intrinsics::run_calibration(dataset, config)` behavior.
- Added state JSON roundtrip tests and problem-level validation/config/export tests.

Bundling note:

- `M1-T01` through `M1-T06` were implemented together because module shape, trait wiring,
  step migration, state contract, tests, and facade compatibility are tightly coupled and
  cannot be landed independently without temporary API or build breakage.

## Files changed

- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/state.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics.rs` (removed)
- `crates/vision-calibration-pipeline/src/lib.rs`
- `crates/vision-calibration/src/lib.rs`
- `docs/backlog.md`

## Validation run

- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- Python bindings still use a dedicated Scheimpflug entrypoint (`M3-T01` pending).
- Planar-family de-duplication is still pending (`M2` tasks).
