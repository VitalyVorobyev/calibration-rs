# M6-T03: Shared Helper Wiring in Planar and Scheimpflug Steps

Date: 2026-03-07
Commit: pending

## Scope

- Locked step-level wiring for planar family so both workflows use shared helper paths:
  - planar and Scheimpflug init steps use shared `bootstrap_planar_intrinsics` helper in pipeline (`M6-T01` path).
  - planar optimize uses `optimize_planar_intrinsics` and Scheimpflug optimize uses `optimize_scheimpflug_intrinsics`, both backed by shared planar-family IR builder in optim (`M6-T02` path).
- Added explicit comments in Scheimpflug steps to make the shared-helper routing clear and maintainable.
- Added direct pipeline unit tests for Scheimpflug step functions:
  - `step_optimize` requires prior `step_init`
  - full `run_calibration` updates output and optimization state on synthetic Scheimpflug data

## Files changed

- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M6-T03-shared-helper-step-wiring.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- `M6-T04` remains open: ADR 0002 still needs an explicit update describing the finalized planar-family structure.
