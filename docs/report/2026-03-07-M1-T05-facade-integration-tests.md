# M1-T05: Facade Scheimpflug Integration Tests

Date: 2026-03-07
Commit: pending

## Scope

- Added integration tests under `crates/vision-calibration/tests/` for the facade Scheimpflug API.
- Coverage includes:
  - synthetic convergence,
  - deterministic-noise convergence,
  - input validation errors,
  - invalid config errors,
  - config/result JSON roundtrip checks.

## Files changed

- `crates/vision-calibration/tests/scheimpflug_intrinsics.rs`
- `docs/backlog.md`

## Validation run

- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- Pipeline-level problem/state tests are already in `vision-calibration-pipeline` and should stay in sync with facade integration behavior.
