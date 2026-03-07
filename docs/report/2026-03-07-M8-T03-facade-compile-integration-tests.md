# M8-T03: Facade Compile-Only Integration Tests

Date: 2026-03-07
Commit: pending

## Scope

- Added integration tests in `vision-calibration` that validate the facade API from an external-user import style.
- Tests use `use vision_calibration::*;` and assert that module sessions and primary runner signatures compile for all public problem modules.
- Added an explicit prelude compile-surface test for hello-world usage (`CalibrationSession<PlanarIntrinsicsProblem>` + `run_planar_intrinsics`).

## Files changed

- `crates/vision-calibration/tests/facade_compile_surface.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M8-T03-facade-compile-integration-tests.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- This task validates compile-time facade shape and import ergonomics; deeper runtime coverage remains owned by problem-specific integration tests.
