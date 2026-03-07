# M5-T06: Session-Only Scheimpflug API

Date: 2026-03-07
Commit: pending

## Scope

- Removed `run_calibration_direct` from `vision-calibration-pipeline::scheimpflug_intrinsics`.
- Updated facade Scheimpflug exports to session-only `run_calibration(&mut session, Option<Config>)`.
- Updated prelude Scheimpflug runner alias to session-based runner.
- Migrated facade integration tests to session-based execution helper.
- Updated Rust docs/examples to use session workflow for Scheimpflug.

## Files changed

- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `crates/vision-calibration/src/lib.rs`
- `crates/vision-calibration/tests/scheimpflug_intrinsics.rs`
- `crates/vision-calibration-py/src/lib.rs`
- `README.md`
- `crates/vision-calibration/README.md`
- `docs/backlog.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass

## Follow-ups / risks

- This is a breaking Rust facade API change for callers using the old direct Scheimpflug helper.
