# M5-T04: Config Type Naming Standardization

Date: 2026-03-07
Commit: pending

## Scope

- Standardized top-level problem config type names to `<ProblemName>Config` in Rust pipeline/facade APIs.
- Applied hard renames:
  - `PlanarConfig` -> `PlanarIntrinsicsConfig`
  - `ScheimpflugIntrinsicsCalibrationConfig` -> `ScheimpflugIntrinsicsConfig`
- Updated module docs, facade exports, integration tests, and README/book snippets to the new names.
- Audited nested config structure across problems and kept current design:
  - stage-grouped nested configs remain for complex workflows (`LaserlineDevice*Config`, `RigHandeye*Config`)
  - simpler workflows may stay flat when structure remains clear.

## Files changed

- `crates/vision-calibration-pipeline/src/planar_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/planar_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `crates/vision-calibration/src/lib.rs`
- `crates/vision-calibration/tests/scheimpflug_intrinsics.rs`
- `crates/vision-calibration/README.md`
- `README.md`
- `book/src/planar_intrinsics.md`
- `crates/vision-calibration-py/src/lib.rs`
- `crates/vision-calibration-py/python/vision_calibration/types.py`
- `docs/backlog.md`
- `docs/report/2026-03-07-M5-T04-config-type-naming.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass
- `python3 -m compileall crates/vision-calibration-py/python/vision_calibration` -> pass

## Follow-ups / risks

- This is a breaking rename for Rust users importing old config type names.
