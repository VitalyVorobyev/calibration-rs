# M5-T03: Stage-Explicit Step Option Naming

Date: 2026-03-07
Commit: pending

## Scope

- Added ADR 0010 to define a single naming convention for step option types: `<Stage><Action>Options`.
- Applied hard API renames across pipeline and facade with explicit stages for all modules.
- Removed abbreviated `*OptimOptions` names in favor of `*OptimizeOptions`.
- Updated backlog status and step-function book docs to reflect the new names.

## Files changed

- `docs/adrs/0010-step-option-naming-convention.md`
- `docs/adrs/README.md`
- `docs/backlog.md`
- `docs/report/2026-03-07-M5-T03-step-option-naming.md`
- `book/src/step_functions.md`
- `crates/vision-calibration-pipeline/src/planar_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/planar_intrinsics/steps.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `crates/vision-calibration-pipeline/src/laserline_device/mod.rs`
- `crates/vision-calibration-pipeline/src/laserline_device/steps.rs`
- `crates/vision-calibration-pipeline/src/single_cam_handeye/mod.rs`
- `crates/vision-calibration-pipeline/src/single_cam_handeye/steps.rs`
- `crates/vision-calibration-pipeline/src/rig_extrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/rig_extrinsics/steps.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/mod.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/steps.rs`
- `crates/vision-calibration/src/lib.rs`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration-py --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass
- `cargo test -p vision-calibration-py --all-features` -> pass

## Follow-ups / risks

- This is a breaking rename for external users importing old option type names.
