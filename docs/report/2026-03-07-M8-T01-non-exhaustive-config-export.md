# M8-T01: `#[non_exhaustive]` for Public Config/Export Types

Date: 2026-03-07
Commit: pending

## Scope

- Added `#[non_exhaustive]` to public config/export structs in pipeline session/workflow contracts, including:
  - `PlanarIntrinsicsConfig`, `PlanarIntrinsicsExport`
  - `ScheimpflugIntrinsicsConfig`, `ScheimpflugIntrinsicsExport`
  - `SingleCamHandeyeConfig`, `SingleCamHandeyeExport`
  - `RigExtrinsicsConfig`, `RigExtrinsicsExport`
  - `RigHandeyeConfig` + nested config groups + `RigHandeyeExport`
  - `LaserlineDeviceConfig` + nested config groups + `LaserlineDeviceExport`
  - `session::ExportRecord`
- Updated external-crate tests (integration/facade) that previously constructed these non-exhaustive config structs via literals, migrating to `Default` plus field mutation.

## Files changed

- `crates/vision-calibration-pipeline/src/session/types.rs`
- `crates/vision-calibration-pipeline/src/planar_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/single_cam_handeye/problem.rs`
- `crates/vision-calibration-pipeline/src/rig_extrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/problem.rs`
- `crates/vision-calibration-pipeline/src/laserline_device/problem.rs`
- `crates/vision-calibration-pipeline/tests/laserline_device.rs`
- `crates/vision-calibration/tests/scheimpflug_intrinsics.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M8-T01-non-exhaustive-config-export.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo clippy -p vision-calibration --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- This is an intentional API hardening change: downstream users can no longer instantiate these non-exhaustive structs with literal syntax and should use constructors/defaults + mutation or helper builders.
