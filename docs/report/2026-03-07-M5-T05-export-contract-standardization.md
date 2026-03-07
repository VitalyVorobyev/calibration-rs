# M5-T05: Export Contract Standardization

Date: 2026-03-07
Commit: pending

## Scope

- Standardized export contracts so every problem module exposes a distinct `<ProblemName>Export` type.
- Added consistent top-level fields to all exports:
  - `mean_reproj_error`
  - `per_cam_reproj_errors`
- Implemented single-camera convention (`per_cam_reproj_errors` has one element) for planar intrinsics, Scheimpflug intrinsics, and laserline-device workflows.
- Replaced alias-based exports with explicit structs for:
  - `PlanarIntrinsicsExport`
  - `ScheimpflugIntrinsicsExport`
  - `LaserlineDeviceExport`
- Updated Python payload contracts and result dataclasses to parse and expose the standardized fields.

## Files changed

- `crates/vision-calibration-pipeline/src/planar_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/planar_intrinsics/mod.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/problem.rs`
- `crates/vision-calibration-pipeline/src/laserline_device/problem.rs`
- `crates/vision-calibration-pipeline/tests/laserline_device.rs`
- `crates/vision-calibration/src/lib.rs`
- `crates/vision-calibration-py/python/vision_calibration/types.py`
- `crates/vision-calibration-py/python/vision_calibration/models.py`
- `book/src/laserline.md`
- `docs/backlog.md`
- `docs/report/2026-03-07-M5-T05-export-contract-standardization.md`

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

- This is a breaking API/schema change for downstream consumers deserializing previous export payload shapes.
