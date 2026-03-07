# M6-T02: Shared Planar Optimization IR Setup in Optim Crate

Date: 2026-03-07
Commit: pending

## Scope

- Added shared planar-family IR builder in `vision-calibration-optim` for common optimization setup:
  - intrinsics/distortion parameter blocks
  - optional Scheimpflug sensor block
  - pose blocks with fixed-pose support
  - per-point reprojection residual block construction
- Refactored `optimize_planar_intrinsics` to build IR through the shared helper.
- Added a dedicated `scheimpflug_intrinsics` optimization module in `vision-calibration-optim` with:
  - solve options and parameter/result types
  - `optimize_scheimpflug_intrinsics` entry point
  - solve result unpacking and reprojection metric computation
- Updated pipeline Scheimpflug `step_optimize` to call the new optimizer API instead of building IR directly in pipeline.

## Files changed

- `crates/vision-calibration-optim/src/problems/planar_family_shared.rs`
- `crates/vision-calibration-optim/src/problems/planar_intrinsics.rs`
- `crates/vision-calibration-optim/src/problems/scheimpflug_intrinsics.rs`
- `crates/vision-calibration-optim/src/problems/mod.rs`
- `crates/vision-calibration-optim/src/lib.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `docs/backlog.md`
- `docs/report/2026-03-07-M6-T02-shared-planar-optimization-ir.md`

## Validation run

- `cargo fmt --all` -> pass
- `cargo clippy -p vision-calibration-optim --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-optim --all-features` -> pass
- `cargo clippy -p vision-calibration-pipeline --all-targets --all-features -- -D warnings` -> pass
- `cargo test -p vision-calibration-pipeline --all-features` -> pass
- `cargo test -p vision-calibration --all-features` -> pass

## Follow-ups / risks

- `M6-T03` remains open and should explicitly lock the step-function contract to shared helpers for both planar variants.
