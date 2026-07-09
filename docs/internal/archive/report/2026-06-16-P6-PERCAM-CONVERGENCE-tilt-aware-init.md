# P6-PERCAM-CONVERGENCE: Tilt-Aware Scheimpflug Initialization

## Scope

Implemented the P6 from-scratch Scheimpflug convergence fix for strong radial
distortion plus sensor tilt rigs.

- Added a low-level tilt-aware Scheimpflug initializer in
  `vision-calibration-linear`.
- Wired the initializer into single-camera Scheimpflug auto init and rig-family
  Scheimpflug bootstrap.
- Retuned staged Scheimpflug refinement around bounded, finite cold-start
  behavior.
- Added a rig-handeye auto-recovery pass for nominally identical cameras: good
  per-camera solves define a shared nominal camera, and bad cameras are retried
  with deterministic nominal candidates scored against good-camera rig poses.
- Kept manual per-camera intrinsics, distortion, and sensor seeds authoritative.

## Files Changed

- `crates/vision-calibration-linear/src/scheimpflug_init.rs`
- `crates/vision-calibration-linear/src/lib.rs`
- `crates/vision-calibration/src/lib.rs`
- `crates/vision-calibration-pipeline/src/scheimpflug_intrinsics/steps.rs`
- `crates/vision-calibration-pipeline/src/rig_family.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/state.rs`
- `crates/vision-calibration-pipeline/src/rig_handeye/steps.rs`
- `crates/vision-calibration-pipeline/src/rig_extrinsics/state.rs`
- `crates/vision-calibration-pipeline/src/rig_extrinsics/steps.rs`
- `crates/vision-calibration-optim/src/problems/scheimpflug_intrinsics.rs`
- `crates/vision-calibration/tests/scheimpflug_intrinsics.rs`
- `docs/backlog.md`

## Validation Run

- `cargo test -p vision-calibration --test scheimpflug_intrinsics` — pass.
- `cargo test -p vision-calibration-linear scheimpflug` — pass.
- `cargo test -p vision-calibration-pipeline rig_family` — pass.
- `cargo test -p vision-calibration-pipeline --test rig_scheimpflug_handeye` — pass.
- `cargo test -p vision-calibration-pipeline --test rig_scheimpflug_extrinsics` — pass.
- `cargo test -p vision-calibration-optim scheimpflug` — pass.
- `RTV3D_REF_MAXITERS=60 cargo run --release --manifest-path crates/vision-calibration-examples-private/Cargo.toml --example rtv3d_ref_rig` — pass for the P6 per-camera intrinsics gate:
  - intrinsics BA per-camera reprojection:
    `[0.3802, 0.2677, 0.2833, 0.3522, 0.4725, 0.3268]`
  - final mean reprojection: `0.4057px`
  - recovered `tau_x`: all cameras within about `1.5°` of oracle.

## Follow-Ups / Remaining Risks

- The shared-nominal bad-camera retry currently lives in `rig_handeye::steps`.
  Factor it into `rig_family` before relying on the same recovery behavior in
  `rig_extrinsics`.
- The final joint hand-eye report still shows cam 0 at about `0.528px` despite
  all per-camera intrinsics solves being below `0.5px`; this is a final BA
  weighting/refinement issue, not the P6 cold-start failure.
- The linear initializer intentionally prioritizes basin selection. Strong
  distortion synthetic tests assert a few-pixel seed and correct tilt basin; BA
  remains responsible for final subpixel refinement.
