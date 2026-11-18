# vision-calibration-pipeline

End-to-end calibration workflows and a session API for `calibration-rs`.

Most users want the facade crate
[`vision-calibration`](https://crates.io/crates/vision-calibration) instead:
it re-exports these workflows next to the camera models, target detection,
dataset manifests and multiple-view geometry, and its module paths are the
stable API. This crate is one of its building blocks.

Each workflow is a `CalibrationSession<P>` over a problem type `P`, driven by
free step functions. Sessions hold input, config, intermediate state and
export, and can be checkpointed to JSON (`to_json` / `from_json`).

## Workflows

| Module | Problem type |
|---|---|
| `planar_intrinsics` | `PlanarIntrinsicsProblem` (any camera model, including a Scheimpflug sensor) |
| `single_cam_handeye` | `SingleCamHandeyeProblem` (any camera model) |
| `laserline_device` | `LaserlineDeviceProblem` (any camera model) |
| `rig_extrinsics` | `RigExtrinsicsProblem` |
| `rig_handeye` | `RigHandeyeProblem` |
| `rig_laserline_device` | `RigLaserlineDeviceProblem` |
| `rig_handeye_laserline` | `RigHandeyeLaserlineProblem` |

Each module also provides a `run_calibration` function that runs all steps.

## Session API

The `vision-calibration` crate documentation opens with a session calibrating
a camera on synthetic data, and the book's
[Getting Started](https://vitalyvorobyev.github.io/calibration-rs/getting_started.html)
walks through it in Rust and Python.

## See Also

- [vision-calibration](https://crates.io/crates/vision-calibration): facade crate re-exporting this API
- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): math types and camera models
- [vision-calibration-linear](https://crates.io/crates/vision-calibration-linear): linear initialization solvers
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): non-linear optimization
- [Book: session API](https://vitalyvorobyev.github.io/calibration-rs/session.html)
