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

Eight synthetic views of an 8×6 board (40 mm pitch) calibrate a camera (the
synthetic data needs the `synthetic` feature of `vision-calibration-core`):

```rust
use vision_calibration_core::synthetic::{planar, poses};
use vision_calibration_core::{BrownConrady5, FxFyCxCy, PlanarDataset, View, make_pinhole_camera};
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsProblem, step_init, step_optimize,
};
use vision_calibration_pipeline::session::CalibrationSession;

fn main() -> anyhow::Result<()> {
    let truth = make_pinhole_camera(
        FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
        BrownConrady5 { k1: 0.05, k2: -0.02, ..BrownConrady5::default() },
    );
    let board = planar::grid_points(8, 6, 0.04);
    let specs = poses::seeded_board_pose_specs(8, 0.35, 3);
    let poses = poses::centered_board_poses(&specs, (0.14, 0.10), 0.6);
    let views = planar::project_views_all(&truth, &board, &poses)?;
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect())?;

    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // Linear initialization, then non-linear refinement.
    step_init(&mut session, None)?;
    step_optimize(&mut session, None)?;

    // Optional: checkpoint the session state as JSON.
    let checkpoint = session.to_json()?;
    assert!(!checkpoint.is_empty());

    let export = session.export()?;
    println!("fx = {:.2} px", export.camera.intrinsics.fx);
    Ok(())
}
```

## See Also

- [vision-calibration](https://crates.io/crates/vision-calibration): facade crate re-exporting this API
- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): math types and camera models
- [vision-calibration-linear](https://crates.io/crates/vision-calibration-linear): linear initialization solvers
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): non-linear optimization
- [Book: session API](https://vitalyvorobyev.github.io/calibration-rs/session.html)
