# vision-calibration-pipeline

End-to-end calibration workflows and a session API for `calibration-rs`.

Each workflow is a `CalibrationSession<P>` over a problem type `P`, driven by
free step functions. Sessions hold input, config, intermediate state and
export, and can be checkpointed to JSON (`to_json` / `from_json`).

## Workflows

| Module | Problem type |
|---|---|
| `planar_intrinsics` | `PlanarIntrinsicsProblem` |
| `scheimpflug_intrinsics` | `ScheimpflugIntrinsicsProblem` |
| `single_cam_handeye` | `SingleCamHandeyeProblem` |
| `laserline_device` | `LaserlineDeviceProblem` |
| `rig_extrinsics` | `RigExtrinsicsProblem` |
| `rig_handeye` | `RigHandeyeProblem` |
| `rig_laserline_device` | `RigLaserlineDeviceProblem` |
| `rig_handeye_laserline` | `RigHandeyeLaserlineProblem` |

Each module also provides a `run_calibration` function that runs all steps.

## Session API

```rust,no_run
use vision_calibration_pipeline::session::CalibrationSession;
use vision_calibration_pipeline::planar_intrinsics::{
    PlanarIntrinsicsProblem, step_init, step_optimize,
};
use vision_calibration_core::PlanarDataset;

# fn main() -> anyhow::Result<()> {
# let dataset: PlanarDataset = unimplemented!();
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;

// Linear initialization, then non-linear refinement.
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Optional: checkpoint the session state as JSON.
let checkpoint = session.to_json()?;

let export = session.export()?;
# let _ = (checkpoint, export);
# Ok(())
# }
```

## See Also

- [vision-calibration](https://crates.io/crates/vision-calibration): facade crate re-exporting this API
- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): math types and camera models
- [vision-calibration-linear](https://crates.io/crates/vision-calibration-linear): linear initialization solvers
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): non-linear optimization
- [Book: session API](https://vitalyvorobyev.github.io/calibration-rs/session.html)
