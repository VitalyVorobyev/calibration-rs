# vision-calibration

High-level entry crate and facade for the `calibration-rs` toolbox.

This is the recommended crate for most users. It re-exports all sub-crates through a unified API.

## Features

- **Session API**: structured calibration workflows with step functions, state tracking, and JSON checkpointing
- **Eight workflows**: planar intrinsics, Scheimpflug intrinsics, single-camera hand-eye, laserline device, rig extrinsics, rig hand-eye, rig laserline device, rig hand-eye + laserline
- **Prelude module**: minimal imports for planar "hello world" calibration
- **Foundation access**: core types, linear solvers, optimization, two-view geometry and MVG when needed

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
vision-calibration = "0.9"
```

### Planar Intrinsics Calibration

```rust,no_run
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

# fn main() -> anyhow::Result<()> {
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
# let dataset: PlanarDataset = unimplemented!();
session.set_input(dataset)?;

step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

let result = session.export()?;
# Ok(())
# }
```

### Single-Camera Hand-Eye Calibration

```rust,no_run
use vision_calibration::session::CalibrationSession;
use vision_calibration::single_cam_handeye::{
    SingleCamHandeyeProblem,
    step_intrinsics_init, step_intrinsics_optimize,
    step_handeye_init, step_handeye_optimize,
};

# fn main() -> anyhow::Result<()> {
let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
# let input = unimplemented!();
session.set_input(input)?;

step_intrinsics_init(&mut session, None)?;
step_intrinsics_optimize(&mut session, None)?;
step_handeye_init(&mut session, None)?;
step_handeye_optimize(&mut session, None)?;

let result = session.export()?;
# Ok(())
# }
```

### Scheimpflug Intrinsics Calibration

```rust,no_run
use vision_calibration::core::PlanarDataset;
use vision_calibration::session::CalibrationSession;
use vision_calibration::scheimpflug_intrinsics::{
    ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsProblem, run_calibration,
};

# fn main() -> anyhow::Result<()> {
# let dataset: PlanarDataset = unimplemented!();
let mut session = CalibrationSession::<ScheimpflugIntrinsicsProblem>::new();
session.set_input(dataset)?;

let config = ScheimpflugIntrinsicsConfig::default();
run_calibration(&mut session, Some(config))?;
let result = session.export()?;
println!("mean reprojection error: {:.4}", result.mean_reproj_error);
# Ok(())
# }
```

## Problem Types

| Problem Type | Steps |
|---|---|
| `PlanarIntrinsicsProblem` | `step_init` → `step_optimize` |
| `ScheimpflugIntrinsicsProblem` | `step_init` → `step_optimize` |
| `SingleCamHandeyeProblem` | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
| `LaserlineDeviceProblem` | `step_init` → `step_optimize` |
| `RigExtrinsicsProblem` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
| `RigHandeyeProblem` | rig extrinsics steps → `step_handeye_init` → `step_handeye_optimize` |
| `RigLaserlineDeviceProblem` | `step_init` → `step_optimize` (frozen rig hand-eye export as input) |
| `RigHandeyeLaserlineProblem` | `run_calibration` (joint solve) |

Each problem type also provides a `run_calibration` convenience function that
runs all steps. The per-module documentation lists the exact step names and
options.

## Module Organization

| Module | Description |
|--------|-------------|
| `session`, `common` | Session framework (`CalibrationSession`, `ProblemType`) and shared step/config types |
| `planar_intrinsics`, `scheimpflug_intrinsics` | Single-camera intrinsics (Zhang's method; Scheimpflug tilt) |
| `single_cam_handeye`, `laserline_device` | Hand-eye and camera + laser plane calibration |
| `rig_extrinsics`, `rig_handeye`, `rig_laserline_device`, `rig_handeye_laserline` | Multi-camera rig workflows |
| `device_seed`, `dataset_runner` | Device-spec seeding and manifest-driven dataset runs |
| `dataset`, `detect` | Dataset manifests and target detection |
| `analysis` | Reprojection-error analysis |
| `core`, `linear`, `optim` | Math types and camera models, closed-form solvers, non-linear optimization |
| `geometry`, `mvg` | Two-view geometry solvers; N-view MVG (bundle adjustment, rectification, dense stereo) |
| `synthetic` | Deterministic synthetic data generation |
| `prelude` | Convenient re-exports |

## Examples

Run from a repository checkout with `cargo run --release -p vision-calibration --example <name>`.
Examples marked "committed data" read datasets under `data/` in the repository.

| Example | Workflow | Data |
|---------|---|---|
| `planar_synthetic` | Planar intrinsics | Synthetic |
| `planar_synthetic_with_images` | Planar intrinsics + image manifest | Synthetic |
| `planar_real` | Planar intrinsics | Committed data (`data/stereo`) |
| `stereo_session` | Rig extrinsics | Committed data (`data/stereo`) |
| `stereo_charuco_session` | Rig extrinsics | Committed data (`data/stereo_charuco`) |
| `manual_init_proof` | Rig extrinsics, manual init | Committed data (`data/stereo_charuco`) |
| `handeye_synthetic` | Single-camera hand-eye | Synthetic |
| `handeye_session` | Single-camera hand-eye | Committed data (`data/kuka_1`) |
| `rig_handeye_synthetic` | Rig hand-eye | Synthetic |
| `laserline_device_session` | Laserline device | Synthetic |
| `mvg_two_view` | Two-view MVG | Synthetic |
| `dense_stereo_real` | Dense stereo | Committed data (`data/stereo`) |
| `viewer_fixtures` | Viewer export generation | Committed data |

## See Also

- [Book](https://vitalyvorobyev.github.io/calibration-rs)
- [API reference](https://vitalyvorobyev.github.io/calibration-rs/api/vision-calibration/index.html)
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): pipelines and session API
