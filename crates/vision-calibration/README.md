# vision-calibration

High-level entry crate and facade for the `calibration-rs` toolbox.

This is the recommended crate for most users. It re-exports all sub-crates through a unified API.

## Features

- **Session API**: structured calibration workflows with step functions, state tracking, and JSON checkpointing
- **Seven workflows**: planar intrinsics (straight or tilted Scheimpflug sensor), single-camera hand-eye, laserline device, rig extrinsics, rig hand-eye, rig laserline device, rig hand-eye + laserline
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

## Problem Types

| Problem Type | Steps |
|---|---|
| `PlanarIntrinsicsProblem` (straight or tilted sensor) | `step_init` → `step_optimize` |
| `SingleCamHandeyeProblem` (any camera model) | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
| `LaserlineDeviceProblem` (any camera model) | `step_init` → `step_optimize` |
| `RigExtrinsicsProblem` (any camera model per camera) | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
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
| `planar_intrinsics` | Single-camera intrinsics (Zhang's method), for any camera model including a Scheimpflug sensor |
| `single_cam_handeye`, `laserline_device` | Hand-eye and camera + laser plane calibration, for any camera model |
| `rig_extrinsics`, `rig_handeye`, `rig_laserline_device`, `rig_handeye_laserline` | Multi-camera rig workflows |
| `device_seed`, `dataset_runner` | Device-spec seeding and manifest-driven dataset runs |
| `dataset`, `detect` | Dataset manifests and target detection |
| `analysis` | Reprojection-error analysis |
| `core`, `linear`, `optim` | Math types and camera models, closed-form solvers, non-linear optimization |
| `geometry`, `mvg` | Two-view geometry solvers; N-view MVG (bundle adjustment, rectification, dense stereo) |
| `synthetic` | Deterministic synthetic data generation |
| `prelude` | Convenient re-exports |

## Examples

Runnable examples (`cargo run --release -p vision-calibration --example <name>`)
live in [`examples/`](examples/); the book's workflow chapters walk through them.

## See Also

- [Book](https://vitalyvorobyev.github.io/calibration-rs)
- [API reference](https://vitalyvorobyev.github.io/calibration-rs/api/vision-calibration/index.html)
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): pipelines and session API
