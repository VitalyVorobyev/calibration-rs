# vision-calibration

High-level entry crate and facade for the `calibration-rs` toolbox.

This is the recommended crate for most users. It re-exports all sub-crates through a unified API.

## Features

- **Session API**: Structured calibration workflows with step functions, state tracking, and JSON checkpointing
- **5 problem types**: Planar intrinsics, single-camera hand-eye, rig extrinsics, rig hand-eye, laserline device
- **Prelude module**: Quick-start imports for common use cases
- **Foundation access**: Direct access to core types, linear solvers, and optimization when needed

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
vision-calibration = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
```

### Planar Intrinsics Calibration

```rust,no_run
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
# let dataset: PlanarDataset = unimplemented!();
session.set_input(dataset)?;

step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

let result = session.export()?;
```

### Single-Camera Hand-Eye Calibration

```rust,no_run
use vision_calibration::prelude::*;
use vision_calibration::single_cam_handeye::{
    step_intrinsics_init, step_intrinsics_optimize,
    step_handeye_init, step_handeye_optimize,
};

let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
# let input = unimplemented!();
session.set_input(input)?;

step_intrinsics_init(&mut session, None)?;
step_intrinsics_optimize(&mut session, None)?;
step_handeye_init(&mut session, None)?;
step_handeye_optimize(&mut session, None)?;

let result = session.export()?;
```

### Using the Prelude

```rust,no_run
use vision_calibration::prelude::*;

// Gives you CalibrationSession, all problem types,
// pipeline functions, core types, and common options.
```

## Available Problem Types

| Problem Type | Steps |
|---|---|
| `PlanarIntrinsicsProblem` | `step_init` → `step_optimize` |
| `SingleCamHandeyeProblem` | `step_intrinsics_init` → `step_intrinsics_optimize` → `step_handeye_init` → `step_handeye_optimize` |
| `RigExtrinsicsProblem` | `step_intrinsics_init_all` → `step_intrinsics_optimize_all` → `step_rig_init` → `step_rig_optimize` |
| `RigHandeyeProblem` | All 6 steps (intrinsics + rig + hand-eye) |
| `LaserlineDeviceProblem` | `step_init` → `step_optimize` |

Each problem type also provides a `run_calibration` convenience function that runs all steps.

## Module Organization

| Module | Description |
|--------|-------------|
| `session` | Calibration session framework (`CalibrationSession`, `ProblemType`) |
| `planar_intrinsics` | Single-camera intrinsics (Zhang's method) |
| `single_cam_handeye` | Single camera + hand-eye calibration |
| `rig_extrinsics` | Multi-camera rig extrinsics |
| `rig_handeye` | Multi-camera rig + hand-eye |
| `laserline_device` | Camera + laser plane device |
| `core` | Math types, camera models, RANSAC |
| `linear` | Closed-form initialization algorithms |
| `optim` | Non-linear optimization |
| `synthetic` | Deterministic synthetic data generation |
| `prelude` | Convenient re-exports |

## When to Use This Crate vs. Sub-Crates

| Use Case | Recommended |
|----------|-------------|
| General calibration tasks | `vision-calibration` (this crate) |
| Only need math types/camera models | `vision-calibration-core` |
| Only need linear initialization | `vision-calibration-linear` |
| Building custom optimization | `vision-calibration-optim` |
| Need pipeline + JSON I/O | `vision-calibration-pipeline` |

## Examples

See `examples/` directory:

| Example | Problem Type | Data |
|---------|---|---|
| `planar_synthetic` | Planar intrinsics | Synthetic |
| `planar_real` | Planar intrinsics | Real stereo images |
| `stereo_session` | Rig extrinsics | Real stereo images |
| `handeye_synthetic` | Single-camera hand-eye | Synthetic |
| `handeye_session` | Single-camera hand-eye | KUKA robot data |
| `rig_handeye_synthetic` | Rig hand-eye | Synthetic |
| `laserline_device_session` | Laserline device | Session API demo |

## See Also

- [vision-calibration-core](../vision-calibration-core): Core primitives
- [vision-calibration-linear](../vision-calibration-linear): Linear solvers
- [vision-calibration-optim](../vision-calibration-optim): Non-linear optimization
- [vision-calibration-pipeline](../vision-calibration-pipeline): Pipelines and session API
