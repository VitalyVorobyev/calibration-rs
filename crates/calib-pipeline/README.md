# calib-pipeline

End-to-end calibration workflows and session APIs for `calibration-rs`.

This crate provides two complementary approaches for camera calibration:
1. **Session API**: Structured workflows with state tracking and JSON checkpointing
2. **Imperative Functions**: Direct access to building blocks for custom workflows

## Features

- **Planar intrinsics calibration**: Zhang's method with Brown-Conrady distortion
- **Hand-eye calibration**: Single-camera and multi-camera rig setups
- **Rig extrinsics**: Multi-camera rig calibration
- **Linescan calibration**: Laser plane + camera intrinsics joint optimization
- **JSON I/O**: Serialize/deserialize all inputs, configs, and reports
- **Checkpointing**: Save and resume calibration sessions

## Dual API Design

### Session API

Best for standard calibration workflows with automatic state management:

```rust
use calib_pipeline::session::{CalibrationSession, PlanarIntrinsicsProblem};
use calib_pipeline::session::problem_types::{
    PlanarIntrinsicsObservations, PlanarIntrinsicsInitOptions, PlanarIntrinsicsOptimOptions
};

// Create session
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

// Set observations
session.set_observations(PlanarIntrinsicsObservations { views });

// Initialize (linear solver)
session.initialize(PlanarIntrinsicsInitOptions::default())?;

// Save checkpoint
let checkpoint = session.to_json()?;

// Optimize (non-linear refinement)
session.optimize(PlanarIntrinsicsOptimOptions::default())?;

// Export results
let report = session.export()?;
```

### Imperative Functions API

Best for custom workflows requiring intermediate inspection:

```rust
use calib_pipeline::{
    homography, zhang_intrinsics, planar_init_seed_from_views,
    optimize_planar_intrinsics_raw, PlanarDataset, BackendSolveOptions,
};

// Compute homographies directly
let H = homography::dlt_homography(&board_2d, &pixel_2d)?;

// Initialize intrinsics
let (init, camera) = planar_init_seed_from_views(&views)?;

// Inspect linear initialization
println!("Initial fx: {}", init.intrinsics.fx);

// Decide whether to continue based on quality
if init.intrinsics.fx > 500.0 && init.intrinsics.fx < 2000.0 {
    let dataset = PlanarDataset::new(views)?;
    let result = optimize_planar_intrinsics_raw(dataset, init, opts, backend_opts)?;
}
```

## Available Problem Types

| Problem | Session Type | Use Case |
|---------|--------------|----------|
| Planar intrinsics | `PlanarIntrinsicsProblem` | Single camera calibration |
| Hand-eye single | `HandEyeSingleProblem` | Robot + camera calibration |
| Rig extrinsics | `RigExtrinsicsProblem` | Multi-camera rig |
| Rig hand-eye | `RigHandEyeProblem` | Robot + multi-camera rig |
| Linescan | `LinescanProblem` | Laser line + camera |

## Re-exported Modules

For custom workflows, access building blocks directly:

**From calib-linear:**
- `homography` - DLT homography estimation + RANSAC
- `zhang_intrinsics` - Intrinsics from homographies
- `pnp` - Perspective-n-Point solvers (DLT, P3P, EPnP)
- `epipolar` - Fundamental/essential matrices
- `triangulation` - Linear triangulation
- `handeye_linear` - Hand-eye initialization
- `linescan` - Laser plane estimation

**From calib-optim:**
- `BackendSolveOptions` - Solver configuration
- `PlanarDataset`, `PlanarIntrinsicsInit` - Optimization inputs
- `RobustLoss` - Huber, Cauchy, Arctan

## JSON I/O Example

```rust
use calib_pipeline::{PlanarIntrinsicsInput, PlanarIntrinsicsConfig, run_planar_intrinsics};

// Load from JSON
let input: PlanarIntrinsicsInput = serde_json::from_str(&input_json)?;
let config: PlanarIntrinsicsConfig = serde_json::from_str(&config_json)?;

// Run calibration
let report = run_planar_intrinsics(&input, &config)?;

// Save results
let output_json = serde_json::to_string_pretty(&report)?;
```

## When to Use Each API

| Scenario | Recommended API |
|----------|----------------|
| Standard single-camera calibration | Session |
| Need checkpointing between stages | Session |
| Inspect linear initialization quality | Imperative |
| Custom multi-step workflow | Imperative |
| Integration into larger system | Imperative |
| Research and experimentation | Imperative |

## Examples

See `examples/` directory:
- `session_basic.rs` - Session API lifecycle
- `custom_workflow.rs` - Imperative functions API
- `compare_apis.rs` - Side-by-side comparison

## See Also

- [calib-core](../calib-core): Math types and camera models
- [calib-linear](../calib-linear): Linear initialization solvers
- [calib-optim](../calib-optim): Non-linear optimization
- [Book: Pipelines](../../book/src/pipeline.md)
- [functions.md](src/functions.md): Comprehensive API reference
