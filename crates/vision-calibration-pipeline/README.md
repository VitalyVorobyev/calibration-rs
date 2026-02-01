# vision-calibration-pipeline

End-to-end calibration workflows and a session API for `calibration-rs`.

This crate provides two complementary approaches for camera calibration:
1. **Session API**: Structured workflows with artifact management + JSON checkpointing
2. **Imperative Functions**: Direct access to pipeline functions for custom workflows

## Features

- **Planar intrinsics calibration**: Zhang's method with Brown-Conrady distortion
- **JSON I/O**: Serialize/deserialize inputs, configs, and results
- **Checkpointing**: Save and resume session state

## Dual API Design

### Session API

Best for standard workflows with branching, artifact tracking, and checkpointing:

```rust
use vision_calibration_pipeline::planar_intrinsics::{PlanarIntrinsicsConfig, PlanarIntrinsicsProblem};
use vision_calibration_pipeline::session::{CalibrationSession, ExportOptions, FilterOptions};
use vision_calibration_pipeline::PlanarDataset;

// Create session
let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();

// Add observations (a PlanarDataset: Vec<View<NoMeta>>)
let obs_id = session.add_observations(dataset);

// Initialize (linear solver)
let config = PlanarIntrinsicsConfig::default();
let init_id = session.run_init(obs_id, config.clone())?;

// Save checkpoint
let checkpoint = session.to_json()?;

// Optimize (non-linear refinement)
let result_id = session.run_optimize(obs_id, init_id, config)?;

// Filter outliers and re-optimize (optional)
let obs_filtered = session.run_filter_obs(obs_id, result_id, FilterOptions::default())?;

// Export results
let estimate = session.run_export(result_id, ExportOptions::default())?;
```

### Imperative Functions API

Best for custom workflows requiring intermediate inspection:

```rust
use vision_calibration_pipeline::planar_intrinsics::{
    planar_init_seed_from_views, run_planar_intrinsics, PlanarIntrinsicsConfig,
};
use vision_calibration_pipeline::PlanarDataset;

let config = PlanarIntrinsicsConfig::default();
let seed = planar_init_seed_from_views(&dataset, config.init_opts.clone())?;
let estimate = run_planar_intrinsics(&dataset, &config)?;
```

## JSON I/O Example

```rust
use vision_calibration_pipeline::{PlanarDataset, PlanarIntrinsicsConfig, run_planar_intrinsics};

// Load from JSON
let input: PlanarDataset = serde_json::from_str(&input_json)?;
let config: PlanarIntrinsicsConfig = serde_json::from_str(&config_json)?;

// Run calibration
let estimate = run_planar_intrinsics(&input, &config)?;

// Save results
let output_json = serde_json::to_string_pretty(&estimate)?;
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

## See Also

- [vision-calibration-core](../vision-calibration-core): Math types and camera models
- [vision-calibration-linear](../vision-calibration-linear): Linear initialization solvers
- [vision-calibration-optim](../vision-calibration-optim): Non-linear optimization
- [Book: Pipelines](../../book/src/pipeline.md)
- [functions.md](src/functions.md): Comprehensive API reference
