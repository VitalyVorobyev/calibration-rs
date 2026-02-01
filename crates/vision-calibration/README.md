# vision-calibration

High-level entry crate and facade for the `calibration-rs` toolbox.

This is the recommended crate for most users. It re-exports all sub-crates through a stable, unified API while hiding internal implementation details.

## Features

- **Stable API surface**: Public interface designed for compatibility
- **Dual API design**: Session API for structured workflows, imperative API for custom workflows
- **All problem types**: Planar intrinsics, hand-eye, rig extrinsics, laserline
- **Prelude module**: Quick-start imports for common use cases

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
vision-calibration = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
```

### Session API (Recommended for Standard Workflows)

```rust
use vision_calibration::session::{CalibrationSession, PlanarIntrinsicsProblem, PlanarIntrinsicsObservations};
use vision_calibration::pipeline::CorrespondenceView;

let views: Vec<CorrespondenceView> = /* load calibration data */;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_observations(PlanarIntrinsicsObservations { views });

session.initialize(Default::default())?;
session.optimize(Default::default())?;
let report = session.export()?;
```

### Imperative API (For Custom Workflows)

```rust
use vision_calibration::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
use vision_calibration::linear::iterative_intrinsics::IterativeIntrinsicsOptions;

let views: Vec<CorrespondenceView> = /* load data */;

// Linear initialization
let init = initialize_planar_intrinsics(&views, &IterativeIntrinsicsOptions::default())?;
println!("Initial fx: {}", init.intrinsics.fx);

// Non-linear refinement
let result = optimize_planar_intrinsics_from_init(&views, &init, &Default::default(), &Default::default())?;
```

### Using the Prelude

```rust
use vision_calibration::prelude::*;

// Now you have access to common types:
// Camera, Pt3, Vec2, CalibrationSession, PlanarIntrinsicsProblem, etc.
```

## Module Organization

| Module | Description |
|--------|-------------|
| `session` | Type-safe calibration session framework |
| `helpers` | Granular helper functions for custom workflows |
| `pipeline` | All-in-one convenience functions |
| `core` | Math types, camera models, RANSAC |
| `linear` | Closed-form initialization algorithms |
| `optim` | Non-linear optimization problems |
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
- `handeyesingle.rs` - Hand-eye calibration workflow
- `handeye_session.rs` - Hand-eye using session API
- `stereo_session.rs` - Stereo rig calibration
- `rig_handeye_session.rs` - Multi-camera rig + hand-eye
- `rig_extrinsics_session.rs` - Rig extrinsics calibration

## See Also

- [Book](../../book/src/SUMMARY.md): Full documentation and tutorials
- [vision-calibration-core](../vision-calibration-core): Core primitives
- [vision-calibration-linear](../vision-calibration-linear): Linear solvers
- [vision-calibration-optim](../vision-calibration-optim): Non-linear optimization
- [vision-calibration-pipeline](../vision-calibration-pipeline): Pipelines and session API
