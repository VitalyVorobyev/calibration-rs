# calib

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
calib = { git = "https://github.com/VitalyVorobyev/calibration-rs" }
```

### Session API (Recommended for Standard Workflows)

```rust
use calib::session::{CalibrationSession, PlanarIntrinsicsProblem, PlanarIntrinsicsObservations};
use calib::pipeline::CorrespondenceView;

let views: Vec<CorrespondenceView> = /* load calibration data */;

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_observations(PlanarIntrinsicsObservations { views });

session.initialize(Default::default())?;
session.optimize(Default::default())?;
let report = session.export()?;
```

### Imperative API (For Custom Workflows)

```rust
use calib::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
use calib::linear::iterative_intrinsics::IterativeIntrinsicsOptions;

let views: Vec<CorrespondenceView> = /* load data */;

// Linear initialization
let init = initialize_planar_intrinsics(&views, &IterativeIntrinsicsOptions::default())?;
println!("Initial fx: {}", init.intrinsics.fx);

// Non-linear refinement
let result = optimize_planar_intrinsics_from_init(&views, &init, &Default::default(), &Default::default())?;
```

### Using the Prelude

```rust
use calib::prelude::*;

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
| General calibration tasks | `calib` (this crate) |
| Only need math types/camera models | `calib-core` |
| Only need linear initialization | `calib-linear` |
| Building custom optimization | `calib-optim` |
| Need pipeline + JSON I/O | `calib-pipeline` |

## Examples

See `examples/` directory:
- `handeyesingle.rs` - Hand-eye calibration workflow
- `handeye_session.rs` - Hand-eye using session API
- `stereo_session.rs` - Stereo rig calibration
- `rig_handeye_session.rs` - Multi-camera rig + hand-eye
- `rig_extrinsics_session.rs` - Rig extrinsics calibration

## See Also

- [Book](../../book/src/SUMMARY.md): Full documentation and tutorials
- [calib-core](../calib-core): Core primitives
- [calib-linear](../calib-linear): Linear solvers
- [calib-optim](../calib-optim): Non-linear optimization
- [calib-pipeline](../calib-pipeline): Pipelines and session API
