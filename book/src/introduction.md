# Introduction

`calibration-rs` is a Rust toolbox for calibrating vision sensors and multi-camera rigs. It provides correct, modern algorithms with a clean separation between math primitives, linear initialization, non-linear refinement, and ready-to-use pipelines.

## What is Camera Calibration?

Camera calibration is the process of determining a camera's internal parameters (intrinsics like focal length and principal point) and lens distortion coefficients. These parameters are essential for:

- **3D reconstruction**: Converting pixel coordinates to real-world measurements
- **Visual odometry**: Tracking camera motion through space
- **Augmented reality**: Overlaying virtual objects on real scenes
- **Robot vision**: Enabling robots to interact with their environment
- **Multi-camera systems**: Relating measurements across multiple viewpoints

## Why Rust?

Rust offers compelling advantages for calibration software:

- **Performance**: Zero-cost abstractions and no garbage collection enable real-time applications
- **Correctness**: Strong type system catches errors at compile time
- **Composability**: Traits and generics allow flexible camera model composition
- **Determinism**: No hidden allocations or unpredictable pauses
- **Safety**: Memory safety without sacrificing performance

## Architecture Overview

The library is organized as a workspace of six crates:

```
                           ┌─────────────────────────┐
                           │         calib           │  ◄── Stable API facade
                           │    (public interface)   │
                           └───────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
              ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   calib-pipeline    │  │    calib-optim      │  │    calib-linear     │
│  Session API, JSON  │  │   Non-linear BA     │  │   Linear solvers    │
│   I/O, workflows    │  │   LM optimization   │  │   Initialization    │
└─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────┐
                       │     calib-core      │  ◄── Math types, camera
                       │   Types, models,    │      models, RANSAC
                       │       RANSAC        │
                       └─────────────────────┘
```

### Crate Responsibilities

| Crate | Purpose | When to Use |
|-------|---------|-------------|
| **calib** | Stable facade | Default choice for most users |
| **calib-core** | Types and models | Need camera models or RANSAC only |
| **calib-linear** | Initialization | Building custom pipelines |
| **calib-optim** | Optimization | Custom non-linear problems |
| **calib-pipeline** | Workflows | Session API or JSON I/O |

## Choosing Your Entry Point

### Session API (Recommended for Standard Workflows)

Use the session API when you want:
- Automatic state management
- JSON checkpointing for long-running calibrations
- Type-safe stage transitions

```rust
use calib::session::{CalibrationSession, PlanarIntrinsicsProblem};

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_observations(observations);
session.initialize(Default::default())?;
session.optimize(Default::default())?;
let report = session.export()?;
```

### Imperative Functions (For Custom Workflows)

Use imperative functions when you need:
- Full control over each step
- Inspection of intermediate results
- Custom composition of algorithms

```rust
use calib::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};

let init = initialize_planar_intrinsics(&views, &opts)?;
println!("Initial estimate: fx={}", init.intrinsics.fx);

// Decide whether to proceed based on initialization quality
let result = optimize_planar_intrinsics_from_init(&views, &init, &solve_opts, &backend_opts)?;
```

### Direct Access (For Advanced Users)

Access individual algorithms directly:

```rust
use calib::linear::homography::dlt_homography;
use calib::optim::planar_intrinsics::optimize_planar_intrinsics;
```

## Supported Problems

| Problem | Description | Session Type |
|---------|-------------|--------------|
| Planar intrinsics | Single camera calibration from planar target | `PlanarIntrinsicsProblem` |
| Hand-eye | Robot + camera calibration | `HandEyeSingleProblem` |
| Rig extrinsics | Multi-camera rig calibration | `RigExtrinsicsProblem` |
| Rig hand-eye | Robot + multi-camera rig | `RigHandEyeProblem` |
| Laserline device | Laser plane + single camera | `LaserlineDeviceProblem` |

## What's Next

- **[Quickstart](quickstart.md)**: Get running with copy-paste examples
- **[Core Concepts](concepts.md)**: Understand coordinate conventions and camera models
- **[Linear Calibration](linear.md)**: Learn about initialization algorithms
- **[Non-linear Optimization](nonlinear.md)**: Configure the optimization backend
- **[Pipelines](pipeline.md)**: Use the session API for structured workflows
