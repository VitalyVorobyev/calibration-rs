# Architecture Overview

calibration-rs is organized as a layered workspace of five Rust crates. This chapter explains the dependency structure, data flow, and key design patterns.

## Crate Dependency Graph

```
vision-calibration          (facade: re-exports everything)
    │
    └── vision-calibration-pipeline    (sessions, workflows, step functions)
            │
            ├── vision-calibration-optim   (non-linear optimization)
            │       │
            │       └── vision-calibration-core
            │
            └── vision-calibration-linear  (closed-form initialization)
                    │
                    └── vision-calibration-core
```

**Key rule**: `vision-calibration-linear` and `vision-calibration-optim` are peers. They both depend on `vision-calibration-core` but never on each other. This keeps initialization algorithms free of optimization dependencies and vice versa.

## Crate Responsibilities

### vision-calibration-core

The foundation layer providing:

- **Math types** — nalgebra-based aliases: `Pt2`, `Pt3`, `Vec3`, `Mat3`, `Iso3`, `Real` (= `f64`)
- **Camera models** — composable trait-based pipeline: `ProjectionModel`, `DistortionModel`, `SensorModel`, `IntrinsicsModel`
- **Observation types** — `CorrespondenceView` (2D-3D point pairs), `View<Meta>`, `PlanarDataset`, `RigDataset`
- **RANSAC engine** — generic `Estimator` trait with configurable options
- **Synthetic data utilities** — grid generation, pose sampling, projection
- **Reprojection error computation** — single-camera and multi-camera rig

### vision-calibration-linear

Closed-form initialization solvers. Each produces an approximate estimate suitable for seeding non-linear optimization:

| Solver | Input | Output |
|--------|-------|--------|
| Zhang's method | Homographies | Intrinsics $K$ |
| Distortion fit | $K$ + homographies | Brown-Conrady coefficients |
| Iterative intrinsics | Observations | Joint $K$ + distortion |
| Homography DLT | 2D-2D correspondences | $3 \times 3$ homography |
| Planar pose | $K$ + homography | SE(3) pose |
| P3P / DLT PnP | 3D-2D + $K$ | SE(3) pose |
| 5-point essential | Normalized correspondences | Essential matrix |
| 8-/7-point fundamental | Pixel correspondences | Fundamental matrix |
| Tsai-Lenz hand-eye | Robot + camera motions | Hand-eye SE(3) |
| Rig extrinsics | Per-camera poses | Camera-to-rig SE(3) |
| Laser plane | Laser pixels + target poses | Plane (normal + distance) |

### vision-calibration-optim

Non-linear refinement with a backend-agnostic architecture:

1. **IR (Intermediate Representation)** — `ProblemIR` with `ParamBlock` and `ResidualBlock` types that describe optimization problems independently of any solver
2. **Factors** — generic residual functions parameterized over `RealField` for automatic differentiation
3. **Backends** — currently `TinySolverBackend` (Levenberg-Marquardt with sparse linear solvers)
4. **Problem builders** — domain-specific functions that construct IR from calibration data

### vision-calibration-pipeline

The session framework providing production-ready workflows:

- `CalibrationSession<P: ProblemType>` — generic state container with config, input, state, output, exports
- **Step functions** — free functions operating on `&mut CalibrationSession<P>` (e.g., `step_init`, `step_optimize`)
- **Pipeline functions** — convenience wrappers chaining all steps
- **JSON checkpointing** — full serialization for session persistence
- Five problem types: `PlanarIntrinsicsProblem`, `SingleCamHandeyeProblem`, `RigExtrinsicsProblem`, `RigHandeyeProblem`, `LaserlineDeviceProblem`

### vision-calibration

Unified facade crate that re-exports everything through a clean module hierarchy:

```rust
use vision_calibration::prelude::*;           // Common imports
use vision_calibration::planar_intrinsics::*;  // Planar workflow
use vision_calibration::core::*;               // Math types
use vision_calibration::linear::*;             // Linear solvers
use vision_calibration::optim::*;              // Optimization
```

## Data Flow

Every calibration workflow follows the same pattern:

```
Observations (2D-3D correspondences)
    │
    ▼
Linear Initialization (vision-calibration-linear)
    │  Closed-form solvers: ~5-40% accuracy
    ▼
Non-Linear Refinement (vision-calibration-optim)
    │  Levenberg-Marquardt: <2% accuracy, <1 px reprojection
    ▼
Calibrated Parameters (K, distortion, poses, ...)
```

The session framework wraps this flow with configuration, state tracking, and checkpointing:

```
CalibrationSession::new()
    │
    ▼
session.set_input(dataset)
    │
    ▼
step_init(&mut session)      ← linear initialization
    │
    ▼
step_optimize(&mut session)  ← non-linear refinement
    │
    ▼
session.export()             ← calibrated parameters
```

## Design Patterns

### Composable Camera Model

The camera projection pipeline is built from four independent traits composed via generics:

```
pixel = K(sensor(distortion(projection(direction))))
```

Each stage can be mixed and matched. For example, a standard camera uses `Pinhole` + `BrownConrady5` + `IdentitySensor` + `FxFyCxCySkew`, while a laser profiler might use `Pinhole` + `BrownConrady5` + `ScheimpflugParams` + `FxFyCxCySkew`.

### Backend-Agnostic Optimization

Problems are defined as an intermediate representation (IR) that is independent of any specific solver. The IR is then *compiled* to a solver-specific form:

```
Problem Builder  →  ProblemIR  →  Backend.compile()  →  Backend.solve()
                  (generic)       (solver-specific)
```

This allows swapping the optimization backend without changing problem definitions.

### Step Functions

Calibration workflows are decomposed into discrete steps implemented as free functions. This allows:

- **Inspection** of intermediate state between steps
- **Resumption** from any point (via JSON checkpointing)
- **Customization** of per-step options
- **Composition** of steps from different problem types
