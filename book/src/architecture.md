# Architecture Overview

calibration-rs is organized as a layered workspace of Rust crates plus Python bindings. This chapter explains the dependency structure, data flow, and key design patterns.

## Crate Dependency Graph

```
vision-calibration (facade: re-exports everything)
    │
    ├── vision-calibration-pipeline   (sessions, problem types, dataset runner)
    │       ├── vision-calibration-optim     (non-linear optimization)
    │       ├── vision-calibration-linear    (closed-form initialization)
    │       ├── vision-calibration-dataset   (dataset manifest)
    │       ├── vision-calibration-detect    (target detectors, detection cache)
    │       └── vision-geometry
    │
    └── vision-mvg                    (N-view geometry, rectification, dense stereo)
            └── vision-geometry       (two-view solvers)

vision-calibration-core is the base layer: linear, optim, vision-geometry
and vision-mvg all depend on it.
```

**Key rule**: `vision-calibration-linear` and `vision-calibration-optim` are peers. They both depend on `vision-calibration-core` but never on each other. This keeps initialization algorithms free of optimization dependencies and vice versa. `vision-calibration-linear` may also use `vision-geometry`; `vision-calibration-pipeline` does not depend on `vision-mvg` (the facade re-exports it directly).

## Crate Responsibilities

### vision-calibration-core

The foundation layer providing:

- **Math types** — nalgebra-based aliases: `Pt2`, `Pt3`, `Vec3`, `Mat3`, `Iso3`, `Real` (= `f64`)
- **Camera models** — composable trait-based pipeline: `ProjectionModel`, `DistortionModel`, `SensorModel`, `IntrinsicsModel`
- **Observation types** — `CorrespondenceView` (2D-3D point pairs), `View<Meta>`, `PlanarDataset`, `RigDataset`
- **RANSAC engine** — generic `Estimator` trait with configurable options
- **Synthetic data utilities** — grid generation, pose sampling, projection
- **Reprojection error computation** — single-camera and multi-camera rig

### vision-geometry

Deterministic two-view solvers, all free functions in per-topic modules:

| Module | Functions |
|--------|-----------|
| `homography` | `dlt_homography`, `dlt_homography_ransac` |
| `epipolar` | `fundamental_8point`, `essential_5point` |
| `triangulation` | `triangulate_point_linear`, `triangulate_point` |
| `camera_matrix` | `dlt_camera_matrix` |

Through the facade they are reached as `vision_calibration::geometry::homography::dlt_homography`, and so on.

### vision-mvg

N-view geometry on top of `vision-geometry`: calibrated relative-pose recovery, N-view triangulation, bundle adjustment (with the `refine` feature), Scheimpflug-aware stereo rectification, and dense stereo matching. Facade path: `vision_calibration::mvg`.

### vision-calibration-dataset and vision-calibration-detect

`vision-calibration-dataset` defines `DatasetSpec`, the on-disk manifest describing images, robot poses, and target metadata. `vision-calibration-detect` provides the target detectors (chessboard, ChArUco, PuzzleBoard, ring grid) and a filesystem detection cache. The pipeline's `dataset_runner` turns a manifest into calibration inputs.

### vision-calibration-linear

Closed-form initialization solvers (the two-view solvers live in `vision-geometry`). Each produces an approximate estimate suitable for seeding non-linear optimization:

| Solver | Input | Output |
|--------|-------|--------|
| Zhang's method | Homographies | Intrinsics $K$ |
| Distortion fit | $K$ + homographies | Brown-Conrady coefficients |
| Iterative intrinsics | Observations | Joint $K$ + distortion |
| Planar pose | $K$ + homography | SE(3) pose |
| P3P / DLT PnP | 3D-2D + $K$ | SE(3) pose |
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
- Eight problem types: `PlanarIntrinsicsProblem`, `ScheimpflugIntrinsicsProblem`, `SingleCamHandeyeProblem`, `LaserlineDeviceProblem`, `RigExtrinsicsProblem`, `RigHandeyeProblem`, `RigLaserlineDeviceProblem`, `RigHandeyeLaserlineProblem`. The two rig problems cover pinhole and Scheimpflug rigs via `SensorMode`.

### vision-calibration

Unified facade crate that re-exports everything through a clean module hierarchy:

```rust
use vision_calibration::prelude::*;            // Minimal planar hello-world imports
use vision_calibration::planar_intrinsics::*;  // Planar workflow
use vision_calibration::core::*;               // Math types
use vision_calibration::linear::pnp;           // Linear solvers (per-module paths)
use vision_calibration::geometry::homography;  // Two-view solvers
use vision_calibration::mvg::rectification;    // N-view geometry
use vision_calibration::optim::RobustLoss;     // Optimization vocabulary (hand-picked)
```

### vision-calibration-py

Python bindings crate (`maturin`/PyO3) exposing high-level session workflows; published to PyPI.

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
Problem Builder  →  ProblemIR  →  OptimBackend::solve()
                  (generic)       (solver-specific)
```

This allows swapping the optimization backend without changing problem definitions.

### Step Functions

Calibration workflows are decomposed into discrete steps implemented as free functions. This allows:

- **Inspection** of intermediate state between steps
- **Resumption** from any point (via JSON checkpointing)
- **Customization** of per-step options
- **Composition** of steps from different problem types
