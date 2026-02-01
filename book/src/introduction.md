# Introduction

**calibration-rs** is a Rust library for camera calibration — the process of estimating the internal parameters (intrinsics, lens distortion) and external parameters (pose, rig geometry, hand-eye transforms) of camera systems from observed correspondences between known 3D points and their 2D projections.

[API reference](https://vitalyvorobyev.github.io/calibration-rs/api)

## Who This Book Is For

This book targets engineers and researchers working in machine vision, robotics, and 3D reconstruction who want to:

- Calibrate single cameras, stereo rigs, or multi-camera systems
- Perform hand-eye calibration for cameras mounted on robot arms
- Calibrate laser triangulation devices (camera + laser plane)
- Understand the mathematical foundations behind calibration algorithms

We assume familiarity with linear algebra (SVD, eigendecomposition, least squares) and basic projective geometry (homogeneous coordinates, projection matrices). Lie group theory (SO(3), SE(3)) is introduced when needed.

## What calibration-rs Provides

The library covers the full calibration pipeline:

1. **Linear initialization** — closed-form solvers (Zhang's method, DLT, PnP, Tsai-Lenz hand-eye, epipolar geometry) that produce approximate parameter estimates
2. **Non-linear refinement** — Levenberg-Marquardt bundle adjustment that minimizes reprojection error to sub-pixel accuracy
3. **Session framework** — a high-level API with step functions, configuration, and JSON checkpointing

## Workspace Structure

calibration-rs is organized as a 5-crate Rust workspace with a layered architecture:

```
vision-calibration (facade)
    → vision-calibration-pipeline (sessions, workflows)
        → vision-calibration-optim (non-linear refinement)
        → vision-calibration-linear (initialization)
            → vision-calibration-core (primitives, camera models, RANSAC)
```

The **key dependency rule**: the linear and optimization crates are peers — they both depend on `vision-calibration-core` but not on each other. This separation keeps initialization algorithms independent of the optimization backend.

## Relation to OpenCV

Readers familiar with OpenCV's calibration module will find analogous functionality throughout:

| OpenCV | calibration-rs |
|--------|---------------|
| `cv::calibrateCamera` | Planar intrinsics pipeline (Zhang init + bundle adjustment) |
| `cv::solvePnP` | `PnpSolver::p3p()`, `PnpSolver::dlt()` |
| `cv::findHomography` | `HomographySolver::dlt()`, `dlt_homography_ransac()` |
| `cv::findFundamentalMat` | `EpipolarSolver::fundamental_8point()` |
| `cv::findEssentialMat` | `EpipolarSolver::essential_5point()` |
| `cv::stereoCalibrate` | Rig extrinsics pipeline |

calibration-rs differs from OpenCV in several ways: it is written in pure Rust, uses a composable camera model with generic type parameters, provides a backend-agnostic optimization IR, and offers a session framework with JSON checkpointing for production workflows.

## Book Organization

The book is structured in seven parts:

- **Part I: Camera Model** — the composable projection pipeline (pinhole, distortion, sensor tilt, intrinsics)
- **Part II: Geometric Primitives** — rigid transforms, RANSAC, robust loss functions
- **Part III: Linear Initialization** — all closed-form solvers with full mathematical derivations
- **Part IV: Non-Linear Optimization** — Levenberg-Marquardt, manifold constraints, autodiff, the IR architecture
- **Part V: Calibration Workflows** — end-to-end pipelines for each problem type, with both synthetic and real data examples
- **Part VI: Session Framework** — the high-level `CalibrationSession` API
- **Part VII: Extending the Library** — adding new problems, backends, and pipeline types

Each algorithm chapter includes a formal **problem statement**, **objective function**, **assumptions**, and a **full derivation** leading to the implementation.
