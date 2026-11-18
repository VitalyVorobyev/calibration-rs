# Introduction

**calibration-rs** is a Rust library for camera calibration — the process of estimating the internal parameters (intrinsics, lens distortion) and external parameters (pose, rig geometry, hand-eye transforms) of camera systems from observed correspondences between known 3D points and their 2D projections.

[API reference](https://vitalyvorobyev.github.io/calibration-rs/api/vision_calibration/index.html)

## Three Ways In

- **The desktop app** runs a calibration from a folder of images and shows the result, its residuals and its uncertainty, without code ([The Desktop App](app.md)).
- **Python**: `pip install --pre vision-calibration` (the release candidate), one call per workflow ([Getting Started](getting_started.md)).
- **Rust**: the `vision-calibration` crate, with the same workflows plus step-by-step sessions ([Getting Started](getting_started.md)).

Not sure which workflow fits your hardware? [Choose Your Workflow](choose_workflow.md) maps your setup to a workflow and a camera model and says what to capture.

## Who This Book Is For

Engineers and researchers in machine vision, robotics and 3D reconstruction who want to:

- calibrate single cameras, stereo rigs or multi-camera systems;
- perform hand-eye calibration for cameras mounted on robot arms;
- calibrate laser triangulation devices (camera + laser plane);
- understand the mathematics behind calibration algorithms.

Running a calibration needs no mathematical background. The theory chapters assume familiarity with linear algebra (SVD, eigendecomposition, least squares) and basic projective geometry (homogeneous coordinates, projection matrices), and introduce Lie group theory (SO(3), SE(3)) when needed.

## What calibration-rs Provides

The library covers the full calibration pipeline:

1. **Linear initialization** — closed-form solvers (Zhang's method, DLT, PnP, Tsai-Lenz hand-eye, epipolar geometry) that produce approximate parameter estimates
2. **Non-linear refinement** — Levenberg-Marquardt bundle adjustment that minimizes reprojection error to sub-pixel accuracy
3. **Session framework** — a high-level API with step functions, configuration, and JSON checkpointing
4. **Data handling and diagnostics** — target detectors with a detection cache, a dataset manifest, per-feature residuals and parameter uncertainty for every result

## Workspace Structure

calibration-rs is a layered Rust workspace: the `vision-calibration` facade re-exports the calibration workflows, the closed-form initialization and non-linear refinement beneath them, the dataset manifest and target detectors, and the multiple-view geometry tools, all built on one crate of primitives and camera models. The initialization and optimization crates are peers, neither depending on the other. [Architecture Overview](architecture.md) draws the crate graph and what each crate holds.

## Relation to OpenCV

Readers familiar with OpenCV's calibration module will find analogous functionality throughout:

| OpenCV | calibration-rs |
|--------|---------------|
| `cv::calibrateCamera` | Planar intrinsics pipeline (Zhang init + bundle adjustment) |
| `cv::solvePnP` | `pnp::p3p()`, `pnp::dlt()` |
| `cv::findHomography` | `dlt_homography()`, `estimate_homography()` |
| `cv::findFundamentalMat` | `fundamental_8point()` |
| `cv::findEssentialMat` | `essential_5point()` |
| `cv::stereoCalibrate` | Rig extrinsics pipeline |

calibration-rs differs from OpenCV in several ways: it is written in pure Rust, uses a composable camera model with generic type parameters, provides a backend-agnostic optimization IR, and offers a session framework with JSON checkpointing for production workflows.

## Book Organization

The book is the one user guide for the library. It is organized by what you want to do first and by the mathematics behind it last:

- **Choose Your Workflow**: from your hardware to a workflow, a camera model and a capture plan.
- **Getting Started**: install the library and run a first calibration in Rust and Python, then a stereo rig on real data.
- **Calibration Workflows**: one chapter per problem type (planar intrinsics, hand-eye, rigs, laser devices), plus choosing a distortion model, seeding the initialization with prior knowledge and reading per-feature residuals.
- **The Session Framework**: the `CalibrationSession` API behind every workflow.
- **Data and Detection**: calibration targets and their detectors, the dataset manifest that describes a folder of images, laser datasets and the generated manifest field reference.
- **Uncertainty**: the covariance every result carries, how to read it, and the projection uncertainty of a camera.
- **Multiple-View Geometry**: pose recovery, triangulation, bundle adjustment, rectification and dense stereo for calibrated cameras.
- **The Desktop App**: running calibrations and inspecting the results without writing code.
- **Theory**: the camera model, geometric primitives and robust estimation, the linear initialization algorithms and the non-linear optimization machinery, each with a formal problem statement and derivation.
- **Extending the Library**: a residual of your own solved with the library's, and a pipeline composed from its pieces.
- **Reference**: synthetic data generation, a type reference and further reading.
- **Contributing to the Library**: adding an optimization problem or a pipeline problem type, and the regression suite.

Each algorithm chapter in the theory parts includes a formal **problem statement**, **objective function**, **assumptions**, and a **full derivation** leading to the implementation.

The chapters show Rust unless a section says otherwise; [Getting Started](getting_started.md) puts Rust and Python side by side.
