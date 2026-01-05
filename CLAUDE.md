# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Building and Testing
```bash
# Build entire workspace
cargo build --workspace

# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test --package calib-optim
cargo test --package calib-linear

# Run a single test
cargo test --package calib-optim --lib synthetic_planar_with_distortion_converges

# Check compilation without building
cargo check --workspace
```

### Formatting and Linting
```bash
# Format code
cargo fmt

# Run clippy linter
cargo clippy --workspace
```

### Documentation
```bash
# Build documentation
cargo doc --workspace --no-deps

# Open docs in browser
cargo doc --workspace --no-deps --open
```

### CLI Usage
```bash
# Run CLI
cargo run -p calib-cli -- --help

# Example calibration run
cargo run -p calib-cli -- --input views.json --config config.json > report.json
```

## Architecture Overview

### Workspace Structure
This is a **6-crate Rust workspace** (~7,200 lines of code) for camera calibration with a clean layered architecture:

```
calib-cli (CLI wrapper)
    ↓
calib (facade) → calib-pipeline (high-level pipelines)
                      ↓
                 calib-optim (non-linear refinement) + calib-linear (initialization)
                      ↓
                 calib-core (primitives, models, RANSAC)
```

**Key dependency rule**: Linear and non-linear layers are peers; they both depend on calib-core but not each other.

### Crate Responsibilities

- **calib-core**: Math types (nalgebra-based), composable camera models (projection → distortion → sensor → intrinsics), generic RANSAC engine, shared test utilities
- **calib-linear**: Closed-form initialization solvers (Zhang, homography, PnP, epipolar, hand-eye, **iterative intrinsics + distortion**) for bootstrapping optimization
- **calib-optim**: Non-linear least-squares with backend-agnostic IR, autodiff-compatible factors, pluggable solvers (currently tiny-solver)
- **calib-pipeline**: Ready-to-use end-to-end calibration workflows with JSON I/O
- **calib-cli**: Command-line interface for batch processing
- **calib**: Convenience re-export facade

### Camera Model Composition

The camera model in `calib-core` is a composable pipeline:

```
pixel = K ∘ sensor ∘ distortion ∘ projection(dir)
```

- `projection`: Camera-frame direction → normalized coordinates (e.g., Pinhole)
- `distortion`: Warp normalized coords (Brown-Conrady radial k1, k2, k3 + tangential p1, p2)
- `sensor`: Apply homography (Identity or Scheimpflug tilt)
- `K`: Intrinsics matrix mapping to pixels (fx, fy, cx, cy, skew)

Each stage is a separate type that can be mixed and matched via generics.

### calib-optim Backend-Agnostic IR Architecture

**Core design**: Optimization problems are defined in a solver-independent intermediate representation (IR), then compiled to specific backends.

**Key types** ([ir/types.rs](crates/calib-optim/src/ir/types.rs)):
- `ProblemIR`: Complete problem definition with parameters and residuals
- `ParamBlock`: Variable definition (name, dimension, manifold, fixed mask, bounds)
- `ResidualBlock`: Factor connecting parameters (references ParamIds)
- `FactorKind`: Enum of supported factor types (e.g., `ReprojPointPinhole4Dist5`)
- `ManifoldKind`: Parameter geometry (Euclidean, SE3, SO3, S2)

**Factor system** ([factors/](crates/calib-optim/src/factors/)):
- Generic residual functions parameterized over `RealField` trait for autodiff compatibility
- Example: `reproj_residual_pinhole4_dist5_se3_generic<T: RealField>()` works with both f64 and dual numbers
- Backend adapters wrap these generic functions in solver-specific factor structs

**Backend pattern** ([backend/](crates/calib-optim/src/backend/)):
1. Problem builder creates `ProblemIR` + initial values map
2. Backend's `compile()` method translates IR → solver-specific problem
3. Backend's `solve()` runs optimization and returns `BackendSolution`
4. Problem-specific code extracts solution into domain types (e.g., `Camera`)

**Currently implemented**:
- **Planar intrinsics problem** ([problems/planar_intrinsics.rs](crates/calib-optim/src/problems/planar_intrinsics.rs)): Zhang-style calibration optimizing intrinsics (fx, fy, cx, cy) + distortion (k1, k2, k3, p1, p2) + poses
- **tiny-solver backend** ([backend/tiny_solver_backend.rs](crates/calib-optim/src/backend/tiny_solver_backend.rs)): Levenberg-Marquardt with sparse linear solvers

### Coordinate Conventions

- **Poses**: `T_C_W` (transform from world/board to camera frame)
- **Fundamental matrix**: Accepts pixel coordinates
- **Essential matrix**: Expects normalized coordinates (after K^-1)
- **PnP solvers**: Accept pixel coordinates + intrinsics, normalize internally
- **SE3 storage**: `[qx, qy, qz, qw, tx, ty, tz]` (quaternion + translation)

### Adding New Optimization Problems

To add a new calibration problem to calib-optim:

1. **Define parameter blocks** in `params/` (e.g., intrinsics, poses, distortion)
2. **Implement generic residual function** in `factors/` using `RealField` trait
3. **Add FactorKind variant** in `ir/types.rs` with validation logic
4. **Create problem builder** in `problems/` that constructs ProblemIR
5. **Integrate with backend** by adding factor compilation in `backend/tiny_solver_backend.rs`
6. **Write tests** with synthetic ground truth data

Example workflow is demonstrated in the planar intrinsics implementation.

### Numerical Considerations

- **Hartley normalization**: Used throughout calib-linear for numerical stability
- **Robust loss functions**: Huber, Cauchy, Arctan available for handling outliers
- **Parameter masking**: Selective fixing of optimization variables (e.g., fix k3 to prevent overfitting)
- **Manifold constraints**: SE3/SO3 parameters use proper Lie group updates via tiny-solver

### Testing Philosophy

- **Synthetic ground truth tests**: Generate data from known parameters, verify convergence
- **Real data regression tests**: calib-linear uses stereo chessboard dataset for coarse validation
- **Loose tolerances for initialization**: Linear solvers aim for ~5% accuracy (sufficient for non-linear refinement)
- **Tight tolerances for optimization**: Non-linear tests verify convergence to <1% error

### Important Implementation Details

**Autodiff compatibility**: When writing residual functions:
- Use `.clone()` liberally (dual numbers are cheap to clone)
- Avoid in-place operations
- Use `T::from_f64().unwrap()` to convert constants
- Generic function signature: `fn residual<T: RealField>(...) -> SVector<T, N>`

**Fixed parameters**: The `FixedMask` type supports:
- Per-index fixing for Euclidean parameters
- All-or-nothing fixing for manifolds (partial fixing not supported by tiny-solver)

**Distortion optimization**: k3 is fixed by default (`fix_k3: true`) because it often causes overfitting with typical calibration data. Only optimize k3 for wide-angle lenses or with high-quality data.

## Iterative Intrinsics Estimation (NEW)

calib-linear now supports **iterative refinement** for jointly estimating camera intrinsics and Brown-Conrady distortion **without requiring ground truth distortion**. This enables realistic calibration workflows.

### Problem

The classic Zhang method assumes distortion-free inputs. When distortion is present, directly applying Zhang to distorted pixels produces **biased intrinsics estimates**.

### Solution

An alternating optimization scheme:
1. **Initial estimate**: Compute K from distorted pixels (ignoring distortion)
2. **Distortion estimation**: Estimate distortion from homography residuals using K
3. **Pixel undistortion**: Apply estimated distortion to correct observations
4. **Intrinsics refinement**: Re-estimate K from undistorted pixels
5. **Iterate**: Repeat steps 2-4 (typically 1-2 iterations sufficient)

### API Usage

```rust
use calib_linear::iterative_intrinsics::{
    IterativeCalibView, IterativeIntrinsicsOptions, IterativeIntrinsicsSolver,
};
use calib_linear::DistortionFitOptions;

// Prepare views from raw corner detections
let views: Vec<IterativeCalibView> = /* load from calibration data */;

// Configure options
let opts = IterativeIntrinsicsOptions {
    iterations: 2,  // 1-3 typically sufficient
    distortion_opts: DistortionFitOptions {
        fix_k3: true,          // Conservative: estimate only k1, k2
        fix_tangential: false, // Estimate p1, p2
        iters: 8,
    },
};

// Run iterative estimation
let result = IterativeIntrinsicsSolver::estimate(&views, opts)?;

// Access results
println!("K: fx={}, fy={}, cx={}, cy={}",
         result.intrinsics.fx, result.intrinsics.fy,
         result.intrinsics.cx, result.intrinsics.cy);
println!("Distortion: k1={}, k2={}, p1={}, p2={}",
         result.distortion.k1, result.distortion.k2,
         result.distortion.p1, result.distortion.p2);

// Use for non-linear refinement initialization
```

### Modules

- **`distortion_fit`**: Closed-form distortion estimation from homography residuals (linear least-squares on pixel residuals)
- **`iterative_intrinsics`**: Alternating K and distortion refinement loop

### Typical Workflow

```
Raw corner detections
    ↓
IterativeIntrinsicsSolver (calib-linear)
    ↓
Initial K + distortion estimates (10-40% accuracy)
    ↓
optimize_planar_intrinsics (calib-optim)
    ↓
Final calibrated camera (<1% accuracy, <1px reprojection error)
```

### Accuracy Expectations

- **After iterative linear init**: 10-40% error on intrinsics (sufficient for initialization)
- **After non-linear refinement**: <2% error on intrinsics, <1px mean reprojection error

### When to Use

- ✅ You have multiple views of a planar calibration pattern
- ✅ You don't have ground truth distortion parameters
- ✅ You need both K and distortion for initialization
- ❌ For distortion-free cameras, use Zhang directly

## Linescan Calibration with Laser Plane

calib-optim supports **linescan sensor calibration** ([problems/linescan_bundle.rs](crates/calib-optim/src/problems/linescan_bundle.rs)) that jointly optimizes camera intrinsics, distortion, poses, and laser plane parameters using both calibration pattern observations and laser line observations.

### Laser Residual Types

Two approaches are available for laser plane calibration, selectable via `LaserResidualType`:

**1. Point-to-Plane Distance (PointToPlane)**
- Undistorts pixel → back-projects to 3D ray → intersects with target plane
- Computes 3D point in camera frame
- Measures signed distance from point to laser plane
- Residual: 1D distance (in meters)

**2. Line-Distance in Normalized Plane (LineDistNormalized)** *(default)*
- Computes 3D intersection line of laser plane and target plane in camera frame
- Projects line onto z=1 normalized camera plane
- Undistorts laser pixels to normalized coordinates (done once per pixel)
- Measures perpendicular distance from pixel to projected line (2D geometry)
- Scales by `sqrt(fx*fy)` for pixel-comparable residual
- Residual: 1D distance (in pixels)

### Choosing the Approach

- **`LineDistNormalized` (default)**: Recommended for most use cases
  - Faster: undistortion done once, no ray-plane intersection per pixel
  - Residuals in pixels (directly comparable to reprojection error)
  - Simpler 2D geometry in normalized plane
  - Uniform handling of all laser pixels in one coordinate system

- **`PointToPlane`**: Alternative approach
  - May be more robust when line projection is degenerate
  - Residuals in metric units (meters)

Both approaches yield similar accuracy in practice. Benchmark tests show convergence to <6% intrinsics error and <5° plane normal error.

### API Usage

```rust
use calib_optim::problems::linescan_bundle::*;
use calib_optim::backend::BackendSolveOptions;

// Prepare dataset with calibration points and laser line observations
let views: Vec<LinescanViewObservations> = /* ... */;
let dataset = LinescanDataset::new_single_plane(views)?;

// Initial estimates
let initial = LinescanInit::new(intrinsics, distortion, poses, planes)?;

// Configure options
let opts = LinescanSolveOptions {
    fix_k3: true,
    fix_poses: vec![0],  // Fix first pose for gauge freedom
    laser_residual_type: LaserResidualType::LineDistNormalized,  // Default
    ..Default::default()
};

let backend_opts = BackendSolveOptions {
    max_iters: 50,
    verbosity: 1,
    ..Default::default()
};

// Run optimization
let result = optimize_linescan(&dataset, &initial, &opts, &backend_opts)?;

println!("Laser plane: normal={:?}, distance={}",
         result.planes[0].normal, result.planes[0].distance);
```

### Implementation Details

**Factors** ([factors/linescan.rs](crates/calib-optim/src/factors/linescan.rs)):
- `laser_plane_pixel_residual_generic()`: Point-to-plane distance
- `laser_line_dist_normalized_generic()`: Line-distance in normalized plane

**IR types** ([ir/types.rs](crates/calib-optim/src/ir/types.rs)):
- `FactorKind::LaserPlanePixel`: Original point-to-plane factor
- `FactorKind::LaserLineDist2D`: New line-distance factor

Both factors:
- Take 4 parameter blocks: [intrinsics, distortion, pose, plane]
- Return 1D residuals
- Support autodiff for analytical Jacobians

### Testing

Integration tests ([tests/linescan_bundle.rs](crates/calib-optim/tests/linescan_bundle.rs)) verify:
- Convergence with synthetic ground truth data
- Comparison of both residual types
- Similar accuracy: ~4-5% intrinsics error, ~1-2° plane normal error

## Project Status

**Current state**: Early development, APIs may change
- calib-linear: ✅ Feature-complete and tested
- calib-optim: ✅ Planar intrinsics with distortion working (just implemented)
- calib-pipeline: 🟡 Basic planar intrinsics pipeline functional
- Multi-camera, bundle adjustment: ❌ Not yet implemented

**Recent work**:
- ✅ Implemented iterative linear intrinsics + distortion estimation in calib-linear
- ✅ Added closed-form distortion fitting from homography residuals
- ✅ Created shared test utilities in calib-core
- ✅ Added realistic calibration tests (no ground truth distortion required)
- All 43+ workspace tests passing
