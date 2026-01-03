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

- **calib-core**: Math types (nalgebra-based), composable camera models (projection → distortion → sensor → intrinsics), generic RANSAC engine
- **calib-linear**: Closed-form initialization solvers (Zhang, homography, PnP, epipolar, hand-eye) for bootstrapping optimization
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

## Project Status

**Current state**: Early development, APIs may change
- calib-linear: ✅ Feature-complete and tested
- calib-optim: ✅ Planar intrinsics with distortion working (just implemented)
- calib-pipeline: 🟡 Basic planar intrinsics pipeline functional
- Multi-camera, bundle adjustment: ❌ Not yet implemented

**Recent work**: Just completed implementation of Brown-Conrady distortion optimization in calib-optim (k1, k2, k3, p1, p2 coefficients). All 38 workspace tests passing.
