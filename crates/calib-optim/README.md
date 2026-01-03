# calib-optim

Non-linear optimization for camera calibration with automatic differentiation.

This crate provides a **backend-agnostic optimization framework** for camera calibration problems. The core design separates problem definition from solver implementation using an intermediate representation (IR) that can be compiled to different optimization backends.

## Features

- ✅ **Automatic differentiation** via generic `RealField` trait
- ✅ **Backend-agnostic IR** for solver portability
- ✅ **Brown-Conrady distortion** optimization (k1, k2, k3, p1, p2)
- ✅ **Flexible parameter fixing** for selective optimization
- ✅ **Robust loss functions** (Huber, Cauchy, Arctan) for outlier handling
- ✅ **Manifold-aware optimization** for SE(3)/SO(3) parameters

## Architecture

The optimization pipeline consists of three stages:

```
Problem Builder → ProblemIR → Backend.compile() → Backend.solve() → Domain Result
```

1. **Problem Definition** - Build a `ProblemIR` describing parameters, factors, and constraints
2. **Backend Compilation** - Translate IR into solver-specific problem (e.g., `TinySolverBackend`)
3. **Optimization** - Run solver and extract solution as domain types

### Key Components

- **`ir`** - Backend-agnostic intermediate representation
- **`params`** - Parameter block definitions (intrinsics, distortion, poses)
- **`factors`** - Residual functions with autodiff support
- **`backend`** - Solver implementations (currently tiny-solver with Levenberg-Marquardt)
- **`problems`** - High-level problem builders (planar intrinsics, etc.)

## Quick Start

### Planar Intrinsics Calibration

```rust
use calib_optim::problems::planar_intrinsics::*;
use calib_optim::params::intrinsics::Intrinsics4;
use calib_optim::params::distortion::BrownConrady5Params;
use calib_optim::ir::RobustLoss;
use calib_optim::BackendSolveOptions;

// 1. Prepare observations (world points + image detections)
let views = vec![/* PlanarViewObservations from corner detections */];
let dataset = PlanarDataset::new(views)?;

// 2. Initialize with linear method (from calib-linear crate)
let init = PlanarIntrinsicsInit {
    intrinsics: Intrinsics4 { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0 },
    distortion: BrownConrady5Params::zeros(),
    poses: vec![/* initial poses from homographies */],
};

// 3. Configure optimization
let opts = PlanarIntrinsicsSolveOptions {
    robust_loss: RobustLoss::Huber { scale: 2.0 },
    fix_k3: true,  // Fix k3 to prevent overfitting
    ..Default::default()
};

// 4. Run optimization
let result = optimize_planar_intrinsics(dataset, init, opts, BackendSolveOptions::default())?;

println!("Calibrated camera: {:?}", result.camera);
println!("Final cost: {}", result.final_cost);
```

## Parameter Fixing

Selectively fix parameters during optimization:

```rust
let opts = PlanarIntrinsicsSolveOptions {
    // Fix intrinsics, optimize only distortion
    fix_fx: true,
    fix_fy: true,

    // Fix tangential distortion
    fix_p1: true,
    fix_p2: true,

    // Fix k3 (recommended for most lenses)
    fix_k3: true,

    ..Default::default()
};
```

## Robust Loss Functions

Handle outliers with M-estimators:

```rust
use calib_optim::ir::RobustLoss;

// Huber loss: L2 near zero, L1 for outliers
let opts = PlanarIntrinsicsSolveOptions {
    robust_loss: RobustLoss::Huber { scale: 2.0 },
    ..Default::default()
};

// Cauchy loss: gradual outlier suppression
let opts = PlanarIntrinsicsSolveOptions {
    robust_loss: RobustLoss::Cauchy { scale: 2.0 },
    ..Default::default()
};
```

## Testing with Real Data

The crate includes integration tests using real stereo chessboard data:

```bash
# Run all tests including real data integration
cargo test --package calib-optim

# Run specific integration test
cargo test --package calib-optim --test planar_intrinsics_real_data
```

Test results with real data:
- **Reprojection error improvement**: 18-20% reduction
- **Final mean error**: < 0.25 pixels
- **Parameter fixing**: Verified working for all parameter subsets

## Implementation Status

| Feature | Status |
|---------|--------|
| Backend-agnostic IR | ✅ Complete |
| tiny-solver backend (LM) | ✅ Complete |
| Planar intrinsics problem | ✅ Complete |
| Pinhole reprojection | ✅ Complete |
| Brown-Conrady distortion | ✅ Complete (k1, k2, k3, p1, p2) |
| Parameter fixing | ✅ Complete |
| Robust loss functions | ✅ Complete |
| Real data validation | ✅ Complete |
| Bundle adjustment | ❌ Not implemented |
| Multi-camera rigs | ❌ Not implemented |

## Performance Tips

- **Always initialize with linear methods** (calib-linear crate) for faster convergence
- **Use Huber loss** with `scale ≈ 2.0` for real data with corner detection noise
- **Fix k3 by default** unless calibrating wide-angle lenses (prevents overfitting)
- **Diverse viewpoints** improve conditioning and reduce correlations between parameters

## Numerical Stability

The implementation uses several techniques for robustness:

- Safe division with epsilon thresholds in projection (prevents division by zero)
- Hartley normalization in linear initialization (via calib-linear)
- Manifold-aware parameter updates for rotations (proper SE(3)/SO(3) handling)
- Sparse linear solvers for large problems (efficient memory usage)

## Relationship to Other Crates

- **calib-core**: Math types, camera models, RANSAC framework
- **calib-linear**: Closed-form initialization solvers (Zhang, homography, PnP, etc.)
- **calib-pipeline**: High-level end-to-end calibration pipelines
- **calib-cli**: Command-line interface for batch processing

## Examples

See [`tests/planar_intrinsics_real_data.rs`](tests/planar_intrinsics_real_data.rs) for a complete example using real stereo chessboard data with:
- Linear initialization via Zhang's method
- Non-linear refinement with distortion
- Parameter fixing demonstrations
- Reprojection error validation

## License

MIT
