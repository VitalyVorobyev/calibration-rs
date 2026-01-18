# calib-optim

Non-linear least-squares optimization (bundle-adjustment style) for camera calibration.

This crate provides a **backend-agnostic optimization framework** for calibration problems. The core
design separates problem definition from solver implementation using an intermediate representation
(IR) that can be compiled to different backends.

## Features

- ✅ **Automatic differentiation support** (factors are `RealField`-generic)
- ✅ **Backend-agnostic IR** (`ir::ProblemIR`) with robust losses and manifolds
- ✅ **Levenberg–Marquardt backend** (tiny-solver) with sparse linear solvers
- ✅ **Built-in problems**
  - `planar_intrinsics`: pinhole intrinsics + Brown-Conrady5 + per-view poses
  - `rig_extrinsics`: multi-camera rig BA (supports missing observations)
  - `handeye`: multi-camera rig + robot hand-eye BA (EyeInHand / EyeToHand) with optional robot pose refinement
  - `linescan_bundle`: linescan bundle refinement
- ✅ **Structured parameter fixing** via `IntrinsicsFixMask` / `DistortionFixMask` / `CameraFixMask` (+ per-camera overrides)
- ✅ **Robust loss functions** (`None`, `Huber`, `Cauchy`, `Arctan`)
- ✅ **Manifold-aware optimization** for SE(3) parameters

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
- **`problems`** - High-level problem builders (planar intrinsics, rig extrinsics, hand-eye, linescan)

## Quick Start

### Planar Intrinsics Calibration

```rust
use calib_optim::planar_intrinsics::*;
use calib_core::{BrownConrady5, CorrespondenceView, DistortionFixMask, FxFyCxCySkew, IntrinsicsFixMask};
use calib_optim::ir::RobustLoss;
use calib_optim::BackendSolveOptions;

// 1. Prepare observations (world points + image detections)
let views: Vec<CorrespondenceView> = Vec::new(); // fill from a detector
let dataset = PlanarDataset::new(views)?;

// 2. Initialize with linear method (from calib-linear crate)
let init = PlanarIntrinsicsInit {
    intrinsics: FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 },
    distortion: BrownConrady5 { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 },
    poses: Vec::new(), // initial poses from homographies
};

// 3. Configure optimization
let opts = PlanarIntrinsicsSolveOptions {
    robust_loss: RobustLoss::Huber { scale: 2.0 },
    fix_intrinsics: IntrinsicsFixMask::default(),
    fix_distortion: DistortionFixMask::default(), // k3 fixed by default
    fix_poses: vec![0], // fix one pose for gauge freedom
    ..Default::default()
};

// 4. Run optimization
let result = optimize_planar_intrinsics(dataset, init, opts, BackendSolveOptions::default())?;

println!("Calibrated camera: {:?}", result.camera);
println!("Final cost: {}", result.final_cost);
```

### Other Built-In Problems

- Rig extrinsics (multi-camera BA): `calib_optim::problems::rig_extrinsics`
- Hand-eye (rig + robot BA): `calib_optim::handeye`
- Linescan bundle refinement: `calib_optim::problems::linescan_bundle`

For end-to-end examples, see the integration tests linked below.

## Parameter Fixing

Selectively fix parameters during optimization:

```rust
use calib_core::{DistortionFixMask, IntrinsicsFixMask};
use calib_optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;

let opts = PlanarIntrinsicsSolveOptions {
    // Fix intrinsics, optimize only distortion
    fix_intrinsics: IntrinsicsFixMask::all_fixed(),

    // Fix tangential distortion, keep k3 fixed (default)
    fix_distortion: DistortionFixMask {
        p1: true,
        p2: true,
        ..Default::default()
    },

    ..Default::default()
};
```

## Robust Loss Functions

Handle outliers with M-estimators:

```rust
use calib_optim::ir::RobustLoss;
use calib_optim::planar_intrinsics::PlanarIntrinsicsSolveOptions;

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

The crate includes integration tests using both synthetic and real data:

```bash
# Run all tests including real data integration
cargo test --package calib-optim

# Run specific integration test
cargo test --package calib-optim --test planar_intrinsics_real_data
```

Other useful tests:
- `cargo test --package calib-optim --test rig_extrinsics`
- `cargo test --package calib-optim --test handeye`
- `cargo test --package calib-optim --test linescan_bundle`

## Implementation Status

| Feature | Status |
|---------|--------|
| Backend-agnostic IR | ✅ Complete |
| tiny-solver backend (LM) | ✅ Complete |
| Planar intrinsics problem | ✅ Complete |
| Rig extrinsics problem | ✅ Complete |
| Hand-eye problem | ✅ Complete |
| Robot pose refinement (hand-eye) | ✅ Complete |
| Linescan bundle problem | ✅ Complete |
| Pinhole reprojection | ✅ Complete |
| Brown-Conrady distortion | ✅ Complete (k1, k2, k3, p1, p2) |
| Parameter fixing | ✅ Complete |
| Robust loss functions | ✅ Complete |
| Real data validation | ✅ Complete |
| Ceres backend | ❌ Not implemented |

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

## Examples

Integration tests with full examples:
- [`tests/planar_intrinsics_real_data.rs`](tests/planar_intrinsics_real_data.rs)
- [`tests/rig_extrinsics.rs`](tests/rig_extrinsics.rs)
- [`tests/handeye.rs`](tests/handeye.rs)
- [`tests/linescan_bundle.rs`](tests/linescan_bundle.rs)

## See Also

- [calib-core](../calib-core): Math types, camera models, RANSAC framework
- [calib-linear](../calib-linear): Closed-form initialization solvers
- [calib-pipeline](../calib-pipeline): High-level end-to-end calibration pipelines
- [Book: Non-linear Optimization](../../book/src/nonlinear.md)
