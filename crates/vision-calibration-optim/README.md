# vision-calibration-optim

Non-linear least-squares optimization (bundle-adjustment style) for camera calibration.

This crate provides a **backend-agnostic optimization framework** for calibration problems. The core
design separates problem definition from solver implementation using an intermediate representation
(IR) that can be compiled to different backends.

## Features

- **Automatic differentiation**: factors are generic over `RealField`
- **Backend-agnostic IR** (`ir::ProblemIR`) with robust losses and manifolds
- **One Levenberg–Marquardt loop, two backends**: tiny-solver (default) or factrs
  linearize the problem, selected by `BackendSolveOptions::backend`; sparse
  Cholesky or QR linear solvers
- **Built-in problems** (`problems::*`):
  - `planar_intrinsics`, `scheimpflug_intrinsics`: single-camera intrinsics + per-view poses
  - `rig_extrinsics`, `rig_extrinsics_scheimpflug`: multi-camera rig bundle adjustment (supports missing observations)
  - `handeye`, `handeye_scheimpflug`: rig + robot hand-eye (eye-in-hand / eye-to-hand) with optional robot pose refinement
  - `laserline_bundle`, `laserline_rig_bundle`: laser plane refinement (single camera / rig)
  - `rig_handeye_laserline_bundle`: joint rig + hand-eye + laser plane
- **Structured parameter fixing** via `IntrinsicsFixMask` / `DistortionFixMask` / `CameraFixMask` (+ per-camera overrides)
- **Robust loss functions** (`None`, `Huber`, `Cauchy`, `Arctan`)
- **Manifold-aware optimization** for SE(3) parameters

## Architecture

The optimization pipeline consists of three stages:

```text
Problem Builder → ProblemIR → Backend.compile() → Backend.solve() → Domain Result
```

1. **Problem Definition** - Build a `ProblemIR` describing parameters, factors, and constraints
2. **Backend Compilation** - Translate the IR into the selected backend's problem (`SolverBackend`: tiny-solver or factrs)
3. **Optimization** - Run the shared Levenberg–Marquardt loop and extract the solution as domain types

### Key Components

- **`ir`** - Backend-agnostic intermediate representation
- **`params`** - Parameter block definitions (intrinsics, distortion, poses)
- **`factors`** - Residual functions with autodiff support
- **`backend`** - The Levenberg–Marquardt loop and its two linearization engines (tiny-solver, factrs)
- **`problems`** - High-level problem builders (intrinsics, rig extrinsics, hand-eye, laserline)

## Quick Start

### Planar Intrinsics Calibration

```rust,no_run
use vision_calibration_core::{
    BrownConrady5, CorrespondenceView, DistortionFixMask, FxFyCxCySkew, Iso3, PlanarDataset, Pt2,
    Pt3, View,
};
use vision_calibration_optim::{
    optimize_planar_intrinsics, BackendSolveOptions, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RobustLoss,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// 1. Prepare observations (target points + image detections); fill from a detector.
let view = View::without_meta(CorrespondenceView::new(
    vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(1.0, 0.0, 0.0), Pt3::new(1.0, 1.0, 0.0), Pt3::new(0.0, 1.0, 0.0)],
    vec![Pt2::new(100.0, 100.0), Pt2::new(200.0, 100.0), Pt2::new(200.0, 200.0), Pt2::new(100.0, 200.0)],
)?);
let dataset = PlanarDataset::new(vec![view])?;

// 2. Initial parameters (from vision-calibration-linear or a prior calibration).
let init = PlanarIntrinsicsParams::new_from_components(
    FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 },
    BrownConrady5 { k1: 0.0, k2: 0.0, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 },
    vec![Iso3::identity()], // one pose per view
)?;

// 3. Configure the solve.
let opts = PlanarIntrinsicsSolveOptions {
    robust_loss: RobustLoss::Huber { scale: 2.0 },
    fix_distortion: DistortionFixMask { k3: true, ..Default::default() },
    ..Default::default()
};

// 4. Optimize.
let result = optimize_planar_intrinsics(&dataset, &init, opts, BackendSolveOptions::default())?;
println!("Calibrated camera: {:?}", result.params.camera);
# Ok(())
# }
```

### Other Built-In Problems

Each lives in `vision_calibration_optim::problems::<name>` (see the feature list
above); the integration tests listed under [Examples](#examples) show end-to-end use.

## Parameter Fixing

Selectively fix parameters during optimization:

```rust
use vision_calibration_core::{DistortionFixMask, IntrinsicsFixMask};
use vision_calibration_optim::PlanarIntrinsicsSolveOptions;

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
use vision_calibration_optim::RobustLoss;
use vision_calibration_optim::PlanarIntrinsicsSolveOptions;

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

## Performance Tips

- **Always initialize with linear methods** (vision-calibration-linear crate) for faster convergence
- **Use Huber loss** with `scale ≈ 2.0` for real data with corner detection noise
- **Fix k3 by default** unless calibrating wide-angle lenses (prevents overfitting)
- **Diverse viewpoints** improve conditioning and reduce correlations between parameters

## Numerical Stability

The implementation uses several techniques for robustness:

- Safe division with epsilon thresholds in projection (prevents division by zero)
- Hartley normalization in linear initialization (via vision-calibration-linear)
- Manifold-aware parameter updates for rotations (proper SE(3)/SO(3) handling)
- Sparse linear solvers for large problems (efficient memory usage)

## Examples

Integration tests with full end-to-end use (run with `cargo test -p vision-calibration-optim --test <name>`):
- [`tests/planar_intrinsics.rs`](tests/planar_intrinsics.rs), [`tests/planar_intrinsics_real_data.rs`](tests/planar_intrinsics_real_data.rs)
- [`tests/rig_extrinsics.rs`](tests/rig_extrinsics.rs)
- [`tests/handeye.rs`](tests/handeye.rs)
- [`tests/laserline_bundle.rs`](tests/laserline_bundle.rs), [`tests/rig_laserline.rs`](tests/rig_laserline.rs)

## See Also

- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): Math types, camera models, RANSAC framework
- [vision-calibration-linear](https://crates.io/crates/vision-calibration-linear): Closed-form initialization solvers
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): High-level end-to-end calibration pipelines
- [Book: Non-linear Optimization](https://vitalyvorobyev.github.io/calibration-rs/nlls_overview.html)
