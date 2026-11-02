# vision-calibration-linear

Linear and closed-form initialization solvers for camera and rig calibration.

This crate provides deterministic, minimal-dependency implementations of classic
computer vision algorithms. These are intended as **starting points** for
non-linear refinement in `vision-calibration-optim` or `vision-calibration-pipeline`.

## Algorithms

| Category | Algorithms |
|----------|------------|
| **Intrinsics** | Zhang's method, iterative with distortion, Scheimpflug-tilt initialization, Kannala-Brandt fisheye from bearings |
| **Pose (PnP)** | DLT, P3P, EPnP, DLT in RANSAC |
| **Planar pose** | Pose from homography + intrinsics, or from bearing vectors (any central camera) |
| **Rig** | Multi-camera extrinsics initialization |
| **Hand-eye** | Tsai-Lenz (AX=XB), eye-in-hand and eye-to-hand |
| **Laserline** | Multi-view laser plane fitting |

Two-view solvers (homography DLT, fundamental/essential matrices, camera matrix,
linear triangulation) live in
[`vision-geometry`](https://crates.io/crates/vision-geometry).

## Expected Accuracy

These solvers are **initialization-grade**. Regression tests validate:

| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| Zhang intrinsics | fx/fy ~5%, cx/cy ~8px | No distortion |
| Iterative intrinsics | fx/fy ~10-40% | With distortion |
| Fisheye intrinsics | fx/fy ~1%, cx/cy ~1px | Kannala-Brandt, beyond 90° |
| Planar pose | R <0.05 rad, t <5 units | Square = 30 units |

Thresholds are intentionally loose for stable initialization.

## Coordinate Conventions

- **Poses**: `T_C_W` (transform from world/board into camera frame)
- **PnP solvers**: Accept pixel coordinates + intrinsics (normalize internally)

## Usage

```rust,no_run
use vision_calibration_core::{FxFyCxCy, Pt2, Pt3, RansacOptions};
use vision_calibration_linear::pnp::PnpSolver;

# fn main() -> Result<(), vision_calibration_linear::Error> {
let world: Vec<Pt3> = vec![/* target points, target frame */];
let image: Vec<Pt2> = vec![/* matching pixel observations */];
let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0 };

// Direct DLT (all points); the pose is T_C_W
let pose = PnpSolver::dlt(&world, &image, &k)?;

// DLT inside RANSAC for outlier rejection
let (pose, inliers) = PnpSolver::dlt_ransac(&world, &image, &k, &RansacOptions::new(2.0))?; // threshold: pixels
# Ok(())
# }
```

## Modules

| Module | Description |
|--------|-------------|
| `scheimpflug_init` | Tilt-aware intrinsics initialization |
| `zhang_intrinsics` | Zhang's closed-form intrinsics |
| `equidistant_init` | Kannala-Brandt fisheye intrinsics from bearings |
| `iterative_intrinsics` | Iterative K + distortion estimation |
| `distortion_fit` | Distortion from homography residuals |
| `planar_pose` | Pose from homography + K, or from bearings |
| `pnp` | DLT, P3P, EPnP, RANSAC |
| `extrinsics` | Multi-camera rig initialization |
| `handeye` | Tsai-Lenz hand-eye |
| `laserline` | Laser plane estimation |

## See Also

- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): Math types, camera models, RANSAC engine
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): Non-linear refinement
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): High-level calibration pipelines
- [Book: Linear Initialization](https://vitalyvorobyev.github.io/calibration-rs/linear_overview.html), with a chapter per solver
