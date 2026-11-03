# vision-calibration-linear

Linear and closed-form initialization solvers for camera and rig calibration.

Most users want [`vision-calibration`](https://crates.io/crates/vision-calibration) instead, whose workflows run these solvers
as their initialization step; depend on this crate to call a solver directly.

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

## Usage

```rust
use nalgebra::{DMatrix, DVector};
use vision_calibration_linear::math::ridge_lstsq;

fn main() -> Result<(), vision_calibration_linear::Error> {
    // Fit y = 2x + 1 through exact samples with the normal equations.
    let a = DMatrix::from_fn(8, 2, |i, j| if j == 0 { i as f64 } else { 1.0 });
    let b = DVector::from_fn(8, |i, _| 2.0 * i as f64 + 1.0);
    let x = ridge_lstsq(&a, &b, 0.0)?;
    assert!((x[0] - 2.0).abs() < 1e-9 && (x[1] - 1.0).abs() < 1e-9);
    Ok(())
}
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
| `extrinsics` | Multi-camera rig initialization |
| `handeye` | Tsai-Lenz hand-eye |
| `laserline` | Laser plane estimation |

## See Also

- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): Math types, camera models, RANSAC engine
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): Non-linear refinement
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): High-level calibration pipelines
- [Book: Linear Initialization](https://vitalyvorobyev.github.io/calibration-rs/linear_overview.html), with a chapter per solver
