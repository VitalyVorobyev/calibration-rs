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
- **PnP solvers**: Accept pixel coordinates + intrinsics (normalize internally)

## Usage

```rust
use vision_calibration_core::{FxFyCxCy, Iso3, Pt2, Pt3, RansacOptions};
use vision_calibration_linear::pnp::PnpSolver;

fn main() -> Result<(), vision_calibration_linear::Error> {
    let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0 };
    // Twelve target points (target frame) and a camera 0.6 m away.
    let world: Vec<Pt3> = (0..12)
        .map(|i| Pt3::new(0.1 * f64::from(i % 4), 0.1 * f64::from(i / 4), 0.05 * f64::from(i * 7 % 5)))
        .collect();
    let truth = Iso3::translation(-0.15, -0.1, 0.6); // T_C_W
    let image: Vec<Pt2> = world
        .iter()
        .map(|p| {
            let c = truth * p;
            Pt2::new(k.fx * c.x / c.z + k.cx, k.fy * c.y / c.z + k.cy)
        })
        .collect();

    // Direct DLT (all points); the pose is T_C_W
    let pose = PnpSolver::dlt(&world, &image, &k)?;
    assert!((pose.translation.vector - truth.translation.vector).norm() < 1e-6);

    // DLT inside RANSAC for outlier rejection (threshold in pixels)
    let (_pose, inliers) = PnpSolver::dlt_ransac(&world, &image, &k, &RansacOptions::new(2.0))?;
    assert_eq!(inliers.len(), world.len());
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
| `pnp` | DLT, P3P, EPnP, RANSAC |
| `extrinsics` | Multi-camera rig initialization |
| `handeye` | Tsai-Lenz hand-eye |
| `laserline` | Laser plane estimation |

## See Also

- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): Math types, camera models, RANSAC engine
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): Non-linear refinement
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): High-level calibration pipelines
- [Book: Linear Initialization](https://vitalyvorobyev.github.io/calibration-rs/linear_overview.html), with a chapter per solver
