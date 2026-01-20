# calib-linear

Linear and closed-form initialization solvers for camera and rig calibration.

This crate provides deterministic, minimal-dependency implementations of classic
computer vision algorithms. These are intended as **starting points** for
non-linear refinement in `calib-optim` or `calib-pipeline`.

## Algorithms

| Category | Algorithms |
|----------|------------|
| **Homography** | Normalized DLT, RANSAC wrapper |
| **Intrinsics** | Zhang's method, iterative with distortion |
| **Pose (PnP)** | DLT, P3P, EPnP, RANSAC variants |
| **Epipolar** | 8-point, 7-point fundamental; 5-point essential |
| **Triangulation** | Linear DLT |
| **Rig** | Multi-camera extrinsics initialization |
| **Hand-eye** | Tsai-Lenz (AX=XB) |
| **Linescan** | Multi-view laser plane fitting |

## Expected Accuracy

These solvers are **initialization-grade**. Regression tests validate:

| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| Zhang intrinsics | fx/fy ~5%, cx/cy ~8px | No distortion |
| Iterative intrinsics | fx/fy ~10-40% | With distortion |
| Planar pose | R <0.05 rad, t <5 units | Square = 30 units |
| Fundamental (8-pt) | Scaled error <0.07 | |
| Essential (5-pt) | Residuals ~3x GT | |
| Triangulation | p90 error <2 units | Ground-truth poses |

Thresholds are intentionally loose for stable initialization.

## Coordinate Conventions

- **Poses**: `T_C_W` (transform from world/board into camera frame)
- **Fundamental matrix**: Accepts pixel coordinates
- **Essential matrix**: Expects normalized coordinates (after K^-1)
- **PnP solvers**: Accept pixel coordinates + intrinsics (normalize internally)

## Usage Examples

### Homography Estimation

```rust
use calib_linear::homography::dlt_homography;
use calib_core::Pt2;

let world = vec![Pt2::new(0.0, 0.0), Pt2::new(1.0, 0.0), Pt2::new(1.0, 1.0), Pt2::new(0.0, 1.0)];
let image = vec![Pt2::new(120.0, 200.0), Pt2::new(220.0, 198.0), Pt2::new(225.0, 300.0), Pt2::new(118.0, 302.0)];

let h = dlt_homography(&world, &image)?;
println!("H = {h}");
# Ok::<(), anyhow::Error>(())
```

### PnP (Perspective-n-Point)

```rust
use calib_linear::pnp::{pnp_dlt, p3p_kneip, pnp_ransac};
use calib_core::{Pt2, Pt3, Mat3};

let points_3d = vec![Pt3::new(0.0, 0.0, 0.0), /* ... */];
let points_2d = vec![Pt2::new(320.0, 240.0), /* ... */];
let k = Mat3::identity(); // Intrinsics matrix

// Direct DLT (all points)
let pose = pnp_dlt(&points_3d, &points_2d, &k)?;

// RANSAC for outlier rejection
let (pose, inliers) = pnp_ransac(&points_3d, &points_2d, &k, ransac_opts)?;
# Ok::<(), anyhow::Error>(())
```

### Iterative Intrinsics (with Distortion)

```rust
use calib_linear::iterative_intrinsics::{IterativeIntrinsicsSolver, IterativeCalibView, IterativeIntrinsicsOptions};

let views: Vec<IterativeCalibView> = /* prepare views */;
let opts = IterativeIntrinsicsOptions::default();
let result = IterativeIntrinsicsSolver::estimate(&views, opts)?;

println!("K: fx={}, fy={}", result.intrinsics.fx, result.intrinsics.fy);
println!("Distortion: k1={}, k2={}", result.distortion.k1, result.distortion.k2);
# Ok::<(), anyhow::Error>(())
```

### Hand-Eye Calibration

```rust
use calib_linear::handeye::estimate_handeye_dlt;
use calib_core::Iso3;

let robot_poses: Vec<Iso3> = /* from robot controller */;
let camera_poses: Vec<Iso3> = /* from calibration */;
let distance_weight = 1.0;

let handeye = estimate_handeye_dlt(&robot_poses, &camera_poses, distance_weight)?;
# Ok::<(), anyhow::Error>(())
```

### Linescan Plane Fitting

```rust
use calib_linear::linescan::{LinescanPlaneSolver, LinescanView};

let views: Vec<LinescanView> = /* views with laser pixels */;
let camera = /* calibrated camera */;

// Multi-view fitting breaks single-view collinearity
let estimate = LinescanPlaneSolver::from_views(&views, &camera)?;
println!("Plane normal: {:?}", estimate.normal);
# Ok::<(), anyhow::Error>(())
```

## Modules

| Module | Description |
|--------|-------------|
| `homography` | DLT homography + RANSAC |
| `zhang_intrinsics` | Zhang's closed-form intrinsics |
| `iterative_intrinsics` | Iterative K + distortion estimation |
| `distortion_fit` | Distortion from homography residuals |
| `planar_pose` | Pose from homography + K |
| `pnp` | DLT, P3P, EPnP, RANSAC |
| `epipolar` | Fundamental/essential matrices |
| `triangulation` | Linear triangulation |
| `extrinsics` | Multi-camera rig initialization |
| `handeye` | Tsai-Lenz hand-eye |
| `linescan` | Laser plane estimation |

## See Also

- [calib-core](../calib-core): Math types, camera models, RANSAC engine
- [calib-optim](../calib-optim): Non-linear refinement
- [calib-pipeline](../calib-pipeline): High-level calibration pipelines
- [Book: Linear Calibration](../../book/src/linear.md)
