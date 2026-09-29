# vision-calibration-linear

Linear and closed-form initialization solvers for camera and rig calibration.

This crate provides deterministic, minimal-dependency implementations of classic
computer vision algorithms. These are intended as **starting points** for
non-linear refinement in `vision-calibration-optim` or `vision-calibration-pipeline`.

## Algorithms

| Category | Algorithms |
|----------|------------|
| **Intrinsics** | Zhang's method, iterative with distortion, Scheimpflug-tilt initialization |
| **Pose (PnP)** | DLT, P3P, EPnP, DLT in RANSAC |
| **Planar pose** | Pose from homography + intrinsics |
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
| Planar pose | R <0.05 rad, t <5 units | Square = 30 units |

Thresholds are intentionally loose for stable initialization.

## Coordinate Conventions

- **Poses**: `T_C_W` (transform from world/board into camera frame)
- **PnP solvers**: Accept pixel coordinates + intrinsics (normalize internally)

## Usage Examples

### PnP (Perspective-n-Point)

```rust,no_run
use vision_calibration_core::{FxFyCxCySkew, Pt2, Pt3, RansacOptions};
use vision_calibration_linear::pnp::PnpSolver;

# fn main() -> Result<(), vision_calibration_linear::Error> {
let world: Vec<Pt3> = vec![/* target points, target frame */];
let image: Vec<Pt2> = vec![/* matching pixel observations */];
let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };

// Direct DLT (all points); the pose is T_C_W
let pose = PnpSolver::dlt(&world, &image, &k)?;

// DLT inside RANSAC for outlier rejection
let (pose, inliers) = PnpSolver::dlt_ransac(&world, &image, &k, &RansacOptions::default())?;
# Ok(())
# }
```

### Iterative Intrinsics (with Distortion)

```rust,no_run
use vision_calibration_core::PlanarDataset;
use vision_calibration_linear::prelude::*;

# fn main() -> Result<(), vision_calibration_linear::Error> {
# let dataset: PlanarDataset = unimplemented!();
let camera = estimate_intrinsics_iterative(&dataset, IterativeIntrinsicsOptions::default())?;
println!("intrinsics: {:?}", camera.k);
# Ok(())
# }
```

### Hand-Eye Calibration

```rust,no_run
use vision_calibration_core::Iso3;
use vision_calibration_linear::handeye::{estimate_gripper_se3_target_dlt, estimate_handeye_dlt};

# fn main() -> Result<(), vision_calibration_linear::Error> {
let base_se3_gripper: Vec<Iso3> = vec![/* T_B_G from the robot controller */];
let min_angle_deg = 5.0;

// Eye-in-hand: gripper -> camera from target -> camera poses.
let target_se3_camera: Vec<Iso3> = vec![/* inverted PnP poses */];
let gripper_se3_camera =
    estimate_handeye_dlt(&base_se3_gripper, &target_se3_camera, min_angle_deg)?;

// Eye-to-hand: gripper -> target from camera -> target poses.
let camera_se3_target: Vec<Iso3> = vec![/* PnP / planar poses */];
let gripper_se3_target =
    estimate_gripper_se3_target_dlt(&base_se3_gripper, &camera_se3_target, min_angle_deg)?;
# Ok(())
# }
```

### Laserline Plane Fitting

```rust,no_run
use vision_calibration_core::{BrownConrady5, Camera, FxFyCxCySkew, IdentitySensor, Pinhole};
use vision_calibration_linear::laserline::{LaserlinePlaneSolver, LaserlineView};

# fn main() -> Result<(), vision_calibration_linear::Error> {
let views: Vec<LaserlineView> = vec![/* views with laser pixels */];
# let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };
# let camera = Camera::new(Pinhole, BrownConrady5::default(), IdentitySensor, k);

// Multiple views at different poses break single-view collinearity.
let estimate = LaserlinePlaneSolver::from_views(&views, &camera)?;
println!("plane normal: {:?}", estimate.normal);
# Ok(())
# }
```

## Modules

| Module | Description |
|--------|-------------|
| `scheimpflug_init` | Tilt-aware intrinsics initialization |
| `zhang_intrinsics` | Zhang's closed-form intrinsics |
| `iterative_intrinsics` | Iterative K + distortion estimation |
| `distortion_fit` | Distortion from homography residuals |
| `planar_pose` | Pose from homography + K |
| `pnp` | DLT, P3P, EPnP, RANSAC |
| `extrinsics` | Multi-camera rig initialization |
| `handeye` | Tsai-Lenz hand-eye |
| `laserline` | Laser plane estimation |

## See Also

- [vision-calibration-core](https://crates.io/crates/vision-calibration-core): Math types, camera models, RANSAC engine
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): Non-linear refinement
- [vision-calibration-pipeline](https://crates.io/crates/vision-calibration-pipeline): High-level calibration pipelines
- [Book: Linear Calibration](https://vitalyvorobyev.github.io/calibration-rs/linear_overview.html)
