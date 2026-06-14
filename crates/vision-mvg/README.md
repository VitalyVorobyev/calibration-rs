# vision-mvg

Multiple-view geometry tools for calibrated cameras.

This crate builds on `vision-geometry` and adds higher-level multi-view
operations: calibrated relative pose recovery, cheirality checks, robust
estimation wrappers, and triangulation helpers with diagnostics.

## Capabilities

- Relative pose recovery from calibrated 2D correspondences
- Cheirality-based essential matrix disambiguation
- Robust essential and homography estimation helpers
- Triangulation helpers and geometric residual utilities
- Convenience re-exports of low-level solvers from `vision-geometry`

## Coordinate Conventions

- Calibrated APIs expect normalized coordinates after applying `K^-1`.
- Pose outputs follow the `T_C_W` convention: world to camera.

## Usage

```rust
use vision_calibration_core::Pt2;
use vision_mvg::recover_relative_pose;

let left = vec![
    Pt2::new(-0.1, 0.2),
    Pt2::new(0.3, -0.1),
    Pt2::new(0.4, 0.5),
    Pt2::new(-0.2, -0.3),
    Pt2::new(0.1, 0.0),
];
let right = vec![
    Pt2::new(-0.08, 0.19),
    Pt2::new(0.32, -0.11),
    Pt2::new(0.42, 0.48),
    Pt2::new(-0.18, -0.28),
    Pt2::new(0.12, -0.01),
];

let pose = recover_relative_pose(&left, &right)?;
println!("rotation = {:?}", pose.rotation);
println!("translation = {:?}", pose.translation);
# Ok::<(), anyhow::Error>(())
```

## See Also

- `vision-geometry` for low-level deterministic solvers
- `vision-calibration-linear` for calibration-specific initialization
- `vision-calibration` for the high-level facade API
