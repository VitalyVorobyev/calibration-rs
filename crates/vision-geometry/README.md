# vision-geometry

Deterministic, allocation-light geometric solvers shared across the
`calibration-rs` workspace.

This crate provides the low-level building blocks behind calibration and
multiple-view geometry workflows: epipolar estimation, homography estimation,
linear triangulation, and camera matrix decomposition.

## Modules

| Module | Description |
|--------|-------------|
| `math` | Hartley normalization and small linear-algebra helpers |
| `epipolar` | Fundamental and essential matrix estimation and decomposition |
| `homography` | Normalized DLT homography estimation with optional RANSAC |
| `triangulation` | Linear DLT triangulation from multiple views |
| `camera_matrix` | Camera matrix estimation and RQ decomposition |

## Coordinate Conventions

- Fundamental matrix solvers accept pixel coordinates.
- Essential matrix solvers expect calibrated coordinates after applying `K^-1`.
- Pose outputs follow the `T_C_W` convention: world to camera.

## Usage

```rust
use vision_calibration_core::Pt2;
use vision_geometry::dlt_homography;

let world = vec![
    Pt2::new(0.0, 0.0),
    Pt2::new(1.0, 0.0),
    Pt2::new(1.0, 1.0),
    Pt2::new(0.0, 1.0),
];
let image = vec![
    Pt2::new(120.0, 200.0),
    Pt2::new(220.0, 198.0),
    Pt2::new(225.0, 300.0),
    Pt2::new(118.0, 302.0),
];

let h = dlt_homography(&world, &image)?;
println!("H = {h}");
# Ok::<(), anyhow::Error>(())
```

## See Also

- `vision-calibration-linear` for calibration-specific initialization solvers
- `vision-mvg` for higher-level multiple-view geometry workflows
- `vision-calibration-core` for shared math aliases, camera models, and RANSAC
