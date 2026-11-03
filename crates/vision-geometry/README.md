# vision-geometry

Deterministic, allocation-light geometric solvers shared across the
`calibration-rs` workspace.

Most users want [`vision-calibration`](https://crates.io/crates/vision-calibration) instead, which re-exports these solvers at
the root of `vision_calibration::mvg`; this crate is one of its building blocks.

This crate provides the low-level building blocks behind calibration and
multiple-view geometry workflows: epipolar estimation, homography estimation,
linear triangulation, camera pose from 3D-2D points (PnP), and camera matrix
decomposition.

## Modules

| Module | Description |
|--------|-------------|
| `math` | Hartley normalization and small linear-algebra helpers |
| `epipolar` | Fundamental and essential matrix estimation and decomposition |
| `homography` | Normalized DLT homography estimation with optional RANSAC |
| `triangulation` | Linear DLT triangulation of bearings from multiple views |
| `pnp` | Camera pose from 3D-2D points: DLT, P3P, EPnP, DLT in RANSAC |
| `camera_matrix` | Camera matrix estimation and RQ decomposition |

## Coordinate Conventions

- Fundamental matrix solvers accept pixel coordinates.
- Essential matrix solvers expect calibrated coordinates after applying `K^-1`.
- Pose outputs follow the `T_C_W` convention: world to camera.
- PnP solvers take pixel coordinates and intrinsics `K`; pass the identity `K` with
  normalized coordinates (`x/z`, `y/z`) to use them with any camera model.

## Usage

```rust
use vision_calibration_core::Pt2;
use vision_geometry::homography::dlt_homography;

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
# Ok::<(), vision_geometry::GeometryError>(())
```

Camera pose from known 3D points and their pixels:

```rust
use vision_calibration_core::{FxFyCxCy, Iso3, Pt2, Pt3};
use vision_geometry::pnp;

let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0 };
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

let pose = pnp::dlt(&world, &image, &k)?;
assert!((pose.translation.vector - truth.translation.vector).norm() < 1e-6);
# Ok::<(), vision_geometry::GeometryError>(())
```

## See Also

- `vision-calibration-linear` for calibration-specific initialization solvers
- `vision-mvg` for higher-level multiple-view geometry workflows
- `vision-calibration-core` for shared math aliases, camera models, and RANSAC
