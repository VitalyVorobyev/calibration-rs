# vision-calibration-core

Core math types, camera models, and RANSAC primitives for `calibration-rs`.

This crate provides the foundational building blocks used by all other crates in the workspace.

## Features

- **Linear algebra types**: `Real`, `Vec2`, `Vec3`, `Pt2`, `Pt3`, `Mat3`, `Iso3` (via nalgebra)
- **Composable camera models**: projection + distortion + sensor + intrinsics pipeline
- **Distortion models**: Brown-Conrady (k1, k2, k3, p1, p2) with iterative undistortion
- **Sensor models**: Identity and Scheimpflug/tilt (OpenCV-compatible)
- **Deterministic RANSAC**: model-agnostic robust estimation engine
- **Synthetic data**: helpers for generating test/benchmark data

## Camera Model

Cameras are modeled as a composable pipeline:

```
pixel = K(sensor(distortion(projection(dir))))
```

Where:
- `projection`: camera-frame direction → normalized coordinates (e.g., Pinhole)
- `distortion`: warp normalized coordinates (Brown-Conrady radial + tangential)
- `sensor`: apply homography (Identity or Scheimpflug tilt)
- `K`: intrinsics matrix mapping to pixels (fx, fy, cx, cy, skew)

## Usage

```rust
use vision_calibration_core::{
    Camera, Pinhole, BrownConrady5, IdentitySensor, FxFyCxCySkew,
    Pt3, Vec3,
};

// Build a camera with pinhole projection and Brown-Conrady distortion
let k = FxFyCxCySkew { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0, skew: 0.0 };
let dist = BrownConrady5 { k1: -0.1, k2: 0.01, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 };
let camera = Camera::new(Pinhole, dist, IdentitySensor, k);

// Project a 3D point
let p_cam = Pt3::new(0.1, 0.2, 1.0);
if let Some(pixel) = camera.project_point(&p_cam) {
    println!("Projected to: ({}, {})", pixel.x, pixel.y);
}
```

### Synthetic Data Generation

```rust
use vision_calibration_core::synthetic::planar;

// Generate a 6x5 chessboard with 40mm squares
let board_points = planar::grid_points(6, 5, 0.04);

// Generate 5 camera poses looking at the board
let poses = planar::poses_yaw_y_z(5, -0.3, 0.15, 0.5, 0.1);

// Project all views (requires a camera)
let views = planar::project_views_all(&camera, &board_points, &poses)?;
```

### RANSAC

```rust
use vision_calibration_core::ransac::{Ransac, RansacModel, RansacConfig};

// Implement RansacModel for your problem, then:
let config = RansacConfig {
    max_iterations: 1000,
    inlier_threshold: 3.0,
    min_inliers: 10,
    seed: Some(42), // Deterministic
};
let ransac = Ransac::new(config);
let result = ransac.run(&data)?;
```

## Modules

| Module | Description |
|--------|-------------|
| `math` | Type aliases and homogeneous coordinate helpers |
| `models` | Camera, projection, distortion, sensor traits and impls |
| `ransac` | Generic RANSAC engine with configurable parameters |
| `synthetic` | Deterministic synthetic data generation |
| `types` | Common observation and result types |

## See Also

- [vision-calibration-linear](../vision-calibration-linear): Linear solvers using these primitives
- [vision-calibration-optim](../vision-calibration-optim): Non-linear refinement
- [Book: Core Concepts](../../book/src/concepts.md)
