# vision-calibration-core

Core math types, camera models, and RANSAC primitives for `calibration-rs`.

This crate provides the foundational building blocks used by all other crates in the workspace.

## Features

- **Linear algebra types**: `Real`, `Vec2`, `Vec3`, `Pt2`, `Pt3`, `Mat3`, `Iso3` (via nalgebra)
- **Composable camera models**: projection + distortion + sensor + intrinsics pipeline
- **Distortion models**: none, Brown-Conrady, rational, thin-prism and division, with iterative undistortion
- **Sensor models**: Identity and Scheimpflug/tilt (OpenCV-compatible)
- **Deterministic RANSAC**: model-agnostic robust estimation engine
- **Synthetic data**: helpers for generating test/benchmark data

## Camera Model

Cameras are modeled as a composable pipeline:

```text
pixel = K(sensor(distortion(projection(dir))))
```

Where:
- `projection`: camera-frame direction → normalized coordinates (e.g., Pinhole)
- `distortion`: warp normalized coordinates (Brown-Conrady radial + tangential)
- `sensor`: apply homography (Identity or Scheimpflug tilt)
- `K`: intrinsics matrix mapping to pixels (fx, fy, cx, cy)

## Usage

```rust
use vision_calibration_core::{
    Camera, Pinhole, BrownConrady5, IdentitySensor, FxFyCxCy,
    Pt3, Vec3,
};

// Build a camera with pinhole projection and Brown-Conrady distortion
let k = FxFyCxCy { fx: 800.0, fy: 800.0, cx: 640.0, cy: 360.0 };
let dist = BrownConrady5 { k1: -0.1, k2: 0.01, k3: 0.0, p1: 0.0, p2: 0.0, iters: 8 };
let camera = Camera::new(Pinhole, dist, IdentitySensor, k);

// Project a 3D point
let p_cam = Pt3::new(0.1, 0.2, 1.0);
if let Some(pixel) = camera.project_point(&p_cam) {
    println!("Projected to: ({}, {})", pixel.x, pixel.y);
}
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

- [vision-calibration-linear](https://crates.io/crates/vision-calibration-linear): Linear solvers using these primitives
- [vision-calibration-optim](https://crates.io/crates/vision-calibration-optim): Non-linear refinement
- [Book: camera model](https://vitalyvorobyev.github.io/calibration-rs/camera_pipeline.html), [RANSAC](https://vitalyvorobyev.github.io/calibration-rs/ransac.html), [synthetic data](https://vitalyvorobyev.github.io/calibration-rs/synthetic.html)
