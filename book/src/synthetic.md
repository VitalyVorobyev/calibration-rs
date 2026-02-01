# Synthetic Data Generation

> **[COLLAB]** This appendix benefits from user collaboration on recommended noise levels and realistic test scenarios.

calibration-rs provides utilities for generating synthetic calibration data, used in examples and tests. Synthetic data allows testing with known ground truth, verifying convergence, and debugging algorithm issues.

## Board Point Generation

```rust
use vision_calibration::synthetic::planar;

// 8×6 grid of 3D points at Z=0, spaced 40mm apart
let board_points = planar::grid_points(8, 6, 0.04);
// Returns Vec<Pt3>: [(0,0,0), (0.04,0,0), ..., (0.28,0.20,0)]
```

Points are generated in the $Z = 0$ plane with the specified column count, row count, and spacing (in meters).

## Pose Generation

```rust
// 6 camera poses with varying yaw and distance
let poses = planar::poses_yaw_y_z(
    6,      // number of views
    -0.2,   // start yaw (radians)
    0.08,   // yaw step
    0.5,    // start Z distance (meters)
    0.05,   // Z step
);
// Returns Vec<Iso3>: camera-to-board transforms
```

Poses are generated looking at the board center, with rotation around the Y axis (yaw) and varying distance along Z.

## Projection

```rust
// Project board points through all poses
let views = planar::project_views_all(&camera, &board_points, &poses)?;
// Returns Vec<CorrespondenceView>: 2D-3D point pairs per view
```

This applies the full camera model (projection + distortion + intrinsics) to generate pixel observations.

## Adding Noise

<!-- [COLLAB]: Provide recommended noise levels for different scenarios -->

For realistic testing, add Gaussian noise to the projected pixels:

```rust
use rand::Rng;
let mut rng = rand::thread_rng();
let noise_sigma = 0.5; // pixels

for view in &mut views {
    for pixel in view.points_2d.iter_mut() {
        pixel.x += rng.gen::<f64>() * noise_sigma;
        pixel.y += rng.gen::<f64>() * noise_sigma;
    }
}
```

Typical noise levels:

| Scenario | Noise sigma (px) |
|----------|-----------------|
| Ideal (testing convergence) | 0.0 |
| High-quality detector | 0.1 - 0.3 |
| Standard detector | 0.3 - 1.0 |
| Noisy conditions | 1.0 - 3.0 |

## Deterministic Seeds

All examples and tests use fixed random seeds for reproducibility:

```rust
use rand::SeedableRng;
let rng = rand::rngs::StdRng::seed_from_u64(42);
```

## Common Test Scenarios

<!-- [COLLAB]: Add recommended scenarios for validating new algorithms -->

### Minimal (3 views, no noise)
For verifying algorithm correctness. Should converge to machine precision.

### Moderate (6-10 views, 0.5 px noise)
For testing realistic convergence. Should achieve <1% intrinsics error after optimization.

### Challenging (20 views, 1.0 px noise, outliers)
For testing robustness. Should achieve <2% intrinsics error with robust loss functions.

### Distortion stress test
Use large distortion ($k_1 = -0.3$, $k_2 = 0.1$) to verify the iterative intrinsics solver handles strong distortion.
