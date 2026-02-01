# Quickstart

This chapter walks through a minimal camera calibration using synthetic data. By the end you will have estimated camera intrinsics and lens distortion from simulated observations of a planar calibration board.

## Add the Dependency

```toml
[dependencies]
vision-calibration = "0.1"
```

## Minimal Example

The following program generates synthetic calibration data, runs the two-step calibration pipeline (linear initialization followed by non-linear refinement), and prints the results:

```rust
use anyhow::Result;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};
use vision_calibration::prelude::*;
use vision_calibration::synthetic::planar;

fn main() -> Result<()> {
    // 1. Define ground truth camera
    let k_gt = FxFyCxCySkew {
        fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0, skew: 0.0,
    };
    let dist_gt = BrownConrady5 {
        k1: 0.05, k2: -0.02, k3: 0.0,
        p1: 0.001, p2: -0.001, iters: 8,
    };
    let camera = vision_calibration::make_pinhole_camera(k_gt, dist_gt);

    // 2. Generate synthetic observations
    let board = planar::grid_points(8, 6, 0.04); // 8×6 grid, 40 mm squares
    let poses = planar::poses_yaw_y_z(6, -0.2, 0.08, 0.5, 0.05);
    let views = planar::project_views_all(&camera, &board, &poses)?;

    // 3. Create session and set input
    let dataset = PlanarDataset::new(
        views.into_iter().map(View::without_meta).collect()
    )?;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // 4. Run calibration
    step_init(&mut session, None)?;      // Linear initialization
    step_optimize(&mut session, None)?;   // Non-linear refinement

    // 5. Inspect results
    let export = session.export()?;
    let k = export.params.intrinsics();
    println!("fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
             k.fx, k.fy, k.cx, k.cy);
    println!("Mean reprojection error: {:.4} px",
             export.mean_reproj_error);

    Ok(())
}
```

## What Happens in Each Step

### Step 1: `step_init` — Linear Initialization

1. Computes a homography from each view's 2D-3D correspondences using DLT (direct linear transform)
2. Estimates intrinsics $K$ from the homographies using Zhang's method
3. Estimates lens distortion coefficients from homography residuals
4. Iterates steps 2-3 to refine the joint $K$ + distortion estimate
5. Recovers each view's camera pose from its homography and $K$

After this step, intrinsics are typically within 10-40% of the true values — good enough to initialize non-linear optimization.

### Step 2: `step_optimize` — Bundle Adjustment

Runs Levenberg-Marquardt optimization minimizing the total reprojection error:

$$\min_{K,\,\mathbf{d},\,\{T_i\}} \sum_{i=1}^{N} \sum_{j=1}^{M_i} \left\| \pi(K, \mathbf{d}, T_i, \mathbf{P}_j) - \mathbf{p}_{ij} \right\|^2$$

where $\pi$ is the camera projection function, $T_i$ is the pose of view $i$, $\mathbf{d}$ are distortion coefficients, $\mathbf{P}_j$ are known 3D board points, and $\mathbf{p}_{ij}$ are observed pixel coordinates.

After optimization, expect <2% error on intrinsics and <1 px mean reprojection error.

## Running the Built-in Example

The library ships with this example (and several others):

```bash
cargo run -p vision-calibration --example planar_synthetic
```

## Alternative: One-Line Pipeline

If you don't need to inspect intermediate results:

```rust
use vision_calibration::planar_intrinsics::run_calibration;

run_calibration(&mut session)?;
```

## Next Steps

- [Architecture Overview](architecture.md) — understand the crate structure
- [Composable Camera Pipeline](camera_pipeline.md) — the projection model in detail
- [Planar Intrinsics Calibration](planar_intrinsics.md) — the full mathematical treatment
