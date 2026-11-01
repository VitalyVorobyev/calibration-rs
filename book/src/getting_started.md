# Getting Started

This chapter calibrates a camera end to end from synthetic data, once in Rust and once in Python. The two programs do the same thing: describe a planar board seen from several poses, run the two-step calibration (linear initialization, then non-linear refinement), and print the intrinsics and the reprojection error.

Both interfaces drive the same solvers and expose the same workflow: a session per problem type with its step functions, and a one-call runner per workflow (`run_calibration` in Rust; `run_planar_intrinsics`, `run_rig_extrinsics`, ... in Python). The Rust program below steps through a session; the Python program uses the runner, and [Step by Step in Python](#step-by-step-in-python) shows the session.

## Install

### Rust

```toml
[dependencies]
vision-calibration = { version = "0.10", features = ["synthetic"] }
anyhow = "1"
```

`vision-calibration` is the facade crate: it re-exports the calibration workflows, the dataset manifest, the target detectors' cache and the multiple-view geometry tools. The `synthetic` feature adds the synthetic-data generators the example below calibrates; a calibration from real images does not need it. Enable `refine` for bundle adjustment in the `mvg` module.

### Python

```bash
pip install vision-calibration
```

Python 3.10 or newer. To build the extension from a checkout instead, run `maturin develop -m crates/vision-calibration-py/Cargo.toml`. The package is `vision_calibration`; the examples below import it as `vc`.

## A Minimal Calibration

Each program builds a synthetic camera with known intrinsics and Brown-Conrady distortion, projects an 8x6 board (40 mm squares) from several poses, and recovers the camera from those pixels alone.

### Rust

```rust
use anyhow::Result;
use vision_calibration::core::make_pinhole_camera;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};
use vision_calibration::prelude::*;
use vision_calibration::synthetic::planar;

fn main() -> Result<()> {
    // 1. Define the ground-truth camera
    let k_gt = FxFyCxCy {
        fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0
    };
    let dist_gt = BrownConrady5 {
        k1: 0.05, k2: -0.02, k3: 0.0,
        p1: 0.001, p2: -0.001, iters: 8,
    };
    let camera = make_pinhole_camera(k_gt, dist_gt);

    // 2. Generate synthetic observations
    let board = planar::grid_points(8, 6, 0.04); // 8x6 grid, 40 mm squares
    let poses = planar::poses_yaw_y_z(6, -0.2, 0.08, 0.5, 0.05);
    let views = planar::project_views_all(&camera, &board, &poses)?;

    // 3. Create a session and set the input
    let dataset = PlanarDataset::new(
        views.into_iter().map(View::without_meta).collect()
    )?;
    let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
    session.set_input(dataset)?;

    // 4. Run the calibration
    step_init(&mut session, None)?;      // linear initialization
    step_optimize(&mut session, None)?;  // non-linear refinement

    // 5. Inspect the result
    let export = session.export()?;
    let k = export.params.camera.intrinsics;
    println!("fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
             k.fx, k.fy, k.cx, k.cy);
    println!("Mean reprojection error: {:.4} px",
             export.mean_reproj_error);

    Ok(())
}
```

The library ships this program as an example:

```bash
cargo run -p vision-calibration --example planar_synthetic
```

### Python

```python
import math
import vision_calibration as vc

# 1. The ground-truth camera and a planar board (8x6 grid, 40 mm squares)
fx, fy, cx, cy = 800.0, 780.0, 640.0, 360.0
k1, k2, p1, p2 = 0.05, -0.02, 0.001, -0.001
board = [(i * 0.04, j * 0.04, 0.0) for i in range(8) for j in range(6)]


def project(pt, yaw, tilt, z):
    """Pixel of a board point for a board turned by (yaw, tilt) at depth z."""
    x, y, _ = pt
    cyw, syw, ct, st = math.cos(yaw), math.sin(yaw), math.cos(tilt), math.sin(tilt)
    xc = cyw * x + syw * st * y          # R = Ry(yaw) * Rx(tilt)
    yc = ct * y
    zc = -syw * x + cyw * st * y + z
    xn, yn = xc / zc, yc / zc
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return (fx * xd + cx, fy * yd + cy)


# 2. Synthetic observations: varied yaw, tilt and distance
poses = [(-0.3 + 0.1 * v, 0.25 * math.sin(v), 0.45 + 0.04 * v) for v in range(8)]
views = [
    vc.PlanarView(
        observation=vc.Observation(
            points_3d=board,
            points_2d=[project(p, yaw, tilt, z) for p in board],
        )
    )
    for yaw, tilt, z in poses
]

# 3. Build the input and run the calibration (initialization + refinement)
dataset = vc.PlanarIntrinsicsInput(views=views)
result = vc.run_planar_intrinsics(dataset)

# 4. Inspect the result
k = result.camera.intrinsics
print(f"fx={k.fx:.1f}, fy={k.fy:.1f}, cx={k.cx:.1f}, cy={k.cy:.1f}")
print(f"Mean reprojection error: {result.mean_reproj_error:.4f} px")
```

Real data replaces step 2 in both languages: detect the board corners in each image and pair every corner with its position on the board. [Planar Intrinsics with Real Data](planar_real_data.md) shows the detection step by hand; [Calibration Targets and Detection](detection.md) covers the detectors the library ships, and [The Dataset Manifest](manifest.md) runs a folder of images (`vc.calibrate_dataset` in Python).

### Step by Step in Python

The runner above is a session run in one call. The session gives the steps, their results and the init log:

```python
session = vc.PlanarIntrinsicsSession()
session.set_input(dataset)
init = session.step_init()                 # linear initialization
print(f"init fx={init.intrinsics.fx:.1f}")
session.step_optimize()                    # non-linear refinement
result = session.export()                  # the same PlanarIntrinsicsExport

cam = vc.CameraModel(result.camera)        # project with the calibrated camera
print(cam.project([0.0, 0.0, 1.0]))        # the principal point
```

## Configuring a Run

Every workflow has a `*Config` type with the same grouped shape (`solver`, `camera`, `init`, ...). Rust updates it on the session; Python passes it to the runner or to the session's `set_config`.

### Rust

```rust
use vision_calibration::optim::RobustLoss;
use vision_calibration::planar_intrinsics::run_calibration;

session.update_config(|c| {
    c.solver.max_iters = 80;
    c.solver.robust_loss = RobustLoss::Huber { scale: 1.0 };
})?;
run_calibration(&mut session)?; // the same two steps in one call
```

### Python

```python
config = vc.PlanarIntrinsicsConfig(
    solver=vc.SolverConfig(max_iters=80, robust_loss=vc.robust_huber(1.0)),
)
result = vc.run_planar_intrinsics(dataset, config)
```

## What Happens in Each Step

### Step 1: Linear Initialization

`step_init` (`session.step_init()` in Python; `run_planar_intrinsics` runs it first):

1. Computes a homography from each view's 2D-3D correspondences using DLT (direct linear transform)
2. Estimates intrinsics $K$ from the homographies using Zhang's method
3. Estimates lens distortion coefficients from homography residuals
4. Iterates steps 2-3 to refine the joint $K$ + distortion estimate
5. Recovers each view's camera pose from its homography and $K$

After this step, intrinsics are typically within 10-40% of the true values: good enough to start the non-linear optimization.

### Step 2: Bundle Adjustment

`step_optimize` runs Levenberg-Marquardt, minimizing the total reprojection error:

$$\min_{K,\,\mathbf{d},\,\{T_i\}} \sum_{i=1}^{N} \sum_{j=1}^{M_i} \left\| \pi(K, \mathbf{d}, T_i, \mathbf{P}_j) - \mathbf{p}_{ij} \right\|^2$$

where $\pi$ is the camera projection function, $T_i$ is the pose of view $i$, $\mathbf{d}$ are distortion coefficients, $\mathbf{P}_j$ are known 3D board points, and $\mathbf{p}_{ij}$ are observed pixel coordinates.

After optimization, expect under 2% error on intrinsics and under 1 px mean reprojection error on real data. The synthetic data above is noise-free, so both programs recover the ground truth almost exactly.

## Where to Go Next

- [Stereo Rig with Real Data](stereo_real_data.md): a two-camera rig from detected corners, in under a minute.
- [Planar Intrinsics Calibration](planar_intrinsics.md): the full mathematical treatment of this workflow.
- [Parameter Uncertainty](uncertainty.md): every result carries the covariance of its parameters.
- [Architecture Overview](architecture.md): how the crates fit together.
- [The Dataset Manifest](manifest.md): run a calibration from a folder of images without writing detection code.
