# Planar Intrinsics Calibration

This is the most common calibration workflow: estimate camera intrinsics and lens distortion from multiple views of a planar calibration board. It combines Zhang's linear initialization with Levenberg-Marquardt bundle adjustment.

## Problem Formulation

**Parameters**:
- Intrinsics: $K = (f_x, f_y, c_x, c_y)$ — 4 scalars
- Distortion: $\mathbf{d} = (k_1, k_2, k_3, p_1, p_2)$ — 5 scalars
- Per-view poses: $\{T_v\}_{v=1}^M$ — $M$ SE(3) transforms (6 DOF each)

**Total**: $9 + 6M$ parameters.

**Observations**: For each view $v$ and board point $j$:
- Known 3D position $\mathbf{P}_j$ (on the board, at $Z = 0$)
- Observed pixel $\mathbf{p}_{vj}$

**Objective**: Minimize total reprojection error:

$$\min_{K, \mathbf{d}, \{T_v\}} \sum_{v=1}^{M} \sum_{j=1}^{N_v} \left\| \pi(K, \mathbf{d}, T_v, \mathbf{P}_j) - \mathbf{p}_{vj} \right\|^2$$

where $\pi$ is the full camera projection pipeline: SE(3) transform → pinhole → distortion → intrinsics.

With robust loss $\rho$:

$$\min_{K, \mathbf{d}, \{T_v\}} \sum_{v=1}^{M} \sum_{j=1}^{N_v} \rho\left( \left\| \pi(K, \mathbf{d}, T_v, \mathbf{P}_j) - \mathbf{p}_{vj} \right\| \right)$$

## Two-Step Pipeline

### Step 1: Linear Initialization (`step_init`)

1. **Homographies**: Compute $H_v$ for each view via DLT (from board points at $Z = 0$ to observed pixels)
2. **Intrinsics**: Estimate $K$ from homographies using Zhang's method, iteratively refined with distortion estimation (see [Iterative Intrinsics](iterative_intrinsics.md))
3. **Distortion**: Estimate $(k_1, k_2, p_1, p_2)$ from homography residuals (see [Distortion Fit](distortion_fit.md))
4. **Poses**: Decompose each homography to recover $T_v$ (see [Pose from Homography](planar_pose.md))

After initialization, intrinsics are typically within 10-40% of the true values.

### Step 2: Non-Linear Refinement (`step_optimize`)

Constructs the optimization problem as IR:

- **Parameter blocks**: `"cam"` (4D, Euclidean), `"dist"` (5D, Euclidean), `"pose/0"`...`"pose/M-1"` (7D, SE3)
- **Residual blocks**: One `ReprojPoint { model: PINHOLE4_DIST5, chain: SinglePose, .. }` per observation (2D residual)
- **Solver**: the Levenberg–Marquardt loop on the backend `solver.backend` selects (tiny-solver by default; see [Solver Backends](solver_backends.md))

After optimization, expect <2% intrinsics error and <1 px mean reprojection error.

## Configuration

`init` and `solver` are shared sub-structs reused by every intrinsics-bearing problem type:

```rust
pub struct PlanarIntrinsicsConfig {
    // Per-camera linear-initialization stage settings.
    pub init: IntrinsicsInitConfig,
    // { init_iterations: usize = 2, fix_k3: bool = true,
    //   fix_tangential: bool = false, zero_skew: bool = true }

    // Non-linear solve stage settings.
    pub solver: SolverConfig,
    // { max_iters: usize = 50, verbosity: usize = 0, robust_loss: RobustLoss = None }

    pub distortion_model: DistortionKind,   // Distortion model (default: BrownConrady5)
                                            // (see "Distortion Models" below)
    pub fix_camera: CameraFixMask,          // { intrinsics: IntrinsicsFixMask, distortion: DistortionFixMask }
    pub fix_poses: Vec<usize>,              // Fix specific view poses
}
```

### Distortion Models

`distortion_model: DistortionKind` selects the lens model that is fitted:

| Variant | Parameters | Model |
|---------|-----------|-------|
| `DistortionKind::None` | (none) | No distortion |
| `DistortionKind::BrownConrady5` | `k1, k2, k3, p1, p2` | Brown-Conrady radial + tangential (default) |
| `DistortionKind::Rational8` | `k1..k6, p1, p2` | OpenCV rational polynomial |
| `DistortionKind::ThinPrism9` | `k1, k2, k3, p1, p2, s1..s4` | Brown-Conrady + thin prism |
| `DistortionKind::Division1` | `lambda` | Fitzgibbon division model |

```rust
use vision_calibration::optim::DistortionKind;

session.update_config(|c| c.distortion_model = DistortionKind::Rational8)?;
```

For the non-default models, `export.params.distortion()` returns `None`; read the coefficients from `export.params.camera.distortion` (`DistortionParams`, see [Serialization and Runtime-Dynamic Types](params.md)).

### Fix Masks

Fine-grained control over which parameters are optimized, via the combined
`CameraFixMask`:

```rust
// Fix cx, cy but optimize fx, fy
session.update_config(|c| {
    c.fix_camera.intrinsics = IntrinsicsFixMask {
        fx: false, fy: false, cx: true, cy: true,
    };
})?;

// Fix k3 and tangential distortion
session.update_config(|c| {
    c.fix_camera.distortion = DistortionFixMask {
        k1: false, k2: false, k3: true, p1: true, p2: true,
    };
})?;
```

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::planar_intrinsics::{step_init, step_optimize};

let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
session.set_input(dataset)?;

// Optional: customize configuration
session.update_config(|c| {
    c.solver.max_iters = 50;
    c.solver.robust_loss = RobustLoss::Huber { scale: 2.0 };
})?;

// Run pipeline
// Inspect initialization
let init = step_init(&mut session, None)?;
println!("Init fx={:.1}, fy={:.1}", init.intrinsics.fx, init.intrinsics.fy);

step_optimize(&mut session, None)?;

// Export results
let export = session.export()?;
let k = export.params.intrinsics();
println!("Final fx={:.1}, fy={:.1}", k.fx, k.fy);
println!("Reprojection error: {:.4} px", export.mean_reproj_error);
```

## Filtering (Optional Step)

After optimization, views or individual observations with high reprojection error can be filtered out:

```rust
use vision_calibration::planar_intrinsics::{step_filter, FilterOptions};

let mut filter_opts = FilterOptions::default();
filter_opts.max_reproj_error = 2.0;      // Remove observations > 2 px
filter_opts.min_points_per_view = 10;    // Minimum points to keep a view
filter_opts.remove_sparse_views = true;  // Drop views below threshold
step_filter(&mut session, filter_opts)?;

// Filtering edits the input, which clears the computed state: run both steps again
step_init(&mut session, None)?;
step_optimize(&mut session, None)?;

// Or, in one call: run_calibration_with_filtering(&mut session, filter_opts)?
```

> **OpenCV equivalence**: `cv::calibrateCamera` performs both initialization and optimization internally. calibration-rs separates these steps for inspection and customization.

## Accuracy Expectations

| Stage | Intrinsics error | Reprojection error |
|-------|-----------------|-------------------|
| After `step_init` | 10-40% | Not computed |
| After `step_optimize` | <2% | <1 px mean |
| After filtering + re-optimize | <1% | <0.5 px mean |

## Input Requirements

- **Minimum 3 views** (for Zhang's method with skew)
- **Minimum 4 points per view** (for homography estimation)
- **View diversity**: Views should include rotation around both axes and vary in distance
- **Board coverage**: Points should span the full image area for good distortion estimation
