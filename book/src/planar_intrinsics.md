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
- **Residual blocks**: One `ReprojPointPinhole4Dist5` per observation (2D residual)
- **Backend**: Levenberg-Marquardt via TinySolverBackend

After optimization, expect <2% intrinsics error and <1 px mean reprojection error.

## Configuration

```rust
pub struct PlanarConfig {
    // Initialization
    pub init_iterations: usize,        // Iterative intrinsics iterations (default: 2)
    pub fix_k3_in_init: bool,          // Fix k3 during init (default: true)
    pub fix_tangential_in_init: bool,  // Fix p1, p2 during init (default: false)
    pub zero_skew: bool,               // Enforce zero skew (default: true)

    // Optimization
    pub max_iters: usize,              // LM iterations (default: 100)
    pub verbosity: u32,                // Solver output level
    pub robust_loss: RobustLoss,          // Robust loss function (default: None)
    pub fix_intrinsics: IntrinsicsFixMask,  // Fix specific intrinsics
    pub fix_distortion: DistortionFixMask,  // Fix specific distortion params
    pub fix_poses: Vec<usize>,         // Fix specific view poses
}
```

### Fix Masks

Fine-grained control over which parameters are optimized:

```rust
// Fix cx, cy but optimize fx, fy
session.update_config(|c| {
    c.fix_intrinsics = IntrinsicsFixMask {
        fx: false, fy: false, cx: true, cy: true,
    };
})?;

// Fix k3 and tangential distortion
session.update_config(|c| {
    c.fix_distortion = DistortionFixMask {
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
    c.max_iters = 50;
    c.robust_loss = RobustLoss::Huber { scale: 2.0 };
})?;

// Run pipeline
step_init(&mut session, None)?;

// Inspect initialization
let init_k = session.state.initial_intrinsics.as_ref().unwrap();
println!("Init fx={:.1}, fy={:.1}", init_k.fx, init_k.fy);

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

let filter_opts = FilterOptions {
    max_reproj_error: 2.0,      // Remove observations > 2 px
    min_points_per_view: 10,     // Minimum points to keep a view
    remove_sparse_views: true,   // Drop views below threshold
};
step_filter(&mut session, filter_opts)?;

// Re-optimize with cleaned data
step_optimize(&mut session, None)?;
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
