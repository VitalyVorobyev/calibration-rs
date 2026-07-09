# Single-Camera Hand-Eye Calibration

Hand-eye calibration estimates the rigid transform between a camera and a robot's gripper (or base). This workflow combines intrinsics calibration with hand-eye estimation in a 4-step pipeline.

## Problem Formulation

### Transformation Chain (Eye-in-Hand)

For a camera mounted on the robot gripper:

$$T_{C,T} = T_{C,G} \cdot T_{G,B} \cdot T_{B,T}$$

where:
- $T_{C,G} = T_{\text{handeye}}^{-1}$: camera-to-gripper (the calibrated hand-eye transform, inverted)
- $T_{G,B} = T_{\text{robot}}^{-1}$: gripper-to-base (known from robot kinematics, inverted)
- $T_{B,T}$: base-to-target (the target's pose in the robot base frame, also calibrated)

Equivalently:

$$T_{C,T} = (T_{\text{robot}} \cdot T_{\text{handeye}})^{-1} \cdot T_{B,T}$$

### Eye-to-Hand Variant

For a fixed camera observing a target on the gripper:

$$T_{C,T} = T_{C,B} \cdot T_{B,G} \cdot T_{G,T}$$

where $T_{C,B}$ is the camera-to-base transform and $T_{G,T}$ is the target-to-gripper transform.

### Optimization Objective

Minimize reprojection error across all views, jointly over all parameters:

$$\min_{K, \mathbf{d}, X, T_{B,T}} \sum_{v=1}^{M} \sum_{j=1}^{N_v} \left\| \pi\!\left(K, \mathbf{d}, T_{C,T}^{(v)}, \mathbf{P}_j\right) - \mathbf{p}_{vj} \right\|^2$$

where $T_{C,T}^{(v)}$ depends on the known robot pose for view $v$, the hand-eye transform $X$, and the target pose $T_{B,T}$.

## 4-Step Pipeline

### Step 1: Intrinsics Initialization (`step_intrinsics_init`)

Runs the [Iterative Intrinsics](iterative_intrinsics.md) solver to estimate $K$ and distortion from the calibration board observations, ignoring robot poses entirely.

### Step 2: Intrinsics Optimization (`step_intrinsics_optimize`)

Runs planar intrinsics bundle adjustment to refine $K$, distortion, and per-view camera-to-target poses. This gives accurate per-view poses $T_{C,T}^{(v)}$ needed for hand-eye initialization.

### Step 3: Hand-Eye Initialization (`step_handeye_init`)

Uses the Tsai-Lenz linear method (see [Hand-Eye Linear](handeye_linear.md)):

1. Compute relative motions from robot poses and camera poses
2. Filter pairs by rotation magnitude (reject small rotations)
3. Estimate rotation via quaternion SVD
4. Estimate translation via linear least squares
5. Estimate $T_{B,T}$ from the hand-eye and camera poses

### Step 4: Hand-Eye Optimization (`step_handeye_optimize`)

Joint non-linear optimization of all parameters:

**Parameter blocks**:
- `"cam"` (4D, Euclidean): intrinsics
- `"dist"` (5D, Euclidean): distortion
- `"handeye"` (7D, SE3): gripper-to-camera transform
- `"target"` (7D, SE3): base-to-target transform

**Residual blocks**: one `ReprojPoint { model: PINHOLE4_DIST5, chain: HandEye { .. }, .. }` per observation.

The robot poses are **not optimized** — they are known constants from the robot kinematics (passed as per-residual data).

### Optional: Robot Pose Refinement

If robot pose accuracy is suspect, the pipeline supports per-view SE(3) corrections:

**Additional parameter blocks**: `"robot_delta/v"` (7D, SE3) per view

**Additional residuals**: `Se3TangentPrior` per view, with configurable rotation and translation sigmas:

$$\mathbf{r}_{\text{prior}} = \begin{bmatrix} \boldsymbol{\omega} / \sigma_r \\ \mathbf{v} / \sigma_t \end{bmatrix}$$

This penalizes deviations from the nominal robot poses, allowing small corrections while preventing the optimizer from absorbing all error into robot pose changes.

## Configuration

Grouped per ADR 0024 — shared sub-structs (`IntrinsicsInitConfig`,
`HandeyeInitConfig`, `SolverConfig`, `RobotPoseConfig`) are reused across
every hand-eye problem type:

```rust
pub struct SingleCamHandeyeConfig {
    // Per-camera linear-initialization stage settings.
    pub intrinsics: IntrinsicsInitConfig,
    // { init_iterations: usize = 2, fix_k3: bool = true,
    //   fix_tangential: bool = false, zero_skew: bool = true }

    // Hand-eye linear-initialization stage settings (Tsai-Lenz DLT).
    pub handeye_init: HandeyeInitConfig,
    // { handeye_mode: HandEyeMode, min_motion_angle_deg: f64 = 5.0 }

    // Non-linear solve stage settings.
    pub solver: SolverConfig, // { max_iters, verbosity, robust_loss }

    // Robot-pose refinement settings for the hand-eye bundle adjustment.
    pub robot_poses: RobotPoseConfig,
    // { refine: bool = true, rot_sigma: f64, trans_sigma: f64 }
}
```

## Input Format

Each view is a `View<HandeyeMeta>` (aliased as `SingleCamHandeyeView`):

```rust
pub struct HandeyeMeta {
    pub base_se3_gripper: Iso3,  // Robot pose (from kinematics)
}

// Type alias
pub type SingleCamHandeyeView = View<HandeyeMeta>;
// Where View<Meta> has fields: obs: CorrespondenceView, meta: Meta
```

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::single_cam_handeye::*;

let views: Vec<SingleCamHandeyeView> = /* load data */;
let input = SingleCamHandeyeInput::new(views)?;

let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
session.set_input(input)?;

step_intrinsics_init(&mut session, None)?;
step_intrinsics_optimize(&mut session, None)?;
step_handeye_init(&mut session, None)?;
step_handeye_optimize(&mut session, None)?;

let export = session.export()?;
println!("Hand-eye: {:?}", export.gripper_se3_camera);
println!("Target in base: {:?}", export.base_se3_target);
println!("Reprojection error: {:.4} px", export.mean_reproj_error);
```

## Practical Considerations

- **Rotation diversity** is critical for Tsai-Lenz initialization. Include rotations around at least 2 axes.
- **5-10 views minimum**, with 10-20 recommended for robustness.
- **Robot accuracy**: If the robot's reported poses have errors > 1mm or > 0.1°, enable robot pose refinement.
- **Initialization failure**: If Tsai-Lenz produces a poor estimate (large reprojection error after optimization), the most likely cause is insufficient rotation diversity.

> **OpenCV equivalence**: `cv::calibrateHandEye` provides the linear initialization step. calibration-rs adds joint non-linear refinement and optional robot pose correction.
