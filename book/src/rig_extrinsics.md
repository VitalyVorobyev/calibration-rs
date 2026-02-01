# Multi-Camera Rig Extrinsics

A multi-camera rig is a set of cameras rigidly attached to a common frame. Rig extrinsics calibration estimates the relative pose of each camera within the rig, in addition to per-camera intrinsics and distortion.

## Problem Formulation

### Parameters

- Per-camera intrinsics: $\{K_k\}$ — one per camera
- Per-camera distortion: $\{\mathbf{d}_k\}$ — one per camera
- Per-camera extrinsics: $\{T_{R,C_k}\}$ — camera-to-rig transforms (one per camera, reference camera = identity)
- Per-view rig poses: $\{T_{R,T}^{(v)}\}$ — rig-to-target transforms (one per view)

### Transform Chain

For camera $k$ in view $v$:

$$T_{C_k, T}^{(v)} = T_{C_k, R} \cdot T_{R, T}^{(v)}$$

where $T_{C_k, R} = (T_{R, C_k})^{-1}$.

### Objective

$$\min \sum_{k=1}^{C} \sum_{v=1}^{M} \sum_{j=1}^{N_{kv}} \left\| \pi\!\left(K_k, \mathbf{d}_k, T_{C_k, R} \cdot T_{R, T}^{(v)}, \mathbf{P}_j\right) - \mathbf{p}_{kvj} \right\|^2$$

This jointly optimizes all cameras' parameters and all rig geometry. Each observation depends on two SE(3) transforms composed together (the `ReprojPointPinhole4Dist5TwoSE3` factor).

## 4-Step Pipeline

### Step 1: Per-Camera Intrinsics Initialization (`step_intrinsics_init_all`)

For each camera independently:
1. Compute homographies from the views where that camera has observations
2. Run iterative intrinsics estimation (Zhang + distortion fit)
3. Recover per-camera, per-view poses

### Step 2: Per-Camera Intrinsics Optimization (`step_intrinsics_optimize_all`)

For each camera independently:
- Run planar intrinsics bundle adjustment
- Refine $K_k$, $\mathbf{d}_k$, and per-view poses

### Step 3: Rig Extrinsics Initialization (`step_rig_init`)

Uses the linear initialization from [Multi-Camera Rig Initialization](rig_init.md):
1. Define the rig frame by the reference camera ($T_{R,C_0} = I$)
2. For each non-reference camera, compute $T_{R,C_k}$ by averaging across views
3. Compute rig-to-target poses from the reference camera's poses

### Step 4: Rig Bundle Adjustment (`step_rig_optimize`)

Joint optimization of all parameters:

**Parameter blocks**:
- `"cam/k"` (4D): per-camera intrinsics
- `"dist/k"` (5D): per-camera distortion
- `"extrinsics/k"` (7D, SE3): per-camera camera-to-rig transforms
- `"rig_pose/v"` (7D, SE3): per-view rig-to-target poses

**Residual blocks**: `ReprojPointPinhole4Dist5TwoSE3` per observation.

**Fixed parameters**: The reference camera's extrinsics are fixed at identity (gauge freedom).

## Configuration

```rust
pub struct RigExtrinsicsConfig {
    // Per-camera settings
    pub fix_intrinsics: Vec<IntrinsicsFixMask>,
    pub fix_distortion: Vec<DistortionFixMask>,

    // Rig settings
    pub fix_extrinsics: Vec<bool>,  // Per-camera (reference always fixed)
    pub fix_rig_poses: Vec<usize>,  // Fix specific view poses

    // Optimization
    pub max_iters: usize,
    pub robust_loss: Option<RobustLoss>,
}
```

## Input Format

```rust
pub struct RigDataset<Meta> {
    pub num_cameras: usize,
    pub views: Vec<RigView<Meta>>,
}

pub struct RigView<Meta> {
    pub obs: RigViewObs,  // Per-camera observations (Option per camera)
    pub meta: Meta,
}
```

Each view has an `Option<CorrespondenceView>` per camera — cameras that don't see the target in a given view have `None`.

## Per-Camera Reprojection Error

After optimization, per-camera reprojection errors are computed independently:

```rust
let export = session.export()?;
for (cam_idx, error) in export.per_cam_reproj_errors.iter().enumerate() {
    println!("Camera {}: {:.4} px", cam_idx, error);
}
println!("Overall: {:.4} px", export.mean_reproj_error);
```

## Stereo Baseline

For a 2-camera rig, the baseline (distance between camera centers) is:

$$\text{baseline} = \|T_{R,C_1}.\text{translation}\|$$

since $T_{R,C_0} = I$ (camera 0 is the reference).

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_extrinsics::*;

let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
session.set_input(rig_dataset)?;

step_intrinsics_init_all(&mut session, None)?;
step_intrinsics_optimize_all(&mut session, None)?;
step_rig_init(&mut session)?;
step_rig_optimize(&mut session, None)?;

let export = session.export()?;
let baseline = export.extrinsics[1].translation.vector.norm();
println!("Stereo baseline: {:.1} mm", baseline * 1000.0);
```

> **OpenCV equivalence**: `cv::stereoCalibrate` for 2-camera rigs. calibration-rs generalizes to $N$ cameras.
