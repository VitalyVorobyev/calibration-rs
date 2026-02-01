# Multi-Camera Rig Hand-Eye

This is the most complex calibration workflow: a multi-camera rig mounted on a robot arm. It combines per-camera intrinsics calibration, rig extrinsics estimation, and hand-eye calibration in a 6-step pipeline.

## Problem Formulation

### Transformation Chain

For camera $k$ in view $v$ with robot pose $T_{B,G}^{(v)}$:

$$T_{C_k, T}^{(v)} = T_{C_k, R} \cdot T_{R, G}^{-1} \cdot (T_{B, G}^{(v)})^{-1} \cdot T_{B, T}$$

where:
- $T_{C_k, R}$: camera $k$ to rig (rig extrinsics)
- $T_{R, G} = T_{\text{handeye}}$: rig to gripper (hand-eye transform)
- $T_{B,G}^{(v)}$: base to gripper (known robot pose for view $v$)
- $T_{B,T}$: base to target (calibrated)

### Parameters

- Per-camera intrinsics: $\{K_k\}$ ($4C$ scalar parameters)
- Per-camera distortion: $\{\mathbf{d}_k\}$ ($5C$ scalar parameters)
- Per-camera extrinsics: $\{T_{C_k, R}\}$ ($6(C-1)$ DOF, reference camera = identity)
- Hand-eye: $T_{R,G}$ (6 DOF)
- Target pose: $T_{B,T}$ (6 DOF)
- Optionally: per-view robot corrections $\{\Delta T_v\}$ ($6M$ DOF, regularized)

## 6-Step Pipeline

### Steps 1-2: Per-Camera Intrinsics

Same as [Rig Extrinsics](rig_extrinsics.md): initialize and optimize each camera's intrinsics independently.

### Steps 3-4: Rig Extrinsics

Same as [Rig Extrinsics](rig_extrinsics.md): initialize camera-to-rig transforms via SE(3) averaging, then jointly optimize the rig geometry.

### Step 5: Hand-Eye Initialization (`step_handeye_init`)

Uses Tsai-Lenz with the rig's reference camera poses and robot poses:
1. Extract relative camera motions from rig-to-target poses
2. Extract relative robot motions from base-to-gripper poses
3. Solve $AX = XB$ for $X = T_{G,R}$ (gripper-to-rig hand-eye)
4. Estimate target-in-base $T_{B,T}$

### Step 6: Hand-Eye Optimization (`step_handeye_optimize`)

Joint optimization of all parameters:

**Parameter blocks**:
- `"cam/k"` (4D): per-camera intrinsics
- `"dist/k"` (5D): per-camera distortion
- `"extrinsics/k"` (7D, SE3): per-camera camera-to-rig
- `"handeye"` (7D, SE3): rig-to-gripper
- `"target"` (7D, SE3): base-to-target

**Factor**: `ReprojPointPinhole4Dist5HandEye` per observation, which composes the full transform chain.

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_handeye::*;

let mut session = CalibrationSession::<RigHandeyeProblem>::new();
session.set_input(rig_dataset_with_robot_poses)?;

// Per-camera calibration
step_intrinsics_init_all(&mut session, None)?;
step_intrinsics_optimize_all(&mut session, None)?;

// Rig geometry
step_rig_init(&mut session)?;
step_rig_optimize(&mut session, None)?;

// Hand-eye
step_handeye_init(&mut session, None)?;
step_handeye_optimize(&mut session, None)?;

let export = session.export()?;
println!("Hand-eye (rig-to-gripper): {:?}", export.gripper_se3_rig);
println!("Baseline: {:.1} mm",
    export.extrinsics[1].translation.vector.norm() * 1000.0);
println!("Per-camera errors: {:?}", export.per_cam_reproj_errors);
```

## Gauge Freedom

The system has a gauge freedom: the rig frame origin and the hand-eye transform are coupled. Fixing the reference camera's extrinsics at identity resolves this by defining the rig frame to coincide with camera 0.

## Practical Considerations

All the advice from [Single-Camera Hand-Eye](handeye_workflow.md) applies, plus:

- **All cameras must observe the target** in at least some views for the rig extrinsics to be well-constrained
- **Views where only some cameras see the target** are handled (missing observations are skipped)
- **The hand-eye transform describes gripper-to-rig**, not gripper-to-individual-camera. The per-camera offset comes from the rig extrinsics.
