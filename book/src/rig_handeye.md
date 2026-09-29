# Multi-Camera Rig Hand-Eye

This is the most complex calibration workflow: a multi-camera rig mounted on a robot arm. It combines per-camera intrinsics calibration, rig extrinsics estimation, and hand-eye calibration in a 6-step pipeline.

## Problem Formulation

### Transformation Chain

Frames follow the `frame_se3_frame` convention: `T_{A,B}` maps coordinates of frame $B$ into frame $A$. For camera $k$ in view $v$ with robot pose $T_{B,G}^{(v)}$ (eye-in-hand):

$$T_{C_k, T}^{(v)} = T_{C_k, R} \cdot T_{G, R}^{-1} \cdot (T_{B, G}^{(v)})^{-1} \cdot T_{B, T}$$

where:
- $T_{C_k, R}$ (`cam_se3_rig`): rig to camera $k$ (rig extrinsics)
- $T_{G, R}$ (`gripper_se3_rig`): rig to gripper (the hand-eye transform)
- $T_{B,G}^{(v)}$ (`base_se3_gripper`): known robot pose for view $v$
- $T_{B,T}$ (`base_se3_target`): target in the robot base frame (calibrated)

In eye-to-hand mode the export instead carries `rig_se3_base` and `gripper_se3_target`.

### Parameters

- Per-camera intrinsics: $\{K_k\}$ ($4C$ scalar parameters)
- Per-camera distortion: $\{\mathbf{d}_k\}$ ($5C$ scalar parameters)
- Per-camera extrinsics: $\{T_{C_k, R}\}$ ($6(C-1)$ DOF, reference camera = identity)
- Hand-eye: $T_{G,R}$ (6 DOF)
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
3. Solve $AX = XB$ for $X = T_{G,R}$ (`gripper_se3_rig`)
4. Estimate the target in the base frame, $T_{B,T}$

### Step 6: Hand-Eye Optimization (`step_handeye_optimize`)

Joint optimization of all parameters:

**Parameters**: per-camera intrinsics and distortion, per-camera camera-to-rig extrinsics (SE3), the hand-eye transform (SE3), and the target pose (SE3).

**Factor**: one `ReprojPoint` with a rig hand-eye chain per observation, which composes the full transform chain.

## Configuration

`RigHandeyeConfig` groups the settings by stage:

```rust
pub struct RigHandeyeConfig {
    pub intrinsics: IntrinsicsInitConfig,   // per-camera linear init
    pub manual_init: Option<RigHandeyeIntrinsicsManualInit>, // optional per-camera seeds
    pub sensor: SensorMode,                 // Pinhole (default) or Scheimpflug { .. }
    pub rig: RigConfig,                     // reference_camera_idx, refine_intrinsics_in_rig_ba
    pub handeye_init: HandeyeInitConfig,    // handeye_mode, min_motion_angle_deg
    pub solver: SolverConfig,               // max_iters, verbosity, robust_loss
    pub handeye_ba: HandeyeBaConfig,        // robot_poses, refine_cam_se3_rig, refine_scheimpflug
}
```

```rust
session.update_config(|c| {
    c.handeye_init.handeye_mode = HandEyeMode::EyeInHand;
    c.solver.robust_loss = RobustLoss::Huber { scale: 2.0 };
    c.handeye_ba.robot_poses.refine = true;
})?;
```

Set `sensor` to `SensorMode::Scheimpflug { .. }` for a rig of tilted-sensor cameras (see [Scheimpflug and Laser Rigs](scheimpflug_laser_rigs.md)).

## Complete Example

```rust
use vision_calibration::prelude::*;
use vision_calibration::rig_handeye::*;
use vision_calibration::optim::{HandEyeMode, RobustLoss};

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
println!("Mode: {:?}", export.handeye_mode);
if let Some(gripper_se3_rig) = export.gripper_se3_rig {
    println!("gripper_se3_rig: {:?}", gripper_se3_rig);
}
println!("Baseline: {:.1} mm",
    export.cam_se3_rig[1].translation.vector.norm() * 1000.0);
println!("Per-camera errors: {:?}", export.per_cam_reproj_errors);
```

## Gauge Freedom

The system has a gauge freedom: the rig frame origin and the hand-eye transform are coupled. Fixing the reference camera's extrinsics at identity resolves this by defining the rig frame to coincide with camera 0.

## Practical Considerations

All the advice from [Single-Camera Hand-Eye](handeye_workflow.md) applies, plus:

- **All cameras must observe the target** in at least some views for the rig extrinsics to be well-constrained
- **Views where only some cameras see the target** are handled (missing observations are skipped)
- **The hand-eye transform (`gripper_se3_rig`) relates the gripper and the rig frame**, not the gripper and an individual camera. The per-camera offset comes from the rig extrinsics.
