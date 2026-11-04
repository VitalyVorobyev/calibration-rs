//! Multi-camera rig hand-eye calibration.
//!
//! This module provides a session API for calibrating a multi-camera rig
//! mounted on a robot arm, including per-camera intrinsics, rig extrinsics,
//! and hand-eye transform.
//!
//! # Pipeline
//!
//! The calibration proceeds in six steps:
//!
//! 1. **Intrinsics initialization**: Zhang's method with iterative distortion estimation (per-camera)
//! 2. **Intrinsics optimization**: Non-linear refinement of each camera's parameters
//! 3. **Rig initialization**: Linear estimation of camera-to-rig transforms
//! 4. **Rig optimization**: Bundle adjustment jointly optimizing extrinsics and rig poses
//! 5. **Hand-eye initialization**: closed-form estimation (`handeye_init.method`)
//! 6. **Hand-eye optimization**: Bundle adjustment with optional robot pose refinement
//!
//! [`RigHandeyeProblem`] has a runnable example; `run_calibration` runs the
//! six steps in order.
//!
//! # Conventions
//!
//! - `cam_se3_rig` = T_C_R (transform from rig frame to camera frame)
//! - mode-specific export fields are explicit:
//!   - EyeInHand: `gripper_se3_rig` (T_G_R), `base_se3_target` (T_B_T)
//!   - EyeToHand: `rig_se3_base` (T_R_B), `gripper_se3_target` (T_G_T)
//! - Reference camera (index configurable) has identity extrinsics
//!
//! # Default Behavior
//!
//! - Reference camera: index 0
//! - Hand-eye mode: `EyeInHand`
//! - Intrinsics: NOT re-refined in rig BA (set `refine_intrinsics: true` to enable)
//! - Rig extrinsics: NOT re-refined in hand-eye BA (set `handeye_optimize.refine_cam_se3_rig: true` to enable)
//! - Robot pose refinement: enabled with 0.5° rotation prior and 1mm translation prior
//! - Single fixed target pose (not per-view)
//! - Every camera has the model of `camera` (pinhole, Brown-Conrady, untilted
//!   sensor by default) unless `camera_overrides` names it; each keeps its
//!   model's default fix lists (`k3` for Brown-Conrady) unless `fix` says
//!   otherwise

mod problem;
mod state;
mod steps;

// Re-export types
pub use problem::{
    HandeyeOptimizeConfig, RigHandeyeConfig, RigHandeyeExport, RigHandeyeInput, RigHandeyeProblem,
};
pub use steps::{
    HandeyeInitResult, HandeyeManualInit, HandeyeOptimizeResult, IntrinsicsInitResult,
    IntrinsicsManualInit, IntrinsicsOptimizeResult, RigInitResult, RigManualInit,
    RigOptimizeResult, run_calibration, step_handeye_init, step_handeye_init_with_seed,
    step_handeye_optimize, step_intrinsics_init_all, step_intrinsics_init_all_with_seed,
    step_intrinsics_optimize_all, step_rig_init, step_rig_init_with_seed, step_rig_optimize,
};
