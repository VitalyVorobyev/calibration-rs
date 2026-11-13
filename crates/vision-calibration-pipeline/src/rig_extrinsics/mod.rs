//! Multi-camera rig extrinsics calibration.
//!
//! This module provides a session API for calibrating a multi-camera rig,
//! estimating per-camera intrinsics and camera-to-rig transforms.
//!
//! # Pipeline
//!
//! The calibration proceeds in four steps:
//!
//! 1. **Intrinsics initialization**: Zhang's method with iterative distortion estimation (per-camera)
//! 2. **Intrinsics optimization**: Non-linear refinement of each camera's parameters
//! 3. **Rig initialization**: Linear estimation of camera-to-rig transforms
//! 4. **Rig optimization**: Bundle adjustment jointly optimizing extrinsics and rig poses
//!
//! [`RigExtrinsicsProblem`] has a runnable example; `run_calibration` runs
//! the four steps in order.
//!
//! # Conventions
//!
//! - `cam_se3_rig` = T_C_R (transform from rig frame to camera frame)
//! - Reference camera (index configurable) has identity extrinsics
//! - `rig_se3_target` = T_R_T (rig pose relative to target, per view)
//!
//! # Default Behavior
//!
//! - Reference camera: index 0 (`rig.reference_camera_idx`); its extrinsic
//!   fix alone removes the rig's full 6-DOF gauge freedom (no rig-from-target
//!   pose is additionally pinned)
//! - Intrinsics: re-refined in rig BA with each camera's fix lists
//!   (`rig.refine_intrinsics`), and so is a Scheimpflug tilt
//!   (`rig.refine_sensor`): the whole rig's views determine it far better
//!   than the camera's own, and the covariance then includes it
//! - Every camera has the model of `camera` (pinhole, Brown-Conrady, untilted
//!   sensor by default) unless `camera_overrides` names it; each keeps its
//!   model's default fix lists (`k3` for Brown-Conrady) unless `fix` says
//!   otherwise

mod problem;
mod state;
mod steps;

// Re-export types
pub use problem::{
    RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsInput, RigExtrinsicsProblem,
};
pub use steps::{
    IntrinsicsInitResult, IntrinsicsOptimizeResult, RigInitResult, RigManualInit,
    RigOptimizeResult, RigPoses, run_calibration, step_intrinsics_init_all,
    step_intrinsics_init_all_with_seed, step_intrinsics_optimize_all, step_rig_init,
    step_rig_init_with_seed, step_rig_optimize,
};
/// The rig bundle adjustment's estimate, the problem's output.
pub use vision_calibration_optim::{RigExtrinsicsEstimate, RigExtrinsicsParams};
