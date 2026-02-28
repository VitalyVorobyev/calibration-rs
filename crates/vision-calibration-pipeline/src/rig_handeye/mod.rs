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
//! 5. **Hand-eye initialization**: Tsai-Lenz linear estimation from pose pairs
//! 6. **Hand-eye optimization**: Bundle adjustment with optional robot pose refinement
//!
//! # Example
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::rig_handeye::{
//!     RigHandeyeProblem, RigHandeyeInput,
//!     step_intrinsics_init_all, step_intrinsics_optimize_all,
//!     step_rig_init, step_rig_optimize,
//!     step_handeye_init, step_handeye_optimize, run_calibration,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let input: RigHandeyeInput = unimplemented!();
//!
//! // Create session and run calibration
//! let mut session = CalibrationSession::<RigHandeyeProblem>::new();
//! session.set_input(input)?;
//!
//! // Option 1: Step-by-step
//! step_intrinsics_init_all(&mut session, None)?;
//! step_intrinsics_optimize_all(&mut session, None)?;
//! step_rig_init(&mut session)?;
//! step_rig_optimize(&mut session, None)?;
//! step_handeye_init(&mut session, None)?;
//! step_handeye_optimize(&mut session, None)?;
//!
//! // Option 2: Full pipeline
//! // run_calibration(&mut session)?;
//!
//! let export = session.export()?;
//! # Ok(())
//! # }
//! ```
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
//! - Intrinsics: NOT re-refined in rig BA (set `refine_intrinsics_in_rig_ba: true` to enable)
//! - Rig extrinsics: NOT re-refined in hand-eye BA (set `refine_cam_se3_rig_in_handeye_ba: true` to enable)
//! - Robot pose refinement: enabled with 0.5° rotation prior and 1mm translation prior
//! - Single fixed target pose (not per-view)
//! - k3 distortion fixed by default

mod problem;
mod state;
mod steps;

// Re-export types
pub use problem::{
    RigHandeyeBaConfig, RigHandeyeConfig, RigHandeyeExport, RigHandeyeInitConfig, RigHandeyeInput,
    RigHandeyeIntrinsicsConfig, RigHandeyeProblem, RigHandeyeRigConfig, RigHandeyeSolverConfig,
};
pub use state::RigHandeyeState;
pub use steps::{
    HandeyeInitOptions, HandeyeOptimOptions, IntrinsicsInitOptions, IntrinsicsOptimOptions,
    RigOptimOptions, run_calibration, step_handeye_init, step_handeye_optimize,
    step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init, step_rig_optimize,
};
