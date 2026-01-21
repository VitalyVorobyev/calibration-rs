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
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::rig_handeye::{
//!     RigHandeyeProblem, RigHandeyeInput,
//!     step_intrinsics_init_all, step_intrinsics_optimize_all,
//!     step_rig_init, step_rig_optimize,
//!     step_handeye_init, step_handeye_optimize, run_calibration,
//! };
//!
//! // Load rig dataset with robot poses
//! let input: RigHandeyeInput = /* load from file or construct */;
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
//! ```
//!
//! # Conventions
//!
//! - `cam_se3_rig` = T_C_R (transform from rig frame to camera frame)
//! - `handeye` = T_G_R (gripper to rig, for EyeInHand mode)
//! - `target_se3_base` = T_T_B (single static target in base frame)
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
pub use problem::{RigHandeyeConfig, RigHandeyeExport, RigHandeyeInput, RigHandeyeProblem};
pub use state::RigHandeyeState;
pub use steps::{
    run_calibration, step_handeye_init, step_handeye_optimize, step_intrinsics_init_all,
    step_intrinsics_optimize_all, step_rig_init, step_rig_optimize, HandeyeInitOptions,
    HandeyeOptimOptions, IntrinsicsInitOptions, IntrinsicsOptimOptions, RigOptimOptions,
};
