//! Multi-camera rig extrinsics calibration.
//!
//! This module provides a v2 session API for calibrating a multi-camera rig,
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
//! # Example
//!
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::rig_extrinsics::{
//!     RigExtrinsicsProblemV2, RigExtrinsicsInput,
//!     step_intrinsics_init_all, step_intrinsics_optimize_all,
//!     step_rig_init, step_rig_optimize, run_calibration,
//! };
//!
//! // Load rig dataset (multiple views with multiple cameras)
//! let input: RigExtrinsicsInput = /* load from file or construct */;
//!
//! // Create session and run calibration
//! let mut session = CalibrationSession::<RigExtrinsicsProblemV2>::new();
//! session.set_input(input)?;
//!
//! // Option 1: Step-by-step
//! step_intrinsics_init_all(&mut session, None)?;
//! step_intrinsics_optimize_all(&mut session, None)?;
//! step_rig_init(&mut session)?;
//! step_rig_optimize(&mut session, None)?;
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
//! - Reference camera (index configurable) has identity extrinsics
//! - `rig_se3_target` = T_R_T (rig pose relative to target, per view)
//!
//! # Default Behavior
//!
//! - Reference camera: index 0
//! - Intrinsics: NOT re-refined in rig BA (set `refine_intrinsics_in_rig_ba: true` to enable)
//! - First rig pose fixed for gauge freedom
//! - k3 distortion fixed by default

mod problem_v2;
mod state;
mod steps;

// Re-export types
pub use problem_v2::{
    RigExtrinsicsConfig, RigExtrinsicsExport, RigExtrinsicsInput, RigExtrinsicsProblemV2,
};
pub use state::RigExtrinsicsState;
pub use steps::{
    run_calibration, step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init,
    step_rig_optimize, IntrinsicsInitOptions, IntrinsicsOptimOptions, RigOptimOptions,
};
