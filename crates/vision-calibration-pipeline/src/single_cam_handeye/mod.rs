//! Single-camera hand-eye calibration (intrinsics + hand-eye).
//!
//! This module provides a session API for calibrating a single camera
//! mounted on a robot arm, including both camera intrinsics and hand-eye transform.
//!
//! # Pipeline
//!
//! The calibration proceeds in four steps:
//!
//! 1. **Intrinsics initialization**: Zhang's method with iterative distortion estimation
//! 2. **Intrinsics optimization**: Non-linear refinement of camera parameters
//! 3. **Hand-eye initialization**: Tsai-Lenz linear estimation from pose pairs
//! 4. **Hand-eye optimization**: Bundle adjustment with optional robot pose refinement
//!
//! # Example
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::single_cam_handeye::{
//!     HandeyeMeta, SingleCamHandeyeProblem, SingleCamHandeyeInput, SingleCamHandeyeView,
//!     step_intrinsics_init, step_intrinsics_optimize,
//!     step_handeye_init, step_handeye_optimize, run_calibration,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let (obs, base_se3_gripper) = unimplemented!();
//!
//! // Create input from robot poses and observations
//! let views = vec![
//!     SingleCamHandeyeView::new(obs, HandeyeMeta { base_se3_gripper }),
//!     // ... more views
//! ];
//! let input = SingleCamHandeyeInput::new(views)?;
//!
//! // Create session and run calibration
//! let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
//! session.set_input(input)?;
//!
//! // Option 1: Step-by-step
//! step_intrinsics_init(&mut session, None)?;
//! step_intrinsics_optimize(&mut session, None)?;
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
//! - **Robot pose**: `base_se3_gripper` (T_B_G) - gripper in base frame
//! - **Hand-eye transform** (EyeInHand mode): `gripper_se3_camera` (T_G_C)
//! - **Target pose**: `base_se3_target` (T_B_T) - single static target in base frame
//!
//! # Default Behavior
//!
//! - Hand-eye mode: `EyeInHand`
//! - Robot pose refinement: enabled with 0.5° rotation prior and 1mm translation prior
//! - Single fixed target pose (not per-view)
//! - k3 distortion fixed by default

mod problem;
mod state;
mod steps;

// Re-export types
pub use problem::{
    HandeyeMeta, SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeInput,
    SingleCamHandeyeProblem, SingleCamHandeyeView,
};
pub use state::SingleCamHandeyeState;
pub use steps::{
    HandeyeInitOptions, HandeyeOptimOptions, IntrinsicsInitOptions, IntrinsicsOptimOptions,
    run_calibration, step_handeye_init, step_handeye_optimize, step_intrinsics_init,
    step_intrinsics_optimize,
};
