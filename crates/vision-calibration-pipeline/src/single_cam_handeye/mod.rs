//! Single-camera hand-eye calibration (intrinsics + hand-eye).
//!
//! This module provides a session API for calibrating a single camera
//! mounted on a robot arm, including both camera intrinsics and hand-eye transform.
//!
//! # Pipeline
//!
//! The calibration proceeds in four steps:
//!
//! 1. **Intrinsics initialization**: Zhang's method with iterative distortion
//!    estimation, embedded into the configured camera model (`camera.model`);
//!    for a Scheimpflug sensor, the linear tilt sweep (or a tilt seed)
//! 2. **Intrinsics optimization**: Non-linear refinement of camera parameters
//!    (the staged tilt multi-start for a Scheimpflug sensor without a seed)
//! 3. **Hand-eye initialization**: closed-form estimation (`handeye_init.method`)
//! 4. **Hand-eye optimization**: Bundle adjustment with optional robot pose refinement
//!
//! [`SingleCamHandeyeProblem`] has a runnable example; `run_calibration`
//! runs the four steps in order.
//!
//! # Conventions
//!
//! - **Robot pose**: `base_se3_gripper` (T_B_G) - gripper in base frame
//! - **Hand-eye transform** (EyeInHand mode): `gripper_se3_cam` (T_G_C)
//! - **Target pose**: `base_se3_target` (T_B_T) - single static target in base frame
//!
//! # Default Behavior
//!
//! - Hand-eye mode: `EyeInHand`
//! - Robot pose refinement: enabled with 0.5° rotation prior and 1mm translation prior
//! - Single fixed target pose (not per-view)
//! - Camera model: pinhole, Brown-Conrady, identity sensor (`camera.model`);
//!   any distortion kind and a Scheimpflug sensor are supported
//! - The model's default fix lists (`camera.fix`): k3 for Brown-Conrady

mod problem;
mod state;
mod steps;

// Re-export types
pub use problem::{
    SingleCamHandeyeConfig, SingleCamHandeyeExport, SingleCamHandeyeInput, SingleCamHandeyeProblem,
    SingleCamHandeyeSolution, SingleCamHandeyeView,
};
pub use steps::{
    HandeyeInitResult, HandeyeManualInit, HandeyeOptimizeResult, IntrinsicsInitResult,
    IntrinsicsManualInit, IntrinsicsOptimizeResult, run_calibration, step_handeye_init,
    step_handeye_init_with_seed, step_handeye_optimize, step_intrinsics_init,
    step_intrinsics_init_with_seed, step_intrinsics_optimize,
};
