//! High-level camera calibration pipelines.
//!
//! This crate provides ready-to-use calibration workflows with multiple APIs:
//!
//! ## Session API - Recommended
//!
//! The new session API uses a mutable state container with step functions.
//! This is the recommended approach for new code.
//!
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblemV2, step_init, step_optimize, run_calibration,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step control
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function
//! // run_calibration(&mut session)?;
//!
//! let export = session.export()?;
//! ```

// Core session framework
pub mod session;

// Problem-specific modules
pub mod planar_intrinsics;
pub mod rig_extrinsics;
pub mod rig_handeye;
pub mod single_cam_handeye;

// ─────────────────────────────────────────────────────────────────────────────
// Session API Re-exports - Recommended
// ─────────────────────────────────────────────────────────────────────────────

/// Session API with mutable state container.
///
/// This module re-exports the session infrastructure and calibration problem
/// step functions for convenient access.
// Session infrastructure
pub use crate::session::{
    CalibrationSession, ExportRecord, InvalidationPolicy, LogEntry, ProblemType, SessionMetadata,
};

// Planar intrinsics
pub use crate::planar_intrinsics::{
    run_calibration as run_planar_intrinsics, run_calibration_with_filtering, step_filter,
    step_init, step_optimize, FilterOptions, InitOptions, OptimizeOptions, PlanarConfig,
    PlanarExport, PlanarIntrinsicsProblem, PlanarState,
};

// Single-camera hand-eye
pub use crate::single_cam_handeye::{
    run_calibration as run_single_cam_handeye, step_handeye_init as single_cam_step_handeye_init,
    step_handeye_optimize as single_cam_step_handeye_optimize,
    step_intrinsics_init as single_cam_step_intrinsics_init,
    step_intrinsics_optimize as single_cam_step_intrinsics_optimize,
    HandeyeInitOptions as SingleCamHandeyeInitOptions,
    HandeyeOptimOptions as SingleCamHandeyeOptimOptions,
    IntrinsicsInitOptions as SingleCamIntrinsicsInitOptions,
    IntrinsicsOptimOptions as SingleCamIntrinsicsOptimOptions, SingleCamHandeyeConfig,
    SingleCamHandeyeExport, SingleCamHandeyeInput, SingleCamHandeyeProblemV2,
    SingleCamHandeyeState, SingleCamHandeyeView,
};

// Rig extrinsics
pub use crate::rig_extrinsics::{
    run_calibration as run_rig_extrinsics,
    step_intrinsics_init_all as rig_step_intrinsics_init_all,
    step_intrinsics_optimize_all as rig_step_intrinsics_optimize_all, step_rig_init,
    step_rig_optimize, IntrinsicsInitOptions as RigIntrinsicsInitOptions,
    IntrinsicsOptimOptions as RigIntrinsicsOptimOptions, RigExtrinsicsConfig, RigExtrinsicsExport,
    RigExtrinsicsInput, RigExtrinsicsProblem, RigExtrinsicsState, RigOptimOptions,
};

// Rig hand-eye
pub use crate::rig_handeye::{
    run_calibration as run_rig_handeye, step_handeye_init as rig_handeye_step_handeye_init,
    step_handeye_optimize as rig_handeye_step_handeye_optimize,
    step_intrinsics_init_all as rig_handeye_step_intrinsics_init_all,
    step_intrinsics_optimize_all as rig_handeye_step_intrinsics_optimize_all,
    step_rig_init as rig_handeye_step_rig_init, step_rig_optimize as rig_handeye_step_rig_optimize,
    HandeyeInitOptions as RigHandeyeInitOptions, HandeyeOptimOptions as RigHandeyeOptimOptions,
    IntrinsicsInitOptions as RigHandeyeIntrinsicsInitOptions,
    IntrinsicsOptimOptions as RigHandeyeIntrinsicsOptimOptions, RigHandeyeConfig, RigHandeyeExport,
    RigHandeyeInput, RigHandeyeProblem, RigHandeyeState,
    RigOptimOptions as RigHandeyeRigOptimOptions,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from other crates
// ─────────────────────────────────────────────────────────────────────────────

// Re-export from calib-core for convenience
pub use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, CameraParams, CorrespondenceView,
    FxFyCxCySkew, Iso3, NoMeta, PinholeCamera, PlanarDataset, Pt2, Pt3, View,
};

// Re-export from calib-optim for convenience
pub use calib_optim::{BackendSolveOptions, HandEyeMode, RobustLoss};

// Hand-eye types re-exported from calib-optim
pub mod handeye {
    //! Hand-eye calibration types re-exported from calib-optim.
    pub use calib_optim::{
        optimize_handeye, HandEyeDataset, HandEyeEstimate, HandEyeParams, HandEyeSolveOptions,
        RigViewObs, RobotPoseMeta, View,
    };
}
