//! High-level camera calibration pipelines.
//!
//! This crate provides ready-to-use calibration workflows with multiple APIs:
//!
//! ## Session API
//!
//! The session API uses a mutable state container with step functions.
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblem, step_init, step_optimize, run_calibration,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let dataset = unimplemented!();
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
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
//! # Ok(())
//! # }
//! ```

// Core session framework
pub mod session;

// Problem-specific modules
pub mod laserline_device;
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
    FilterOptions, InitOptions, OptimizeOptions, PlanarConfig, PlanarExport,
    PlanarIntrinsicsProblem, PlanarState, run_calibration as run_planar_intrinsics,
    run_calibration_with_filtering, step_filter, step_init, step_optimize,
};

// Single-camera hand-eye
pub use crate::single_cam_handeye::{
    HandeyeInitOptions as SingleCamHandeyeInitOptions, HandeyeMeta,
    HandeyeOptimOptions as SingleCamHandeyeOptimOptions,
    IntrinsicsInitOptions as SingleCamIntrinsicsInitOptions,
    IntrinsicsOptimOptions as SingleCamIntrinsicsOptimOptions, SingleCamHandeyeConfig,
    SingleCamHandeyeExport, SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeState,
    SingleCamHandeyeView, run_calibration as run_single_cam_handeye,
    step_handeye_init as single_cam_step_handeye_init,
    step_handeye_optimize as single_cam_step_handeye_optimize,
    step_intrinsics_init as single_cam_step_intrinsics_init,
    step_intrinsics_optimize as single_cam_step_intrinsics_optimize,
};

// Rig extrinsics
pub use crate::rig_extrinsics::{
    IntrinsicsInitOptions as RigIntrinsicsInitOptions,
    IntrinsicsOptimOptions as RigIntrinsicsOptimOptions, RigExtrinsicsConfig, RigExtrinsicsExport,
    RigExtrinsicsInput, RigExtrinsicsProblem, RigExtrinsicsState, RigOptimOptions,
    run_calibration as run_rig_extrinsics,
    step_intrinsics_init_all as rig_step_intrinsics_init_all,
    step_intrinsics_optimize_all as rig_step_intrinsics_optimize_all, step_rig_init,
    step_rig_optimize,
};

// Rig hand-eye
pub use crate::rig_handeye::{
    HandeyeInitOptions as RigHandeyeInitOptions, HandeyeOptimOptions as RigHandeyeOptimOptions,
    IntrinsicsInitOptions as RigHandeyeIntrinsicsInitOptions,
    IntrinsicsOptimOptions as RigHandeyeIntrinsicsOptimOptions, RigHandeyeConfig, RigHandeyeExport,
    RigHandeyeInput, RigHandeyeProblem, RigHandeyeState,
    RigOptimOptions as RigHandeyeRigOptimOptions, run_calibration as run_rig_handeye,
    step_handeye_init as rig_handeye_step_handeye_init,
    step_handeye_optimize as rig_handeye_step_handeye_optimize,
    step_intrinsics_init_all as rig_handeye_step_intrinsics_init_all,
    step_intrinsics_optimize_all as rig_handeye_step_intrinsics_optimize_all,
    step_rig_init as rig_handeye_step_rig_init, step_rig_optimize as rig_handeye_step_rig_optimize,
};

// Laserline device
pub use crate::laserline_device::{
    InitOptions as LaserlineInitOptions, LaserlineDeviceConfig, LaserlineDeviceExport,
    LaserlineDeviceInput, LaserlineDeviceOutput, LaserlineDeviceProblem, LaserlineDeviceState,
    OptimizeOptions as LaserlineOptimizeOptions, run_calibration as run_laserline_device,
    step_init as laserline_step_init, step_optimize as laserline_step_optimize,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from other crates
// ─────────────────────────────────────────────────────────────────────────────

// Re-export from vision-calibration-core for convenience
pub use vision_calibration_core::{
    BrownConrady5, CameraParams, CorrespondenceView, FxFyCxCySkew, Iso3, NoMeta, PinholeCamera,
    PlanarDataset, Pt2, Pt3, View, make_pinhole_camera, pinhole_camera_params,
};

// Re-export from vision-calibration-optim for convenience
pub use vision_calibration_optim::{BackendSolveOptions, HandEyeMode, RobustLoss};

// Hand-eye types re-exported from vision-calibration-optim
pub mod handeye {
    //! Hand-eye calibration types re-exported from vision-calibration-optim.
    pub use vision_calibration_optim::{
        HandEyeDataset, HandEyeEstimate, HandEyeParams, HandEyeSolveOptions, RigViewObs,
        RobotPoseMeta, View, optimize_handeye,
    };
}
