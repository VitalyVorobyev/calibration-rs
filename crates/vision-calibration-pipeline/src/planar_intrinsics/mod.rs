//! Planar intrinsics calibration (Zhang's method with distortion).
//!
//! This module provides multiple APIs for planar intrinsics calibration:
//!
//! ## Session API
//!
//! The session API uses a mutable state container with step functions.
//!
//! ```ignore
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::planar_intrinsics::{
//!     PlanarIntrinsicsProblemV2, PlanarConfig,
//!     step_init, step_optimize, run_calibration,
//! };
//!
//! let mut session = CalibrationSession::<PlanarIntrinsicsProblemV2>::new();
//! session.set_input(dataset)?;
//!
//! // Option 1: Step-by-step
//! step_init(&mut session, None)?;
//! step_optimize(&mut session, None)?;
//!
//! // Option 2: Pipeline function
//! // run_calibration(&mut session)?;
//!
//! let export = session.export()?;
//! ```

// Session API
mod problem;
mod state;
mod steps;

// ─────────────────────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────────────────────
pub use problem::{PlanarConfig, PlanarExport, PlanarIntrinsicsProblem};
pub use state::PlanarState;
pub use steps::{
    FilterOptions, InitOptions, OptimizeOptions, run_calibration, run_calibration_with_filtering,
    step_filter, step_init, step_optimize,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from vision-calibration-optim
// ─────────────────────────────────────────────────────────────────────────────

pub use vision_calibration_optim::{
    PlanarIntrinsicsEstimate, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
};
