//! Planar intrinsics calibration (Zhang's method with distortion).
//!
//! A [`CalibrationSession`](crate::session::CalibrationSession) over
//! [`PlanarIntrinsicsProblem`] runs the steps `step_init` → `step_optimize`
//! (optionally `step_filter` and a second solve), or all of them through
//! `run_calibration`; [`PlanarIntrinsicsProblem`] has a runnable example.

// Session API
mod problem;
mod state;
mod steps;

// ─────────────────────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────────────────────
pub(crate) use problem::MIN_VIEW_POINTS;
pub use problem::{
    PlanarIntrinsicsConfig, PlanarIntrinsicsExport, PlanarIntrinsicsInput, PlanarIntrinsicsProblem,
    PlanarOptimizeConfig,
};
pub use steps::{
    IntrinsicsFilterOptions, IntrinsicsInitResult, IntrinsicsOptimizeResult, run_calibration,
    run_calibration_with_filtering, step_filter, step_init, step_init_with_seed, step_optimize,
};

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports from vision-calibration-optim
// ─────────────────────────────────────────────────────────────────────────────

pub use vision_calibration_optim::{PlanarIntrinsicsEstimate, PlanarIntrinsicsParams};
