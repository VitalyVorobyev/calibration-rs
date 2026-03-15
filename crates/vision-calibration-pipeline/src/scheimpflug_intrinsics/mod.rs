//! Planar intrinsics calibration with Scheimpflug sensor tilt.
//!
//! This module follows the standard pipeline shape used in this workspace:
//! `problem` + `state` + `steps`.

mod problem;
mod state;
mod steps;

pub use problem::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsConfig, ScheimpflugIntrinsicsExport,
    ScheimpflugIntrinsicsInput, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
    ScheimpflugIntrinsicsResult,
};
pub use state::ScheimpflugIntrinsicsState;
pub use steps::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, ScheimpflugManualInit, run_calibration,
    step_init, step_optimize, step_set_init,
};
