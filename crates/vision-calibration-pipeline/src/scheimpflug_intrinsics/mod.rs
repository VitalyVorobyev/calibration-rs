//! Planar intrinsics calibration with Scheimpflug sensor tilt.
//!
//! This module follows the standard pipeline shape used in this workspace:
//! `problem` + `state` + `steps`.

mod problem;
mod state;
mod steps;

pub use problem::{
    ScheimpflugFixMask, ScheimpflugIntrinsicsCalibrationConfig, ScheimpflugIntrinsicsExport,
    ScheimpflugIntrinsicsInput, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsProblem,
    ScheimpflugIntrinsicsResult,
};
pub use state::ScheimpflugIntrinsicsState;
pub use steps::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, run_calibration, step_init, step_optimize,
};
