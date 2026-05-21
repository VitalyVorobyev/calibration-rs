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
#[allow(deprecated)]
pub use steps::step_set_init;
pub use steps::{
    IntrinsicsInitOptions, IntrinsicsOptimizeOptions, ScheimpflugIntrinsicsInitResult,
    ScheimpflugIntrinsicsOptimizeResult, ScheimpflugManualInit, run_calibration, step_init,
    step_init_with_seed, step_optimize,
};
