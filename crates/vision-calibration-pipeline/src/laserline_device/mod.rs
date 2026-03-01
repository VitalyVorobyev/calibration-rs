//! Single laserline device calibration pipeline.
//!
//! Calibrates a single area camera + laser plane device using planar target
//! observations and laser line pixels (no hand-eye at this stage).

mod problem;
mod state;
mod steps;

// Public API
pub use problem::{
    LaserlineDeviceConfig, LaserlineDeviceExport, LaserlineDeviceInitConfig, LaserlineDeviceInput,
    LaserlineDeviceOptimizeConfig, LaserlineDeviceOutput, LaserlineDeviceProblem,
    LaserlineDeviceSolverConfig,
};
pub use state::LaserlineDeviceState;
pub use steps::{InitOptions, OptimizeOptions, run_calibration, step_init, step_optimize};
