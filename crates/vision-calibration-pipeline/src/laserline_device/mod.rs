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
pub use steps::{
    DeviceInitOptions, DeviceOptimizeOptions, LaserlineDeviceInitResult, LaserlineDeviceManualInit,
    LaserlineDeviceOptimizeResult, run_calibration, step_init, step_init_with_seed, step_optimize,
};
