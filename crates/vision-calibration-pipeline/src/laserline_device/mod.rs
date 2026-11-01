//! Single laserline device calibration pipeline.
//!
//! Calibrates a single area camera + laser plane device using planar target
//! observations and laser line pixels (no hand-eye at this stage). The
//! camera is the configured model (`camera.model`): any distortion kind, an
//! identity or a Scheimpflug sensor.

mod problem;
mod state;
mod steps;

// Public API
pub use problem::{
    DeviceOptimizeConfig, LaserlineDeviceConfig, LaserlineDeviceEstimate, LaserlineDeviceExport,
    LaserlineDeviceInput, LaserlineDeviceProblem,
};
pub use steps::{
    DeviceInitOptions, DeviceInitResult, DeviceManualInit, DeviceOptimizeOptions,
    DeviceOptimizeResult, run_calibration, step_init, step_init_with_seed, step_optimize,
};
// The input's per-view types.
pub use vision_calibration_optim::{LaserlineMeta, LaserlineView};
