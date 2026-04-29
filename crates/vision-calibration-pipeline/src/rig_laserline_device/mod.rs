//! Rig-level laserline calibration.
//!
//! Given a rig calibration (intrinsics, Scheimpflug sensors, cam_se3_rig,
//! per-view rig_se3_target), calibrates one laser plane per camera and
//! reports each plane in the rig frame.

mod problem;
mod state;
mod steps;

pub use problem::{
    RigLaserlineDeviceConfig, RigLaserlineDeviceExport, RigLaserlineDeviceInput,
    RigLaserlineDeviceProblem, RigUpstreamCalibration,
};
pub use state::RigLaserlineDeviceState;
pub use steps::{
    RigLaserlineDeviceManualInit, StepOptions, run_calibration, step_init, step_optimize,
    step_set_init,
};
