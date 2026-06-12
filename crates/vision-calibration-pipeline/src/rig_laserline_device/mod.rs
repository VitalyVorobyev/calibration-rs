//! Rig-level laserline calibration.
//!
//! Given a rig calibration (intrinsics, optional Scheimpflug sensors,
//! cam_se3_rig, per-view rig_se3_target), calibrates one laser plane per
//! camera and reports each plane in the rig frame. Pinhole rigs are handled
//! as zero-tilt sensors (exactly the identity sensor mapping).

mod geometry;
mod problem;
mod state;
mod steps;

pub use geometry::pixel_to_gripper_point;
pub use problem::{
    RigLaserlineDeviceConfig, RigLaserlineDeviceExport, RigLaserlineDeviceInput,
    RigLaserlineDeviceProblem, RigUpstreamCalibration,
};
pub use steps::{
    RigLaserlineDeviceManualInit, StepOptions, run_calibration, step_init, step_init_with_seed,
    step_optimize,
};
