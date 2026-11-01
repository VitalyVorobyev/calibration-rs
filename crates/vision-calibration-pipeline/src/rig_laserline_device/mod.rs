//! Rig-level laserline calibration.
//!
//! Given a rig calibration (each camera's `CameraParams`, cam_se3_rig,
//! per-view rig_se3_target), calibrates one laser plane per camera and
//! reports each plane in the rig frame. Each camera's laser residuals are
//! evaluated in its own model: any distortion kind, an identity or a
//! Scheimpflug sensor.

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
    PlaneInitOptions, PlaneManualInit, PlaneOptimizeOptions, run_calibration, step_init,
    step_init_with_seed, step_optimize,
};
// The input's dataset type and the plane solve's estimate (the problem's
// output); the per-view type is shared, in `common`.
pub use vision_calibration_optim::{RigLaserlineDataset, RigLaserlineEstimate};
