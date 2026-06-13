//! Joint rig hand-eye + laserline calibration.
//!
//! This pipeline matches the rtv3d V5 benchmark flow: run rig hand-eye,
//! initialize per-camera laser planes from the frozen hand-eye geometry, then
//! jointly refine the rig/hand-eye/laser parameters.

mod problem;
mod state;
mod steps;

pub use problem::{
    JointCameraFixMask, RigHandeyeLaserlineBaConfig, RigHandeyeLaserlineConfig,
    RigHandeyeLaserlineExport, RigHandeyeLaserlineInput, RigHandeyeLaserlineOutput,
    RigHandeyeLaserlineProblem,
};
pub use steps::run_calibration;
