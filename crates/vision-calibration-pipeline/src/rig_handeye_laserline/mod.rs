//! Joint rig hand-eye + laserline calibration.
//!
//! The pipeline runs rig hand-eye, initializes per-camera laser planes from the
//! frozen hand-eye geometry, then jointly refines the rig/hand-eye/laser
//! parameters.

mod problem;
mod state;
mod steps;

pub use problem::{
    RigHandeyeLaserlineBaConfig, RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport,
    RigHandeyeLaserlineInput, RigHandeyeLaserlineOutput, RigHandeyeLaserlineProblem,
};
pub use steps::run_calibration;
