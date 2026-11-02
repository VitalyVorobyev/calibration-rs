//! Joint rig hand-eye + laserline calibration.
//!
//! The pipeline runs rig hand-eye, initializes per-camera laser planes from the
//! frozen hand-eye geometry, then jointly refines the rig/hand-eye/laser
//! parameters. Each camera keeps its own model (`handeye.camera` and
//! `handeye.camera_overrides`); the joint stage names its fixed parameters
//! per camera in `joint_optimize.fix`.
//!
//! [`run_calibration_with_seed`] starts from an earlier rig hand-eye result
//! and/or given laser planes ([`JointManualInit`]) instead of estimating them.

mod problem;
mod state;
mod steps;

pub use problem::{
    JointOptimizeConfig, RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport,
    RigHandeyeLaserlineInput, RigHandeyeLaserlineProblem,
};
pub use steps::{JointManualInit, run_calibration, run_calibration_with_seed};
// The input's view type and the joint solve's estimate (the problem's
// output); the shared parts of a view are in `common`.
pub use vision_calibration_optim::{RigHandeyeLaserlineEstimate, RigHandeyeLaserlineView};
