//! Joint rig hand-eye + laserline calibration.
//!
//! The pipeline runs rig hand-eye, initializes per-camera laser planes from the
//! frozen hand-eye geometry, then jointly refines the rig/hand-eye/laser
//! parameters. Each camera keeps its own model (`handeye.camera` and
//! `handeye.camera_overrides`); the joint stage names its fixed parameters
//! per camera in `joint_optimize.fix`.

mod problem;
mod state;
mod steps;

pub use problem::{
    JointOptimizeConfig, RigHandeyeLaserlineConfig, RigHandeyeLaserlineExport,
    RigHandeyeLaserlineInput, RigHandeyeLaserlineProblem,
};
pub use steps::run_calibration;
// The input's view types, so the workflow is usable from this module alone,
// and the joint solve's estimate (the problem's output).
pub use vision_calibration_optim::{
    RigHandeyeLaserlineEstimate, RigHandeyeLaserlineView, RigLaserlineView, RobotPoseMeta,
};
