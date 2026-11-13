//! Problem builders for different calibration tasks.

pub mod handeye;
mod handeye_chain;
pub mod laserline_bundle;
pub mod laserline_rig_bundle;
pub mod planar_intrinsics;
pub mod rig_extrinsics;
pub mod rig_handeye_laserline_bundle;
mod robot_poses;
pub use robot_poses::{RobotPoseSigma, RobotRefinement};
pub mod scheimpflug_intrinsics;
