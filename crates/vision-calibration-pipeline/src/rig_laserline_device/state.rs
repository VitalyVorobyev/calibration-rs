//! Intermediate state for rig laserline calibration.

use serde::{Deserialize, Serialize};
use vision_calibration_optim::LaserPlane;

/// Intermediate state for rig laserline calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigLaserlineDeviceState {
    /// Per-camera initial plane estimates in camera frame.
    pub initial_planes_cam: Option<Vec<LaserPlane>>,
}

impl RigLaserlineDeviceState {
    /// Whether initial planes have been set.
    pub fn has_initial_planes(&self) -> bool {
        self.initial_planes_cam.is_some()
    }
}
