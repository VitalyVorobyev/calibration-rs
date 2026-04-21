//! Intermediate state for Scheimpflug rig extrinsics calibration.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{Iso3, PinholeCamera, ScheimpflugParams};

/// Intermediate state for Scheimpflug rig extrinsics calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigScheimpflugExtrinsicsState {
    /// Per-camera intrinsics + distortion (pinhole core, Scheimpflug sensor held separately).
    pub per_cam_intrinsics: Option<Vec<PinholeCamera>>,
    /// Per-camera Scheimpflug sensor tilt parameters.
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
    /// Per-camera target poses: `[view][cam] -> Option<Iso3>`, `cam_se3_target`.
    pub per_cam_target_poses: Option<Vec<Vec<Option<Iso3>>>>,
    /// Per-camera mean reprojection error from intrinsics calibration.
    pub per_cam_reproj_errors: Option<Vec<f64>>,

    /// Initial `cam_se3_rig` per camera.
    pub initial_cam_se3_rig: Option<Vec<Iso3>>,
    /// Initial `rig_se3_target` per view.
    pub initial_rig_se3_target: Option<Vec<Iso3>>,

    /// Final cost from rig BA.
    pub rig_ba_final_cost: Option<f64>,
    /// Mean reprojection error after rig BA.
    pub rig_ba_reproj_error: Option<f64>,
    /// Per-camera mean reprojection error after rig BA.
    pub rig_ba_per_cam_reproj_errors: Option<Vec<f64>>,
}

impl RigScheimpflugExtrinsicsState {
    /// Check if per-camera intrinsics have been computed.
    pub fn has_per_cam_intrinsics(&self) -> bool {
        self.per_cam_intrinsics.is_some() && self.per_cam_sensors.is_some()
    }

    /// Check if rig initialization has run.
    pub fn has_rig_init(&self) -> bool {
        self.initial_cam_se3_rig.is_some() && self.initial_rig_se3_target.is_some()
    }

    /// Check if rig BA has run.
    pub fn has_rig_optimized(&self) -> bool {
        self.rig_ba_final_cost.is_some()
    }
}
