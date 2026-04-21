//! Intermediate state for Scheimpflug rig hand-eye calibration.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{Iso3, PinholeCamera, ScheimpflugParams};

/// Intermediate state for Scheimpflug rig hand-eye calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigScheimpflugHandeyeState {
    /// Per-camera pinhole core (intrinsics + distortion).
    pub per_cam_intrinsics: Option<Vec<PinholeCamera>>,
    /// Per-camera Scheimpflug sensor.
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
    /// Per-view per-camera camera_se3_target.
    pub per_cam_target_poses: Option<Vec<Vec<Option<Iso3>>>>,
    /// Per-camera mean reprojection error from intrinsics calibration.
    pub per_cam_reproj_errors: Option<Vec<f64>>,
    /// Per-camera flag: `true` means the camera's initial intrinsics came from
    /// a fallback (e.g., Zhang failed and the module reused another camera's
    /// solution). Fallback cameras skip per-camera Scheimpflug BA.
    pub per_cam_used_fallback: Option<Vec<bool>>,

    /// Initial cam_se3_rig.
    pub initial_cam_se3_rig: Option<Vec<Iso3>>,
    /// Initial rig_se3_target.
    pub initial_rig_se3_target: Option<Vec<Iso3>>,

    /// Refined cam_se3_rig from rig BA.
    pub rig_ba_cam_se3_rig: Option<Vec<Iso3>>,
    /// Refined rig_se3_target from rig BA.
    pub rig_ba_rig_se3_target: Option<Vec<Iso3>>,
    /// Mean reprojection error after rig BA.
    pub rig_ba_reproj_error: Option<f64>,
    /// Per-camera reprojection error after rig BA.
    pub rig_ba_per_cam_reproj_errors: Option<Vec<f64>>,

    /// Initial hand-eye transform (gripper_se3_rig for EyeInHand).
    pub initial_handeye: Option<Iso3>,
    /// Initial base_se3_target (EyeInHand).
    pub initial_base_se3_target: Option<Iso3>,

    /// Final cost from hand-eye BA.
    pub final_cost: Option<f64>,
    /// Final reprojection error after hand-eye BA.
    pub final_reproj_error: Option<f64>,
}

impl RigScheimpflugHandeyeState {
    /// Whether intrinsics have been initialized.
    pub fn has_per_cam_intrinsics(&self) -> bool {
        self.per_cam_intrinsics.is_some() && self.per_cam_sensors.is_some()
    }
    /// Whether rig init has run.
    pub fn has_rig_init(&self) -> bool {
        self.initial_cam_se3_rig.is_some() && self.initial_rig_se3_target.is_some()
    }
    /// Whether rig BA has run.
    pub fn has_rig_optimized(&self) -> bool {
        self.rig_ba_cam_se3_rig.is_some()
    }
    /// Whether hand-eye init has run.
    pub fn has_handeye_init(&self) -> bool {
        self.initial_handeye.is_some() && self.initial_base_se3_target.is_some()
    }
}
