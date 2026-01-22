//! Intermediate state for multi-camera rig extrinsics calibration.
//!
//! This module defines `RigExtrinsicsState`, which holds intermediate results
//! computed during the calibration pipeline.

use calib_core::{Iso3, PinholeCamera};
use serde::{Deserialize, Serialize};

/// Intermediate state for rig extrinsics calibration.
///
/// Stores by-products of the calibration pipeline including:
/// - Per-camera intrinsics from individual calibration
/// - Per-camera target poses
/// - Initial rig extrinsics from linear estimation
/// - Optimization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigExtrinsicsState {
    // ─────────────────────────────────────────────────────────────────────────
    // Per-camera intrinsics
    // ─────────────────────────────────────────────────────────────────────────
    /// Per-camera calibrated intrinsics + distortion.
    pub per_cam_intrinsics: Option<Vec<PinholeCamera>>,

    /// Per-camera target poses: `[view][cam] -> Option<Iso3>`.
    /// `cam_se3_target` (T_C_T) for each camera in each view.
    pub per_cam_target_poses: Option<Vec<Vec<Option<Iso3>>>>,

    /// Per-camera mean reprojection error from intrinsics calibration.
    pub per_cam_reproj_errors: Option<Vec<f64>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig extrinsics initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial rig-to-camera transforms: `cam_se3_rig` (T_C_R) per camera.
    /// Reference camera has identity.
    pub initial_cam_se3_rig: Option<Vec<Iso3>>,

    /// Initial target-to-rig poses: `rig_se3_target` (T_R_T) per view.
    pub initial_rig_se3_target: Option<Vec<Iso3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig BA (metrics; result in output)
    // ─────────────────────────────────────────────────────────────────────────
    /// Final cost from rig BA.
    pub rig_ba_final_cost: Option<f64>,

    /// Mean reprojection error after rig BA.
    pub rig_ba_reproj_error: Option<f64>,
}

impl RigExtrinsicsState {
    /// Check if per-camera intrinsics have been computed.
    pub fn has_per_cam_intrinsics(&self) -> bool {
        self.per_cam_intrinsics.is_some()
    }

    /// Check if rig initialization has been run.
    pub fn has_rig_init(&self) -> bool {
        self.initial_cam_se3_rig.is_some() && self.initial_rig_se3_target.is_some()
    }

    /// Check if rig BA has been run.
    pub fn has_rig_optimized(&self) -> bool {
        self.rig_ba_final_cost.is_some()
    }

    /// Clear rig-related results, keeping per-camera intrinsics.
    pub fn clear_rig(&mut self) {
        self.initial_cam_se3_rig = None;
        self.initial_rig_se3_target = None;
        self.rig_ba_final_cost = None;
        self.rig_ba_reproj_error = None;
    }

    /// Clear everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{make_pinhole_camera, BrownConrady5, FxFyCxCySkew};

    fn make_test_camera() -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        )
    }

    #[test]
    fn default_state_not_initialized() {
        let state = RigExtrinsicsState::default();
        assert!(!state.has_per_cam_intrinsics());
        assert!(!state.has_rig_init());
        assert!(!state.has_rig_optimized());
    }

    #[test]
    fn has_rig_init_requires_both() {
        let state = RigExtrinsicsState {
            initial_cam_se3_rig: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        assert!(!state.has_rig_init()); // missing rig_se3_target

        let state = RigExtrinsicsState {
            initial_cam_se3_rig: Some(vec![Iso3::identity()]),
            initial_rig_se3_target: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        assert!(state.has_rig_init());
    }

    #[test]
    fn clear_rig_keeps_intrinsics() {
        let mut state = RigExtrinsicsState {
            per_cam_intrinsics: Some(vec![make_test_camera()]),
            initial_cam_se3_rig: Some(vec![Iso3::identity()]),
            initial_rig_se3_target: Some(vec![Iso3::identity()]),
            rig_ba_final_cost: Some(0.001),
            ..Default::default()
        };

        state.clear_rig();

        assert!(state.has_per_cam_intrinsics()); // Kept
        assert!(!state.has_rig_init()); // Cleared
        assert!(!state.has_rig_optimized()); // Cleared
    }

    #[test]
    fn json_roundtrip() {
        let state = RigExtrinsicsState {
            per_cam_intrinsics: Some(vec![make_test_camera(), make_test_camera()]),
            per_cam_reproj_errors: Some(vec![0.5, 0.6]),
            initial_cam_se3_rig: Some(vec![Iso3::identity(), Iso3::identity()]),
            initial_rig_se3_target: Some(vec![Iso3::identity()]),
            rig_ba_final_cost: Some(0.001),
            rig_ba_reproj_error: Some(0.3),
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: RigExtrinsicsState = serde_json::from_str(&json).unwrap();

        assert!(restored.has_per_cam_intrinsics());
        assert!(restored.has_rig_init());
        assert!(restored.has_rig_optimized());
        assert_eq!(restored.per_cam_intrinsics.unwrap().len(), 2);
    }
}
