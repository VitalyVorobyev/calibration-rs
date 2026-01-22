//! Intermediate state for multi-camera rig hand-eye calibration.
//!
//! This module defines `RigHandeyeState`, which holds intermediate results
//! computed during the calibration pipeline.

use calib_core::{Iso3, PinholeCamera};
use serde::{Deserialize, Serialize};

/// Intermediate state for rig hand-eye calibration.
///
/// Stores by-products of the calibration pipeline including:
/// - Per-camera intrinsics from individual calibration
/// - Per-camera target poses
/// - Rig extrinsics from linear and BA estimation
/// - Hand-eye initialization
/// - Final optimization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigHandeyeState {
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
    /// Initial camera-to-rig transforms: `cam_se3_rig` (T_C_R) per camera.
    /// Reference camera has identity.
    pub initial_cam_se3_rig: Option<Vec<Iso3>>,

    /// Initial rig-to-target poses: `rig_se3_target` (T_R_T) per view.
    pub initial_rig_se3_target: Option<Vec<Iso3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig BA results
    // ─────────────────────────────────────────────────────────────────────────
    /// Refined camera-to-rig transforms from rig BA.
    pub rig_ba_cam_se3_rig: Option<Vec<Iso3>>,

    /// Refined rig-to-target poses from rig BA.
    pub rig_ba_rig_se3_target: Option<Vec<Iso3>>,

    /// Mean reprojection error after rig BA.
    pub rig_ba_reproj_error: Option<f64>,

    /// Mean reprojection error after rig BA, per camera.
    pub rig_ba_per_cam_reproj_errors: Option<Vec<f64>>,

    // ─────────────────────────────────────────────────────────────────────────
    // Hand-eye initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial hand-eye transform from linear estimation.
    /// `gripper_se3_rig` (T_G_R) for EyeInHand mode.
    pub initial_handeye: Option<Iso3>,

    /// Initial target pose in base frame: `target_se3_base` (T_T_B).
    /// Single static target.
    pub initial_target_se3_base: Option<Iso3>,

    // ─────────────────────────────────────────────────────────────────────────
    // Final BA metrics (result in output)
    // ─────────────────────────────────────────────────────────────────────────
    /// Final cost from hand-eye BA.
    pub final_cost: Option<f64>,

    /// Mean reprojection error after final BA.
    pub final_reproj_error: Option<f64>,
}

impl RigHandeyeState {
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
        self.rig_ba_cam_se3_rig.is_some()
    }

    /// Check if hand-eye initialization has been run.
    pub fn has_handeye_init(&self) -> bool {
        self.initial_handeye.is_some() && self.initial_target_se3_base.is_some()
    }

    /// Check if final BA has been run.
    pub fn has_final_optimized(&self) -> bool {
        self.final_cost.is_some()
    }

    /// Clear rig-related results, keeping per-camera intrinsics.
    pub fn clear_rig(&mut self) {
        self.initial_cam_se3_rig = None;
        self.initial_rig_se3_target = None;
        self.rig_ba_cam_se3_rig = None;
        self.rig_ba_rig_se3_target = None;
        self.rig_ba_reproj_error = None;
        self.rig_ba_per_cam_reproj_errors = None;
    }

    /// Clear hand-eye and final BA results.
    pub fn clear_handeye(&mut self) {
        self.initial_handeye = None;
        self.initial_target_se3_base = None;
        self.final_cost = None;
        self.final_reproj_error = None;
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
        let state = RigHandeyeState::default();
        assert!(!state.has_per_cam_intrinsics());
        assert!(!state.has_rig_init());
        assert!(!state.has_rig_optimized());
        assert!(!state.has_handeye_init());
        assert!(!state.has_final_optimized());
    }

    #[test]
    fn has_rig_init_requires_both() {
        let state = RigHandeyeState {
            initial_cam_se3_rig: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        assert!(!state.has_rig_init()); // missing rig_se3_target

        let state = RigHandeyeState {
            initial_cam_se3_rig: Some(vec![Iso3::identity()]),
            initial_rig_se3_target: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        assert!(state.has_rig_init());
    }

    #[test]
    fn has_handeye_init_requires_both() {
        let state = RigHandeyeState {
            initial_handeye: Some(Iso3::identity()),
            ..Default::default()
        };
        assert!(!state.has_handeye_init()); // missing target_se3_base

        let state = RigHandeyeState {
            initial_handeye: Some(Iso3::identity()),
            initial_target_se3_base: Some(Iso3::identity()),
            ..Default::default()
        };
        assert!(state.has_handeye_init());
    }

    #[test]
    fn clear_rig_keeps_intrinsics() {
        let mut state = RigHandeyeState {
            per_cam_intrinsics: Some(vec![make_test_camera()]),
            initial_cam_se3_rig: Some(vec![Iso3::identity()]),
            initial_rig_se3_target: Some(vec![Iso3::identity()]),
            rig_ba_cam_se3_rig: Some(vec![Iso3::identity()]),
            rig_ba_reproj_error: Some(0.5),
            ..Default::default()
        };

        state.clear_rig();

        assert!(state.has_per_cam_intrinsics()); // Kept
        assert!(!state.has_rig_init()); // Cleared
        assert!(!state.has_rig_optimized()); // Cleared
    }

    #[test]
    fn json_roundtrip() {
        let state = RigHandeyeState {
            per_cam_intrinsics: Some(vec![make_test_camera(), make_test_camera()]),
            per_cam_reproj_errors: Some(vec![0.5, 0.6]),
            initial_cam_se3_rig: Some(vec![Iso3::identity(), Iso3::identity()]),
            initial_rig_se3_target: Some(vec![Iso3::identity()]),
            rig_ba_cam_se3_rig: Some(vec![Iso3::identity(), Iso3::identity()]),
            rig_ba_reproj_error: Some(0.3),
            initial_handeye: Some(Iso3::identity()),
            initial_target_se3_base: Some(Iso3::identity()),
            final_cost: Some(0.001),
            final_reproj_error: Some(0.3),
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: RigHandeyeState = serde_json::from_str(&json).unwrap();

        assert!(restored.has_per_cam_intrinsics());
        assert!(restored.has_rig_init());
        assert!(restored.has_rig_optimized());
        assert!(restored.has_handeye_init());
        assert!(restored.has_final_optimized());
        assert_eq!(restored.per_cam_intrinsics.unwrap().len(), 2);
    }
}
