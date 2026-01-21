//! Intermediate state for single-camera hand-eye calibration.
//!
//! This module defines `SingleCamHandeyeState`, which holds intermediate results
//! computed during the calibration pipeline.

use calib_core::{BrownConrady5, FxFyCxCySkew, Iso3, PinholeCamera, Real};
use serde::{Deserialize, Serialize};

/// Intermediate state for single-camera hand-eye calibration.
///
/// Stores by-products of the calibration pipeline including:
/// - Initial intrinsics from linear estimation
/// - Optimized camera parameters
/// - Hand-eye initialization
/// - Optimization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SingleCamHandeyeState {
    // ─────────────────────────────────────────────────────────────────────────
    // From intrinsics initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial intrinsics estimate (fx, fy, cx, cy, skew).
    pub initial_intrinsics: Option<FxFyCxCySkew<Real>>,

    /// Initial distortion estimate.
    pub initial_distortion: Option<BrownConrady5<Real>>,

    /// Initial target poses (cam_se3_target) per view from PnP.
    pub initial_target_poses: Option<Vec<Iso3>>,

    // ─────────────────────────────────────────────────────────────────────────
    // From intrinsics optimization
    // ─────────────────────────────────────────────────────────────────────────
    /// Optimized camera (intrinsics + distortion).
    pub optimized_camera: Option<PinholeCamera>,

    /// Optimized target poses (cam_se3_target) per view.
    pub optimized_target_poses: Option<Vec<Iso3>>,

    /// Mean reprojection error after intrinsics optimization.
    pub intrinsics_reproj_error: Option<f64>,

    // ─────────────────────────────────────────────────────────────────────────
    // From hand-eye initialization
    // ─────────────────────────────────────────────────────────────────────────
    /// Initial hand-eye transform from linear estimation.
    /// For EyeInHand: gripper_se3_camera (T_G_C).
    /// For EyeToHand: camera_se3_base (T_C_B).
    pub initial_handeye: Option<Iso3>,

    /// Initial target pose in base frame (for EyeInHand: T_B_T).
    pub initial_target_se3_base: Option<Iso3>,

    // ─────────────────────────────────────────────────────────────────────────
    // From hand-eye optimization (metrics only; result in output)
    // ─────────────────────────────────────────────────────────────────────────
    /// Final cost from hand-eye BA.
    pub handeye_final_cost: Option<f64>,

    /// Mean reprojection error after hand-eye BA.
    pub handeye_reproj_error: Option<f64>,
}

impl SingleCamHandeyeState {
    /// Check if intrinsics initialization has been run.
    pub fn has_intrinsics_init(&self) -> bool {
        self.initial_intrinsics.is_some() && self.initial_target_poses.is_some()
    }

    /// Check if intrinsics optimization has been run.
    pub fn has_intrinsics_optimized(&self) -> bool {
        self.optimized_camera.is_some() && self.optimized_target_poses.is_some()
    }

    /// Check if hand-eye initialization has been run.
    pub fn has_handeye_init(&self) -> bool {
        self.initial_handeye.is_some()
    }

    /// Check if hand-eye optimization has been run.
    pub fn has_handeye_optimized(&self) -> bool {
        self.handeye_final_cost.is_some()
    }

    /// Clear hand-eye results, keeping intrinsics.
    pub fn clear_handeye(&mut self) {
        self.initial_handeye = None;
        self.initial_target_se3_base = None;
        self.handeye_final_cost = None;
        self.handeye_reproj_error = None;
    }

    /// Clear everything.
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_not_initialized() {
        let state = SingleCamHandeyeState::default();
        assert!(!state.has_intrinsics_init());
        assert!(!state.has_intrinsics_optimized());
        assert!(!state.has_handeye_init());
        assert!(!state.has_handeye_optimized());
    }

    #[test]
    fn has_intrinsics_init_requires_both() {
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };

        let state = SingleCamHandeyeState {
            initial_intrinsics: Some(intrinsics),
            ..Default::default()
        };
        assert!(!state.has_intrinsics_init()); // missing poses

        let state = SingleCamHandeyeState {
            initial_intrinsics: Some(intrinsics),
            initial_target_poses: Some(vec![Iso3::identity()]),
            ..Default::default()
        };
        assert!(state.has_intrinsics_init());
    }

    #[test]
    fn clear_handeye_keeps_intrinsics() {
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };

        let mut state = SingleCamHandeyeState {
            initial_intrinsics: Some(intrinsics),
            initial_target_poses: Some(vec![Iso3::identity()]),
            initial_handeye: Some(Iso3::identity()),
            handeye_final_cost: Some(0.001),
            ..Default::default()
        };

        state.clear_handeye();

        assert!(state.has_intrinsics_init()); // Kept
        assert!(!state.has_handeye_init()); // Cleared
        assert!(!state.has_handeye_optimized()); // Cleared
    }

    #[test]
    fn json_roundtrip() {
        let state = SingleCamHandeyeState {
            initial_intrinsics: Some(FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            }),
            initial_distortion: Some(BrownConrady5 {
                k1: -0.1,
                k2: 0.05,
                k3: 0.0,
                p1: 0.001,
                p2: -0.001,
                iters: 8,
            }),
            initial_target_poses: Some(vec![Iso3::identity()]),
            handeye_final_cost: Some(0.001),
            handeye_reproj_error: Some(0.5),
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&state).unwrap();
        let restored: SingleCamHandeyeState = serde_json::from_str(&json).unwrap();

        assert!(restored.has_intrinsics_init());
        assert!(restored.has_handeye_optimized());
        assert_eq!(restored.initial_intrinsics.unwrap().fx, 800.0);
    }
}
