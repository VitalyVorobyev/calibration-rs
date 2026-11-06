//! Intermediate state for multi-camera rig hand-eye calibration.
//!
//! This module defines `RigHandeyeState`, which holds intermediate results
//! computed during the calibration pipeline.

use serde::{Deserialize, Serialize};
use vision_calibration_core::Iso3;

use crate::common::HandeyeSolution;
use crate::rig_family::RigStagesState;

/// Intermediate state for rig hand-eye calibration.
///
/// Stores by-products of the calibration pipeline including:
/// - The rig stages' state: per-camera parameters and target poses, the
///   initial rig and the rig BA's errors
/// - Rig extrinsics from the rig BA
/// - Hand-eye initialization
/// - Final optimization metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RigHandeyeState {
    /// The rig stages' state (serialized inline).
    #[serde(flatten)]
    pub rig: RigStagesState,

    /// Refined camera-to-rig transforms from rig BA.
    pub rig_ba_cam_se3_rig: Option<Vec<Iso3>>,

    /// Refined rig-to-target poses from rig BA.
    pub rig_ba_rig_se3_target: Option<Vec<Iso3>>,

    /// Initial hand-eye and target pose of the configured mode.
    pub initial_handeye: Option<HandeyeSolution>,

    /// Final cost from hand-eye BA.
    pub final_cost: Option<f64>,

    /// Mean reprojection error after final BA.
    pub final_reproj_error: Option<f64>,
}

impl RigHandeyeState {
    /// Check if rig BA has been run.
    pub fn has_rig_optimized(&self) -> bool {
        self.rig_ba_cam_se3_rig.is_some()
    }

    /// Check if hand-eye initialization has been run.
    pub fn has_handeye_init(&self) -> bool {
        self.initial_handeye.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_handeye_init_follows_the_solution() {
        assert!(!RigHandeyeState::default().has_handeye_init());
        let state = RigHandeyeState {
            initial_handeye: Some(HandeyeSolution::EyeInHand {
                gripper_se3_rig: Iso3::identity(),
                base_se3_target: Iso3::identity(),
            }),
            ..Default::default()
        };
        assert!(state.has_handeye_init());
    }

    #[test]
    fn json_keeps_the_stage_fields_inline() {
        let mut state = RigHandeyeState {
            rig_ba_cam_se3_rig: Some(vec![Iso3::identity()]),
            final_cost: Some(0.002),
            ..Default::default()
        };
        state.rig.per_cam_reproj_errors = Some(vec![0.5, 0.6]);
        state.rig.per_cam_intrinsics_auto = true;

        let json = serde_json::to_value(&state).unwrap();
        assert_eq!(json["per_cam_reproj_errors"][1], 0.6);
        assert_eq!(json["per_cam_intrinsics_auto"], true);
        assert!(json.get("rig").is_none());

        let restored: RigHandeyeState = serde_json::from_value(json).unwrap();
        assert!(restored.has_rig_optimized());
        assert!(restored.rig.per_cam_intrinsics_auto);
        assert_eq!(restored.final_cost, Some(0.002));
    }
}
