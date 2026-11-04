//! Intermediate state for multi-camera rig extrinsics calibration.
//!
//! This module defines `RigExtrinsicsState`, which holds intermediate results
//! computed during the calibration pipeline.

use serde::{Deserialize, Serialize};

use crate::rig_family::RigStagesState;

/// Intermediate state for rig extrinsics calibration: the rig stages'
/// by-products (per-camera parameters and target poses, the initial rig) and
/// the rig BA's metrics; the result is the output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct RigExtrinsicsState {
    /// The rig stages' state (serialized inline).
    #[serde(flatten)]
    pub rig: RigStagesState,

    /// Final cost from rig BA.
    pub rig_ba_final_cost: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::Iso3;

    #[test]
    fn json_keeps_the_stage_fields_inline() {
        let mut state = RigExtrinsicsState {
            rig_ba_final_cost: Some(0.001),
            ..Default::default()
        };
        state.rig.initial_cam_se3_rig = Some(vec![Iso3::identity(), Iso3::identity()]);
        state.rig.initial_rig_se3_target = Some(vec![Iso3::identity()]);
        state.rig.rig_ba_reproj_error = Some(0.3);

        let json = serde_json::to_value(&state).unwrap();
        assert_eq!(json["rig_ba_reproj_error"], 0.3);
        assert_eq!(json["rig_ba_final_cost"], 0.001);
        assert!(json.get("rig").is_none());

        let restored: RigExtrinsicsState = serde_json::from_value(json).unwrap();
        assert!(restored.rig.has_rig_init());
        assert!(!restored.rig.has_per_cam_intrinsics());
        assert_eq!(restored.rig_ba_final_cost, Some(0.001));
    }
}
