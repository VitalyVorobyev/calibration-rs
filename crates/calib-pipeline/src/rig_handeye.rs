//! High-level multi-camera rig + robot hand-eye pipeline.

use crate::session::{problem_types::RigHandEyeProblem, CalibrationSession};
use anyhow::Result;
use serde::{Deserialize, Serialize};

pub use crate::session::problem_types::{
    CameraViewData as RigHandEyeCameraViewData, RigHandEyeInitOptions,
    RigHandEyeObservations as RigHandEyeInput, RigHandEyeOptimOptions,
    RigHandEyeOptimized as RigHandEyeReport, RigHandEyeViewData,
};

/// End-to-end rig + hand-eye configuration (init + joint BA).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigHandEyeConfig {
    #[serde(default)]
    pub init: RigHandEyeInitOptions,
    #[serde(default)]
    pub optim: RigHandEyeOptimOptions,
}

/// Run the full rig + hand-eye pipeline (seed + optimize) and return a report.
pub fn run_rig_handeye(
    input: &RigHandEyeInput,
    config: &RigHandEyeConfig,
) -> Result<RigHandEyeReport> {
    let mut session = CalibrationSession::<RigHandEyeProblem>::new();
    session.set_observations(input.clone());
    session.initialize(config.init.clone())?;
    session.optimize(config.optim.clone())?;
    session.export()
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{
        BrownConrady5, FxFyCxCySkew, IntrinsicsParams, Iso3, ProjectionParams, Pt3, SensorParams,
        Vec2,
    };
    use calib_optim::ir::{HandEyeMode, RobustLoss};

    #[test]
    fn rig_handeye_config_json_roundtrip() {
        let mut config = RigHandEyeConfig::default();
        config.optim.solve_opts.robust_loss = RobustLoss::Cauchy { scale: 1.5 };
        config.optim.backend_opts.max_iters = 42;

        let json = serde_json::to_string_pretty(&config).unwrap();
        let de: RigHandEyeConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.optim.backend_opts.max_iters, 42);
        match de.optim.solve_opts.robust_loss {
            RobustLoss::Cauchy { scale } => assert!((scale - 1.5).abs() < 1e-12),
            other => panic!("unexpected robust_loss: {other:?}"),
        }
    }

    #[test]
    fn rig_handeye_input_json_roundtrip() {
        let input = RigHandEyeInput {
            views: vec![RigHandEyeViewData {
                cameras: vec![
                    Some(RigHandEyeCameraViewData {
                        points_3d: vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.05, 0.0, 0.0)],
                        points_2d: vec![Vec2::new(100.0, 100.0), Vec2::new(200.0, 100.0)],
                        weights: None,
                    }),
                    None,
                ],
                base_from_gripper: Iso3::identity(),
            }],
            num_cameras: 2,
            mode: HandEyeMode::EyeInHand,
        };

        let json = serde_json::to_string_pretty(&input).unwrap();
        let de: RigHandEyeInput = serde_json::from_str(&json).unwrap();
        assert_eq!(de.num_cameras, 2);
        assert_eq!(de.views.len(), 1);
        assert_eq!(de.views[0].cameras.len(), 2);
        assert!(de.views[0].cameras[0].is_some());
        assert!(de.views[0].cameras[1].is_none());
    }

    #[test]
    fn rig_handeye_report_json_roundtrip() {
        let cam = calib_core::CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: calib_core::DistortionParams::BrownConrady5 {
                params: BrownConrady5 {
                    k1: 0.0,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            },
            sensor: SensorParams::Identity,
            intrinsics: IntrinsicsParams::FxFyCxCySkew {
                params: FxFyCxCySkew {
                    fx: 800.0,
                    fy: 780.0,
                    cx: 640.0,
                    cy: 360.0,
                    skew: 0.0,
                },
            },
        };

        let report = RigHandEyeReport {
            mode: HandEyeMode::EyeInHand,
            ref_cam_idx: 0,
            cameras: vec![cam],
            cam_to_rig: vec![Iso3::identity()],
            handeye: Iso3::identity(),
            target_poses: vec![Iso3::identity()],
            robot_deltas: Some(vec![[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            final_cost: 1.23,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let de: RigHandEyeReport = serde_json::from_str(&json).unwrap();
        assert_eq!(de.ref_cam_idx, 0);
        assert_eq!(de.cameras.len(), 1);
        assert_eq!(de.cam_to_rig.len(), 1);
        assert_eq!(de.target_poses.len(), 1);
        assert_eq!(de.robot_deltas.unwrap().len(), 1);
        assert!((de.final_cost - 1.23).abs() < 1e-12);
    }
}
