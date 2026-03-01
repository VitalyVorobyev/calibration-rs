//! [`ProblemType`] implementation for multi-camera rig hand-eye calibration.
//!
//! This module provides the `RigHandeyeProblem` type that implements
//! the v2 session API's `ProblemType` trait.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{Iso3, PinholeCamera};
use vision_calibration_optim::{
    HandEyeEstimate, HandEyeMode, RigDataset, RobotPoseMeta, RobustLoss,
};
#[cfg(test)]
use vision_calibration_optim::{HandEyeParams, SolveReport};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigHandeyeState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Type
// ─────────────────────────────────────────────────────────────────────────────

/// Input for rig hand-eye calibration.
///
/// Reuses `RigDataset<RobotPoseMeta>` from vision_calibration_optim.
/// Each view contains robot pose metadata and per-camera observations.
pub type RigHandeyeInput = RigDataset<RobotPoseMeta>;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-camera rig hand-eye calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigHandeyeConfig {
    /// Per-camera intrinsics initialization options.
    pub intrinsics: RigHandeyeIntrinsicsConfig,
    /// Rig and gauge options.
    pub rig: RigHandeyeRigConfig,
    /// Hand-eye linear initialization options.
    pub handeye_init: RigHandeyeInitConfig,
    /// Shared solver settings for optimization stages.
    pub solver: RigHandeyeSolverConfig,
    /// Final hand-eye bundle-adjustment options.
    pub handeye_ba: RigHandeyeBaConfig,
}

/// Per-camera intrinsics initialization options for rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeIntrinsicsConfig {
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,
    /// Fix k3 during intrinsics calibration.
    pub fix_k3: bool,
    /// Fix tangential distortion (p1, p2).
    pub fix_tangential: bool,
    /// Enforce zero skew.
    pub zero_skew: bool,
}

impl Default for RigHandeyeIntrinsicsConfig {
    fn default() -> Self {
        Self {
            init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
        }
    }
}

/// Rig-specific options for rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeRigConfig {
    /// Reference camera index for rig frame (identity extrinsics).
    pub reference_camera_idx: usize,
    /// Re-refine intrinsics in rig BA (default: false).
    pub refine_intrinsics_in_rig_ba: bool,
    /// Fix first rig pose for gauge freedom (default: true, fixes view 0).
    pub fix_first_rig_pose: bool,
}

impl Default for RigHandeyeRigConfig {
    fn default() -> Self {
        Self {
            reference_camera_idx: 0,
            refine_intrinsics_in_rig_ba: false,
            fix_first_rig_pose: true,
        }
    }
}

/// Hand-eye linear initialization options for rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeInitConfig {
    /// Hand-eye mode: EyeInHand or EyeToHand.
    pub handeye_mode: HandEyeMode,
    /// Minimum motion angle (degrees) for linear hand-eye initialization.
    pub min_motion_angle_deg: f64,
}

impl Default for RigHandeyeInitConfig {
    fn default() -> Self {
        Self {
            handeye_mode: HandEyeMode::EyeInHand,
            min_motion_angle_deg: 5.0,
        }
    }
}

/// Solver options shared across rig and hand-eye optimization stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeSolverConfig {
    /// Maximum iterations for optimization.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
    /// Robust loss function for outlier handling.
    pub robust_loss: RobustLoss,
}

impl Default for RigHandeyeSolverConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
        }
    }
}

/// Hand-eye bundle-adjustment options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeBaConfig {
    /// Refine robot poses in hand-eye BA (default: true).
    pub refine_robot_poses: bool,
    /// Robot rotation sigma for prior (radians). Default: 0.5° ≈ 0.0087 rad.
    pub robot_rot_sigma: f64,
    /// Robot translation sigma for prior (meters). Default: 1mm = 0.001.
    pub robot_trans_sigma: f64,
    /// Refine cam_se3_rig in hand-eye BA (default: false).
    /// When true, rig extrinsics are further refined during hand-eye optimization.
    pub refine_cam_se3_rig_in_handeye_ba: bool,
}

impl Default for RigHandeyeBaConfig {
    fn default() -> Self {
        Self {
            refine_robot_poses: true,
            robot_rot_sigma: 0.5_f64.to_radians(), // 0.5 degrees
            robot_trans_sigma: 0.001,              // 1 mm
            refine_cam_se3_rig_in_handeye_ba: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Export format for rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeExport {
    /// Per-camera calibrated intrinsics + distortion.
    pub cameras: Vec<PinholeCamera>,

    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    /// Transform from rig frame to camera frame.
    pub cam_se3_rig: Vec<Iso3>,

    /// Hand-eye mode used to interpret mode-dependent transforms.
    pub handeye_mode: HandEyeMode,

    /// Eye-in-hand: gripper-to-rig transform `gripper_se3_rig` (T_G_R).
    ///
    /// `None` for EyeToHand mode.
    pub gripper_se3_rig: Option<Iso3>,

    /// Eye-to-hand: rig-to-base transform `rig_se3_base` (T_R_B).
    ///
    /// `None` for EyeInHand mode.
    pub rig_se3_base: Option<Iso3>,

    /// Eye-in-hand: base-to-target transform `base_se3_target` (T_B_T).
    ///
    /// `None` for EyeToHand mode.
    pub base_se3_target: Option<Iso3>,

    /// Eye-to-hand: gripper-to-target transform `gripper_se3_target` (T_G_T).
    ///
    /// `None` for EyeInHand mode.
    pub gripper_se3_target: Option<Iso3>,

    /// Per-view robot pose corrections (if refinement enabled).
    /// Each element is [rx, ry, rz, tx, ty, tz] in se(3).
    pub robot_deltas: Option<Vec<[f64; 6]>>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,

    /// Per-camera reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// ProblemType Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-camera rig hand-eye calibration problem.
///
/// Calibrates a multi-camera rig mounted on a robot arm, including:
/// - Per-camera intrinsics and distortion
/// - Per-camera extrinsics (camera-to-rig transforms)
/// - Rig hand-eye transform (gripper-to-rig)
/// - Static target pose in robot base frame
///
/// # Conventions
///
/// - `cam_se3_rig` = T_C_R (transform from rig to camera frame)
/// - EyeInHand export: `gripper_se3_rig` (T_G_R), `base_se3_target` (T_B_T)
/// - EyeToHand export: `rig_se3_base` (T_R_B), `gripper_se3_target` (T_G_T)
/// - Reference camera has identity extrinsics (defines rig frame)
///
/// # Example
///
/// ```no_run
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::rig_handeye::{
///     RigHandeyeProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
///     step_rig_init, step_rig_optimize, step_handeye_init, step_handeye_optimize,
/// };
/// # fn main() -> anyhow::Result<()> {
/// # let rig_dataset = unimplemented!();
///
/// let mut session = CalibrationSession::<RigHandeyeProblem>::new();
/// session.set_input(rig_dataset)?;
///
/// step_intrinsics_init_all(&mut session, None)?;
/// step_intrinsics_optimize_all(&mut session, None)?;
/// step_rig_init(&mut session)?;
/// step_rig_optimize(&mut session, None)?;
/// step_handeye_init(&mut session, None)?;
/// step_handeye_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RigHandeyeProblem;

impl ProblemType for RigHandeyeProblem {
    type Config = RigHandeyeConfig;
    type Input = RigHandeyeInput;
    type State = RigHandeyeState;
    type Output = HandEyeEstimate;
    type Export = RigHandeyeExport;

    fn name() -> &'static str {
        "rig_handeye_v2"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<()> {
        ensure!(
            input.num_views() >= 3,
            "need at least 3 views for calibration (got {})",
            input.num_views()
        );

        ensure!(
            input.num_cameras >= 2,
            "need at least 2 cameras for rig calibration (got {})",
            input.num_cameras
        );

        // Check each view has correct number of cameras and robot pose
        for (i, view) in input.views.iter().enumerate() {
            ensure!(
                view.obs.cameras.len() == input.num_cameras,
                "view {} has {} cameras, expected {}",
                i,
                view.obs.cameras.len(),
                input.num_cameras
            );

            // Check at least one camera has observations in this view
            let has_obs = view.obs.cameras.iter().any(|c| c.is_some());
            ensure!(has_obs, "view {} has no observations from any camera", i);
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<()> {
        ensure!(config.solver.max_iters > 0, "max_iters must be positive");
        ensure!(
            config.intrinsics.init_iterations > 0,
            "intrinsics_init_iterations must be positive"
        );
        ensure!(
            config.handeye_init.min_motion_angle_deg > 0.0,
            "min_motion_angle_deg must be positive"
        );
        ensure!(
            config.handeye_ba.robot_rot_sigma > 0.0,
            "robot_rot_sigma must be positive"
        );
        ensure!(
            config.handeye_ba.robot_trans_sigma > 0.0,
            "robot_trans_sigma must be positive"
        );
        Ok(())
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<()> {
        ensure!(
            config.rig.reference_camera_idx < input.num_cameras,
            "reference_camera_idx {} is out of range (num_cameras = {})",
            config.rig.reference_camera_idx,
            input.num_cameras
        );
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(output: &Self::Output, config: &Self::Config) -> Result<Self::Export> {
        let cam_se3_rig: Vec<Iso3> = output
            .params
            .cam_to_rig
            .iter()
            .map(|t| t.inverse())
            .collect();

        let target_pose = output
            .params
            .target_poses
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("no target pose in output"))?;

        let (gripper_se3_rig, rig_se3_base, base_se3_target, gripper_se3_target) = match config
            .handeye_init
            .handeye_mode
        {
            HandEyeMode::EyeInHand => (Some(output.params.handeye), None, Some(target_pose), None),
            HandEyeMode::EyeToHand => (None, Some(output.params.handeye), None, Some(target_pose)),
        };

        Ok(RigHandeyeExport {
            cameras: output.params.cameras.clone(),
            cam_se3_rig,
            handeye_mode: config.handeye_init.handeye_mode,
            gripper_se3_rig,
            rig_se3_base,
            base_se3_target,
            gripper_se3_target,
            robot_deltas: output.robot_deltas.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{CorrespondenceView, Pt2, Pt3, RigDataset, RigView, RigViewObs};

    fn make_minimal_obs() -> CorrespondenceView {
        CorrespondenceView::new(
            vec![
                Pt3::new(0.0, 0.0, 0.0),
                Pt3::new(0.05, 0.0, 0.0),
                Pt3::new(0.05, 0.05, 0.0),
                Pt3::new(0.0, 0.05, 0.0),
            ],
            vec![
                Pt2::new(100.0, 100.0),
                Pt2::new(200.0, 100.0),
                Pt2::new(200.0, 200.0),
                Pt2::new(100.0, 200.0),
            ],
        )
        .unwrap()
    }

    fn make_minimal_input() -> RigHandeyeInput {
        let views = (0..3)
            .map(|_| RigView {
                meta: RobotPoseMeta {
                    base_se3_gripper: Iso3::identity(),
                },
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs()), Some(make_minimal_obs())],
                },
            })
            .collect();

        RigDataset::new(views, 2).unwrap()
    }

    #[test]
    fn validate_input_requires_3_views() {
        let input = make_minimal_input();
        let result = RigHandeyeProblem::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_requires_2_cameras() {
        // Create single-camera input (should fail)
        let views = (0..3)
            .map(|_| RigView {
                meta: RobotPoseMeta {
                    base_se3_gripper: Iso3::identity(),
                },
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs())],
                },
            })
            .collect();

        let input = RigDataset::new(views, 1).unwrap();
        let result = RigHandeyeProblem::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 cameras"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = RigHandeyeConfig::default();
        let result = RigHandeyeProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_config_checks_reference_camera() {
        let input = make_minimal_input();
        let config = RigHandeyeConfig {
            rig: RigHandeyeRigConfig {
                reference_camera_idx: 5, // Out of range
                ..RigHandeyeRigConfig::default()
            },
            ..RigHandeyeConfig::default()
        };

        let result = RigHandeyeProblem::validate_input_config(&input, &config);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("reference_camera_idx")
        );
    }

    #[test]
    fn config_json_roundtrip() {
        let config = RigHandeyeConfig {
            solver: RigHandeyeSolverConfig {
                max_iters: 100,
                robust_loss: RobustLoss::Huber { scale: 2.5 },
                ..RigHandeyeSolverConfig::default()
            },
            rig: RigHandeyeRigConfig {
                reference_camera_idx: 1,
                refine_intrinsics_in_rig_ba: true,
                ..RigHandeyeRigConfig::default()
            },
            handeye_init: RigHandeyeInitConfig {
                handeye_mode: HandEyeMode::EyeToHand,
                ..RigHandeyeInitConfig::default()
            },
            handeye_ba: RigHandeyeBaConfig {
                refine_robot_poses: false,
                ..RigHandeyeBaConfig::default()
            },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigHandeyeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.solver.max_iters, 100);
        assert_eq!(restored.rig.reference_camera_idx, 1);
        assert!(restored.rig.refine_intrinsics_in_rig_ba);
        assert!(!restored.handeye_ba.refine_robot_poses);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigHandeyeProblem::name(), "rig_handeye_v2");
        assert_eq!(RigHandeyeProblem::schema_version(), 1);
    }

    fn make_dummy_output() -> HandEyeEstimate {
        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );

        HandEyeEstimate {
            params: HandEyeParams {
                cameras: vec![camera],
                cam_to_rig: vec![Iso3::identity()],
                handeye: Iso3::identity(),
                target_poses: vec![Iso3::identity()],
            },
            report: SolveReport { final_cost: 0.0 },
            robot_deltas: None,
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0],
        }
    }

    #[test]
    fn export_eye_in_hand_is_explicit() {
        let output = make_dummy_output();
        let config = RigHandeyeConfig {
            handeye_init: RigHandeyeInitConfig {
                handeye_mode: HandEyeMode::EyeInHand,
                ..RigHandeyeInitConfig::default()
            },
            ..Default::default()
        };
        let export = RigHandeyeProblem::export(&output, &config).unwrap();

        assert!(matches!(export.handeye_mode, HandEyeMode::EyeInHand));
        assert!(export.gripper_se3_rig.is_some());
        assert!(export.base_se3_target.is_some());
        assert!(export.rig_se3_base.is_none());
        assert!(export.gripper_se3_target.is_none());
    }

    #[test]
    fn export_eye_to_hand_is_explicit() {
        let output = make_dummy_output();
        let config = RigHandeyeConfig {
            handeye_init: RigHandeyeInitConfig {
                handeye_mode: HandEyeMode::EyeToHand,
                ..RigHandeyeInitConfig::default()
            },
            ..Default::default()
        };
        let export = RigHandeyeProblem::export(&output, &config).unwrap();

        assert!(matches!(export.handeye_mode, HandEyeMode::EyeToHand));
        assert!(export.rig_se3_base.is_some());
        assert!(export.gripper_se3_target.is_some());
        assert!(export.gripper_se3_rig.is_none());
        assert!(export.base_se3_target.is_none());
    }
}
