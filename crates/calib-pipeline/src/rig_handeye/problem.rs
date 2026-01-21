//! [`ProblemType`] implementation for multi-camera rig hand-eye calibration.
//!
//! This module provides the `RigHandeyeProblemV2` type that implements
//! the v2 session API's `ProblemType` trait.

use anyhow::{ensure, Result};
use calib_core::{Iso3, PinholeCamera};
use calib_optim::{HandEyeEstimate, HandEyeMode, RigDataset, RobotPoseMeta, RobustLoss};
use serde::{Deserialize, Serialize};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigHandeyeState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Type
// ─────────────────────────────────────────────────────────────────────────────

/// Input for rig hand-eye calibration.
///
/// Reuses `RigDataset<RobotPoseMeta>` from calib_optim.
/// Each view contains robot pose metadata and per-camera observations.
pub type RigHandeyeInput = RigDataset<RobotPoseMeta>;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-camera rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigHandeyeConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Per-camera intrinsics options
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of iterations for iterative intrinsics estimation.
    pub intrinsics_init_iterations: usize,

    /// Fix k3 during intrinsics calibration.
    pub fix_k3: bool,

    /// Fix tangential distortion (p1, p2).
    pub fix_tangential: bool,

    /// Enforce zero skew.
    pub zero_skew: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig options
    // ─────────────────────────────────────────────────────────────────────────
    /// Reference camera index for rig frame (identity extrinsics).
    pub reference_camera_idx: usize,

    // ─────────────────────────────────────────────────────────────────────────
    // Hand-eye options
    // ─────────────────────────────────────────────────────────────────────────
    /// Hand-eye mode: EyeInHand or EyeToHand.
    pub handeye_mode: HandEyeMode,

    /// Minimum motion angle (degrees) for linear hand-eye initialization.
    pub min_motion_angle_deg: f64,

    // ─────────────────────────────────────────────────────────────────────────
    // Optimization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Maximum iterations for optimization.
    pub max_iters: usize,

    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,

    /// Robust loss function for outlier handling.
    pub robust_loss: RobustLoss,

    // ─────────────────────────────────────────────────────────────────────────
    // Rig BA options
    // ─────────────────────────────────────────────────────────────────────────
    /// Re-refine intrinsics in rig BA (default: false).
    pub refine_intrinsics_in_rig_ba: bool,

    /// Fix first rig pose for gauge freedom (default: true, fixes view 0).
    pub fix_first_rig_pose: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Hand-eye BA options
    // ─────────────────────────────────────────────────────────────────────────
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

impl Default for RigHandeyeConfig {
    fn default() -> Self {
        Self {
            // Intrinsics
            intrinsics_init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            // Rig
            reference_camera_idx: 0,
            // Hand-eye
            handeye_mode: HandEyeMode::EyeInHand,
            min_motion_angle_deg: 5.0,
            // Optimization
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            // Rig BA
            refine_intrinsics_in_rig_ba: false,
            fix_first_rig_pose: true,
            // Hand-eye BA
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

    /// Rig hand-eye transform.
    /// For EyeInHand: `gripper_se3_rig` (T_G_R).
    pub handeye: Iso3,

    /// Target pose in base frame: `target_se3_base` (T_T_B).
    /// Single static target.
    pub target_se3_base: Iso3,

    /// Per-view robot pose corrections (if refinement enabled).
    /// Each element is [rx, ry, rz, tx, ty, tz] in se(3).
    pub robot_deltas: Option<Vec<[f64; 6]>>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
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
/// - `handeye` = T_G_R (gripper to rig, for EyeInHand mode)
/// - `target_se3_base` = T_T_B (single static target in base frame)
/// - Reference camera has identity extrinsics (defines rig frame)
///
/// # Example
///
/// ```ignore
/// use calib_pipeline::session::CalibrationSession;
/// use calib_pipeline::rig_handeye::{
///     RigHandeyeProblemV2, step_intrinsics_init_all, step_intrinsics_optimize_all,
///     step_rig_init, step_rig_optimize, step_handeye_init, step_handeye_optimize,
/// };
///
/// let mut session = CalibrationSession::<RigHandeyeProblemV2>::new();
/// session.set_input(rig_dataset);
///
/// step_intrinsics_init_all(&mut session, None);
/// step_intrinsics_optimize_all(&mut session, None);
/// step_rig_init(&mut session);
/// step_rig_optimize(&mut session, None);
/// step_handeye_init(&mut session, None);
/// step_handeye_optimize(&mut session, None);
///
/// let export = session.export();
/// ```
#[derive(Debug)]
pub struct RigHandeyeProblemV2;

impl ProblemType for RigHandeyeProblemV2 {
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
            ensure!(
                has_obs,
                "view {} has no observations from any camera",
                i
            );
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<()> {
        ensure!(config.max_iters > 0, "max_iters must be positive");
        ensure!(
            config.intrinsics_init_iterations > 0,
            "intrinsics_init_iterations must be positive"
        );
        ensure!(
            config.min_motion_angle_deg > 0.0,
            "min_motion_angle_deg must be positive"
        );
        ensure!(
            config.robot_rot_sigma > 0.0,
            "robot_rot_sigma must be positive"
        );
        ensure!(
            config.robot_trans_sigma > 0.0,
            "robot_trans_sigma must be positive"
        );
        Ok(())
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<()> {
        ensure!(
            config.reference_camera_idx < input.num_cameras,
            "reference_camera_idx {} is out of range (num_cameras = {})",
            config.reference_camera_idx,
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

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
        // Compute mean reprojection error from final cost
        let mean_reproj_error = output.report.final_cost.sqrt();

        // Extract robot deltas if present
        let robot_deltas = output.robot_deltas.clone();

        Ok(RigHandeyeExport {
            cameras: output.params.cameras.clone(),
            cam_se3_rig: output.params.cam_to_rig.clone(),
            handeye: output.params.handeye,
            target_se3_base: output
                .params
                .target_poses
                .first()
                .copied()
                .unwrap_or(Iso3::identity()),
            robot_deltas,
            mean_reproj_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{CorrespondenceView, Pt2, Pt3, RigDataset, RigView, RigViewObs};

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
                    robot_pose: Iso3::identity(),
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
        let result = RigHandeyeProblemV2::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_requires_2_cameras() {
        // Create single-camera input (should fail)
        let views = (0..3)
            .map(|_| RigView {
                meta: RobotPoseMeta {
                    robot_pose: Iso3::identity(),
                },
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs())],
                },
            })
            .collect();

        let input = RigDataset::new(views, 1).unwrap();
        let result = RigHandeyeProblemV2::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 cameras"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = RigHandeyeConfig::default();
        let result = RigHandeyeProblemV2::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_config_checks_reference_camera() {
        let input = make_minimal_input();
        let mut config = RigHandeyeConfig::default();
        config.reference_camera_idx = 5; // Out of range

        let result = RigHandeyeProblemV2::validate_input_config(&input, &config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("reference_camera_idx"));
    }

    #[test]
    fn config_json_roundtrip() {
        let config = RigHandeyeConfig {
            max_iters: 100,
            reference_camera_idx: 1,
            refine_intrinsics_in_rig_ba: true,
            handeye_mode: HandEyeMode::EyeToHand,
            refine_robot_poses: false,
            robust_loss: RobustLoss::Huber { scale: 2.5 },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigHandeyeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.max_iters, 100);
        assert_eq!(restored.reference_camera_idx, 1);
        assert!(restored.refine_intrinsics_in_rig_ba);
        assert!(!restored.refine_robot_poses);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigHandeyeProblemV2::name(), "rig_handeye_v2");
        assert_eq!(RigHandeyeProblemV2::schema_version(), 1);
    }
}
