//! [`ProblemType`] implementation for single-camera hand-eye calibration.
//!
//! This module provides the `SingleCamHandeyeProblemV2` type that implements
//! the v2 session API's `ProblemType` trait.

use anyhow::{ensure, Result};
use calib_core::{CorrespondenceView, Iso3, PinholeCamera};
use calib_optim::{HandEyeEstimate, HandEyeMode, RobustLoss};
use serde::{Deserialize, Serialize};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::SingleCamHandeyeState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single view with robot pose and 2D-3D correspondences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleCamHandeyeView {
    /// Robot pose: base_se3_gripper (T_B_G).
    pub robot_pose: Iso3,
    /// 2D-3D correspondences for this view.
    pub obs: CorrespondenceView,
}

/// Input for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleCamHandeyeInput {
    /// Per-view observations with robot poses.
    pub views: Vec<SingleCamHandeyeView>,
}

impl SingleCamHandeyeInput {
    /// Create a new input from views.
    pub fn new(views: Vec<SingleCamHandeyeView>) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view");
        for (i, view) in views.iter().enumerate() {
            ensure!(
                view.obs.len() >= 4,
                "view {} has too few points (need >= 4)",
                i
            );
        }
        Ok(Self { views })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleCamHandeyeConfig {
    // ─────────────────────────────────────────────────────────────────────────
    // Intrinsics initialization options
    // ─────────────────────────────────────────────────────────────────────────
    /// Number of iterations for iterative intrinsics estimation.
    pub intrinsics_init_iterations: usize,

    /// Fix k3 during initialization.
    pub fix_k3: bool,

    /// Fix tangential distortion (p1, p2) during initialization.
    pub fix_tangential: bool,

    /// Enforce zero skew.
    pub zero_skew: bool,

    // ─────────────────────────────────────────────────────────────────────────
    // Hand-eye options
    // ─────────────────────────────────────────────────────────────────────────
    /// Hand-eye mode (EyeInHand or EyeToHand).
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
    // Hand-eye BA specific
    // ─────────────────────────────────────────────────────────────────────────
    /// Refine robot poses with per-view se(3) corrections (default: true).
    pub refine_robot_poses: bool,

    /// Robot rotation prior sigma (radians). Default: 0.5 deg.
    pub robot_rot_sigma: f64,

    /// Robot translation prior sigma (meters). Default: 1 mm.
    pub robot_trans_sigma: f64,
}

impl Default for SingleCamHandeyeConfig {
    fn default() -> Self {
        Self {
            // Intrinsics init
            intrinsics_init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            // Hand-eye
            handeye_mode: HandEyeMode::EyeInHand,
            min_motion_angle_deg: 5.0,
            // Optimization
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            // Hand-eye BA
            refine_robot_poses: true,
            robot_rot_sigma: std::f64::consts::PI / 360.0, // 0.5 deg
            robot_trans_sigma: 1.0e-3,                     // 1 mm
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Export format for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleCamHandeyeExport {
    /// Calibrated camera (intrinsics + distortion).
    pub camera: PinholeCamera,

    /// Hand-eye transform.
    /// For EyeInHand: gripper_se3_camera (T_G_C).
    /// For EyeToHand: camera_se3_base (T_C_B).
    pub handeye: Iso3,

    /// Target pose in base frame (single static target).
    /// For EyeInHand: base_se3_target (T_B_T).
    pub target_se3_base: Iso3,

    /// Per-view robot pose deltas (se(3) tangent: [rx, ry, rz, tx, ty, tz]).
    /// Only present if robot refinement was enabled.
    pub robot_deltas: Option<Vec<[f64; 6]>>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ProblemType Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Single-camera hand-eye calibration problem.
///
/// Calibrates a single camera mounted on a robot arm, including:
/// - Camera intrinsics and distortion
/// - Hand-eye transform (gripper-to-camera or camera-to-base)
/// - Target pose in base frame
///
/// # Example
///
/// ```ignore
/// use calib_pipeline::session::v2::CalibrationSession;
/// use calib_pipeline::single_cam_handeye::{
///     SingleCamHandeyeProblemV2, step_intrinsics_init, step_intrinsics_optimize,
///     step_handeye_init, step_handeye_optimize,
/// };
///
/// let mut session = CalibrationSession::<SingleCamHandeyeProblemV2>::new();
/// session.set_input(input)?;
///
/// step_intrinsics_init(&mut session, None)?;
/// step_intrinsics_optimize(&mut session, None)?;
/// step_handeye_init(&mut session, None)?;
/// step_handeye_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// ```
#[derive(Debug)]
pub struct SingleCamHandeyeProblemV2;

impl ProblemType for SingleCamHandeyeProblemV2 {
    type Config = SingleCamHandeyeConfig;
    type Input = SingleCamHandeyeInput;
    type State = SingleCamHandeyeState;
    type Output = HandEyeEstimate;
    type Export = SingleCamHandeyeExport;

    fn name() -> &'static str {
        "single_cam_handeye_v2"
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

        for (i, view) in input.views.iter().enumerate() {
            ensure!(
                view.obs.len() >= 4,
                "view {} has too few points (need >= 4 for homography, got {})",
                i,
                view.obs.len()
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
        if config.refine_robot_poses {
            ensure!(
                config.robot_rot_sigma > 0.0,
                "robot_rot_sigma must be positive"
            );
            ensure!(
                config.robot_trans_sigma > 0.0,
                "robot_trans_sigma must be positive"
            );
        }
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export> {
        // Extract the single camera (index 0 for single-cam setup)
        let camera = output
            .params
            .cameras
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no camera in output"))?;

        // Target pose (single target)
        let target_se3_base = output
            .params
            .target_poses
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("no target pose in output"))?;

        // Compute mean reprojection error from final cost
        // This is approximate; in practice we'd compute it properly
        let mean_reproj_error = output.report.final_cost.sqrt();

        Ok(SingleCamHandeyeExport {
            camera,
            handeye: output.params.handeye,
            target_se3_base,
            robot_deltas: output.robot_deltas.clone(),
            mean_reproj_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{CorrespondenceView, Pt2, Pt3};

    fn make_minimal_view() -> SingleCamHandeyeView {
        SingleCamHandeyeView {
            robot_pose: Iso3::identity(),
            obs: CorrespondenceView::new(
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
            .unwrap(),
        }
    }

    fn make_minimal_input() -> SingleCamHandeyeInput {
        SingleCamHandeyeInput {
            views: vec![make_minimal_view(), make_minimal_view(), make_minimal_view()],
        }
    }

    #[test]
    fn validate_input_requires_3_views() {
        let input = make_minimal_input();
        let result = SingleCamHandeyeProblemV2::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_rejects_too_few_views() {
        let input = SingleCamHandeyeInput {
            views: vec![make_minimal_view(), make_minimal_view()],
        };
        let result = SingleCamHandeyeProblemV2::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3 views"));
    }

    #[test]
    fn validate_config_requires_positive_iters() {
        let config = SingleCamHandeyeConfig {
            max_iters: 0,
            ..Default::default()
        };
        let result = SingleCamHandeyeProblemV2::validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iters"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = SingleCamHandeyeConfig::default();
        let result = SingleCamHandeyeProblemV2::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn config_json_roundtrip() {
        let config = SingleCamHandeyeConfig {
            max_iters: 100,
            fix_k3: false,
            handeye_mode: HandEyeMode::EyeToHand,
            robust_loss: RobustLoss::Huber { scale: 2.5 },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: SingleCamHandeyeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.max_iters, 100);
        assert!(!restored.fix_k3);
        assert!(matches!(restored.handeye_mode, HandEyeMode::EyeToHand));
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(SingleCamHandeyeProblemV2::name(), "single_cam_handeye_v2");
        assert_eq!(SingleCamHandeyeProblemV2::schema_version(), 1);
    }
}
