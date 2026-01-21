//! [`ProblemType`] implementation for multi-camera rig extrinsics calibration.
//!
//! This module provides the `RigExtrinsicsProblem` type that implements
//! the session API's `ProblemType` trait.

use anyhow::{ensure, Result};
use calib_core::{Iso3, NoMeta, PinholeCamera, RigDataset};
use calib_optim::{RigExtrinsicsEstimate, RobustLoss};
use serde::{Deserialize, Serialize};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigExtrinsicsState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Type
// ─────────────────────────────────────────────────────────────────────────────

/// Input for rig extrinsics calibration.
///
/// Reuses `RigDataset<NoMeta>` from calib_core.
pub type RigExtrinsicsInput = RigDataset<NoMeta>;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for multi-camera rig extrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsConfig {
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
}

impl Default for RigExtrinsicsConfig {
    fn default() -> Self {
        Self {
            // Intrinsics
            intrinsics_init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            // Rig
            reference_camera_idx: 0,
            // Optimization
            max_iters: 50,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            // Rig BA
            refine_intrinsics_in_rig_ba: false,
            fix_first_rig_pose: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Export format for rig extrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigExtrinsicsExport {
    /// Per-camera calibrated intrinsics + distortion.
    pub cameras: Vec<PinholeCamera>,

    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    /// Transform from rig frame to camera frame.
    pub cam_se3_rig: Vec<Iso3>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ProblemType Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-camera rig extrinsics calibration problem.
///
/// Calibrates a multi-camera rig, including:
/// - Per-camera intrinsics and distortion
/// - Per-camera extrinsics (camera-to-rig transforms)
/// - Per-view rig poses (rig-to-target)
///
/// # Conventions
///
/// - `cam_se3_rig` = T_C_R (transform from rig to camera frame)
/// - Reference camera has identity extrinsics (defines rig frame)
///
/// # Example
///
/// ```ignore
/// use calib_pipeline::session::v2::CalibrationSession;
/// use calib_pipeline::rig_extrinsics::{
///     RigExtrinsicsProblem, step_intrinsics_init_all, step_intrinsics_optimize_all,
///     step_rig_init, step_rig_optimize,
/// };
///
/// let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
/// session.set_input(rig_dataset)?;
///
/// step_intrinsics_init_all(&mut session, None)?;
/// step_intrinsics_optimize_all(&mut session, None)?;
/// step_rig_init(&mut session)?;
/// step_rig_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// ```
#[derive(Debug)]
pub struct RigExtrinsicsProblem;

impl ProblemType for RigExtrinsicsProblem {
    type Config = RigExtrinsicsConfig;
    type Input = RigExtrinsicsInput;
    type State = RigExtrinsicsState;
    type Output = RigExtrinsicsEstimate;
    type Export = RigExtrinsicsExport;

    fn name() -> &'static str {
        "rig_extrinsics_v2"
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

        // Check each view has correct number of cameras
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
        ensure!(config.max_iters > 0, "max_iters must be positive");
        ensure!(
            config.intrinsics_init_iterations > 0,
            "intrinsics_init_iterations must be positive"
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

        Ok(RigExtrinsicsExport {
            cameras: output.params.cameras.clone(),
            cam_se3_rig: output.params.cam_to_rig.clone(),
            mean_reproj_error,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{CorrespondenceView, Pt2, Pt3, RigView, RigViewObs};

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

    fn make_minimal_input() -> RigExtrinsicsInput {
        let views = (0..3)
            .map(|_| RigView {
                meta: NoMeta,
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
        let result = RigExtrinsicsProblem::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_requires_2_cameras() {
        // Create single-camera input (should fail)
        let views = (0..3)
            .map(|_| RigView {
                meta: NoMeta,
                obs: RigViewObs {
                    cameras: vec![Some(make_minimal_obs())],
                },
            })
            .collect();

        let input = RigDataset::new(views, 1).unwrap();
        let result = RigExtrinsicsProblem::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2 cameras"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = RigExtrinsicsConfig::default();
        let result = RigExtrinsicsProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_config_checks_reference_camera() {
        let input = make_minimal_input();
        let config = RigExtrinsicsConfig {
            reference_camera_idx: 5, // Out of range
            ..RigExtrinsicsConfig::default()
        };

        let result = RigExtrinsicsProblem::validate_input_config(&input, &config);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("reference_camera_idx"));
    }

    #[test]
    fn config_json_roundtrip() {
        let config = RigExtrinsicsConfig {
            max_iters: 100,
            reference_camera_idx: 1,
            refine_intrinsics_in_rig_ba: true,
            robust_loss: RobustLoss::Huber { scale: 2.5 },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigExtrinsicsConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.max_iters, 100);
        assert_eq!(restored.reference_camera_idx, 1);
        assert!(restored.refine_intrinsics_in_rig_ba);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigExtrinsicsProblem::name(), "rig_extrinsics_v2");
        assert_eq!(RigExtrinsicsProblem::schema_version(), 1);
    }
}
