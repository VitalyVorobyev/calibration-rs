//! [`ProblemType`] implementation for multi-camera rig hand-eye calibration.
//!
//! This module provides the `RigHandeyeProblem` type that implements
//! the v2 session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    Camera, FeatureResidualHistogram, ImageManifest, Iso3, PerFeatureResiduals, Pinhole,
    PinholeCamera, ScheimpflugParams, build_feature_histogram, compute_rig_target_residuals,
};
use vision_calibration_optim::{
    HandEyeEstimate as PinholeHandEyeEstimate, HandEyeMode,
    HandEyeScheimpflugEstimate as ScheimpflugHandEyeEstimate, RigDataset, RobotPoseMeta,
    RobustLoss, handeye_observer_se3_target,
};
#[cfg(test)]
use vision_calibration_optim::{HandEyeParams, SolveReport};

pub use crate::rig_family::SensorMode;

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
///
/// Shared between pinhole and Scheimpflug rigs; the [`SensorMode`] field
/// `sensor` selects the sensor flavour.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RigHandeyeConfig {
    /// Per-camera intrinsics initialization options.
    pub intrinsics: RigHandeyeIntrinsicsConfig,
    /// Sensor flavour (pinhole or Scheimpflug). Default is `Pinhole`.
    #[serde(default)]
    pub sensor: SensorMode,
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
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
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
    /// Refine Scheimpflug tilt parameters in hand-eye BA (default: false).
    /// Only consulted when [`SensorMode::Scheimpflug`] is configured; ignored
    /// for [`SensorMode::Pinhole`].
    #[serde(default)]
    pub refine_scheimpflug_in_handeye_ba: bool,
}

impl Default for RigHandeyeBaConfig {
    fn default() -> Self {
        Self {
            refine_robot_poses: true,
            robot_rot_sigma: 0.5_f64.to_radians(), // 0.5 degrees
            robot_trans_sigma: 0.001,              // 1 mm
            refine_cam_se3_rig_in_handeye_ba: false,
            refine_scheimpflug_in_handeye_ba: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output (pinhole or Scheimpflug variant)
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the hand-eye BA stage. Pinhole and Scheimpflug rigs return
/// structurally similar but type-distinct optim estimates; this enum
/// preserves both as `Self::Output` for [`RigHandeyeProblem`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RigHandeyeOutput {
    /// Pinhole hand-eye BA estimate.
    Pinhole(PinholeHandEyeEstimate),
    /// Scheimpflug hand-eye BA estimate (per-camera sensors included).
    Scheimpflug(ScheimpflugHandEyeEstimate),
}

impl RigHandeyeOutput {
    pub fn final_cost(&self) -> f64 {
        match self {
            Self::Pinhole(e) => e.report.final_cost,
            Self::Scheimpflug(e) => e.report.final_cost,
        }
    }

    pub fn mean_reproj_error(&self) -> f64 {
        match self {
            Self::Pinhole(e) => e.mean_reproj_error,
            Self::Scheimpflug(e) => e.mean_reproj_error,
        }
    }

    pub fn cameras(&self) -> &[PinholeCamera] {
        match self {
            Self::Pinhole(e) => &e.params.cameras,
            Self::Scheimpflug(e) => &e.params.cameras,
        }
    }

    pub fn cam_to_rig(&self) -> &[Iso3] {
        match self {
            Self::Pinhole(e) => &e.params.cam_to_rig,
            Self::Scheimpflug(e) => &e.params.cam_to_rig,
        }
    }

    pub fn target_poses(&self) -> &[Iso3] {
        match self {
            Self::Pinhole(e) => &e.params.target_poses,
            Self::Scheimpflug(e) => &e.params.target_poses,
        }
    }

    pub fn handeye(&self) -> &Iso3 {
        match self {
            Self::Pinhole(e) => &e.params.handeye,
            Self::Scheimpflug(e) => &e.params.handeye,
        }
    }

    pub fn robot_deltas(&self) -> Option<&[[f64; 6]]> {
        match self {
            Self::Pinhole(e) => e.robot_deltas.as_deref(),
            Self::Scheimpflug(e) => e.robot_deltas.as_deref(),
        }
    }

    pub fn sensors(&self) -> Option<&[ScheimpflugParams]> {
        match self {
            Self::Pinhole(_) => None,
            Self::Scheimpflug(e) => Some(&e.params.sensors),
        }
    }

    pub fn per_cam_reproj_errors(&self) -> &[f64] {
        match self {
            Self::Pinhole(e) => &e.per_cam_reproj_errors,
            Self::Scheimpflug(e) => &e.per_cam_reproj_errors,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Export format for rig hand-eye calibration.
///
/// Common to pinhole and Scheimpflug rigs. `sensors` is `None` for pinhole
/// rigs and `Some(_)` for Scheimpflug rigs, matching the configured
/// [`SensorMode`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigHandeyeExport {
    /// Per-camera calibrated intrinsics + distortion (pinhole core).
    pub cameras: Vec<PinholeCamera>,

    /// Per-camera Scheimpflug sensor parameters. `None` for pinhole rigs;
    /// `Some(_)` for Scheimpflug rigs (one entry per camera).
    #[serde(default)]
    pub sensors: Option<Vec<ScheimpflugParams>>,

    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    /// Transform from rig frame to camera frame.
    pub cam_se3_rig: Vec<Iso3>,

    /// Per-view rig poses: `rig_se3_target` (T_R_T), derived from the
    /// hand-eye chain (`handeye_observer_se3_target`) so downstream
    /// viewers (3D scene, epipolar overlay) can read board poses
    /// without re-implementing the chain. One entry per input view.
    /// `#[serde(default)]` keeps older exports forward-compatible at
    /// load time; they decode with an empty Vec.
    #[serde(default)]
    pub rig_se3_target: Vec<Iso3>,

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

    /// Per-feature reprojection residuals (ADR 0012). Per-view
    /// `rig_se3_target` is derived from the handeye chain
    /// (see [`handeye_observer_se3_target`](vision_calibration_optim::handeye_observer_se3_target)),
    /// then composed with `cam_se3_rig` for projection.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,

    /// Optional image manifest (ADR 0014, viewer-side contract). When
    /// populated, downstream viewers (the diagnose UI) can locate the source
    /// image for each `(pose, camera)` slot. Tiled multi-camera frames
    /// (e.g. 6× 720×540 horizontal strips on the puzzle 130×130 rig) point
    /// multiple `FrameRef`s at the same `path` with disjoint ROIs. `None`
    /// means "no images shipped"; the calibration pipeline never reads
    /// this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
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
    type Output = RigHandeyeOutput;
    type Export = RigHandeyeExport;

    fn name() -> &'static str {
        "rig_handeye_v2"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        if input.num_views() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: input.num_views(),
            });
        }

        if input.num_cameras < 2 {
            return Err(Error::InsufficientData {
                need: 2,
                got: input.num_cameras,
            });
        }

        // Check each view has correct number of cameras and robot pose
        for (i, view) in input.views.iter().enumerate() {
            if view.obs.cameras.len() != input.num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {} has {} cameras, expected {}",
                    i,
                    view.obs.cameras.len(),
                    input.num_cameras
                )));
            }

            // Check at least one camera has observations in this view
            let has_obs = view.obs.cameras.iter().any(|c| c.is_some());
            if !has_obs {
                return Err(Error::invalid_input(format!(
                    "view {} has no observations from any camera",
                    i
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.solver.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.intrinsics.init_iterations == 0 {
            return Err(Error::invalid_input(
                "intrinsics_init_iterations must be positive",
            ));
        }
        if config.handeye_init.min_motion_angle_deg <= 0.0 {
            return Err(Error::invalid_input(
                "min_motion_angle_deg must be positive",
            ));
        }
        if config.handeye_ba.robot_rot_sigma <= 0.0 {
            return Err(Error::invalid_input("robot_rot_sigma must be positive"));
        }
        if config.handeye_ba.robot_trans_sigma <= 0.0 {
            return Err(Error::invalid_input("robot_trans_sigma must be positive"));
        }
        Ok(())
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        if config.rig.reference_camera_idx >= input.num_cameras {
            return Err(Error::invalid_input(format!(
                "reference_camera_idx {} is out of range (num_cameras = {})",
                config.rig.reference_camera_idx, input.num_cameras
            )));
        }
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        let cam_se3_rig: Vec<Iso3> = output.cam_to_rig().iter().map(|t| t.inverse()).collect();

        let target_pose = output
            .target_poses()
            .first()
            .copied()
            .ok_or_else(|| Error::invalid_input("no target pose in output"))?;

        let mode = config.handeye_init.handeye_mode;
        let handeye = *output.handeye();
        let (gripper_se3_rig, rig_se3_base, base_se3_target, gripper_se3_target) = match mode {
            HandEyeMode::EyeInHand => (Some(handeye), None, Some(target_pose), None),
            HandEyeMode::EyeToHand => (None, Some(handeye), None, Some(target_pose)),
        };

        // Per-feature residuals: derive rig_se3_target per view from the
        // handeye chain, then project. For Scheimpflug, splice tilted
        // sensors into the projection chain.
        let robot_poses: Vec<Iso3> = input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper)
            .collect();
        let rig_se3_target = handeye_observer_se3_target(
            mode,
            &handeye,
            &target_pose,
            &robot_poses,
            output.robot_deltas(),
        );
        let target = match output {
            RigHandeyeOutput::Pinhole(estimate) => compute_rig_target_residuals(
                &estimate.params.cameras,
                input,
                &cam_se3_rig,
                &rig_se3_target,
            )?,
            RigHandeyeOutput::Scheimpflug(estimate) => {
                let scheimpflug_cameras: Vec<_> = estimate
                    .params
                    .cameras
                    .iter()
                    .zip(estimate.params.sensors.iter())
                    .map(|(cam, sensor)| Camera::new(Pinhole, cam.dist, sensor.compile(), cam.k))
                    .collect();
                compute_rig_target_residuals(
                    &scheimpflug_cameras,
                    input,
                    &cam_se3_rig,
                    &rig_se3_target,
                )?
            }
        };
        let target_hist_per_camera: Vec<FeatureResidualHistogram> = (0..input.num_cameras)
            .map(|cam_idx| {
                build_feature_histogram(
                    target
                        .iter()
                        .filter(|r| r.camera == cam_idx)
                        .filter_map(|r| r.error_px),
                )
            })
            .collect();

        Ok(RigHandeyeExport {
            cameras: output.cameras().to_vec(),
            sensors: output.sensors().map(|s| s.to_vec()),
            cam_se3_rig,
            rig_se3_target: rig_se3_target.clone(),
            handeye_mode: mode,
            gripper_se3_rig,
            rig_se3_base,
            base_se3_target,
            gripper_se3_target,
            robot_deltas: output.robot_deltas().map(|d| d.to_vec()),
            mean_reproj_error: output.mean_reproj_error(),
            per_cam_reproj_errors: output.per_cam_reproj_errors().to_vec(),
            per_feature_residuals: PerFeatureResiduals {
                target,
                laser: Vec::new(),
                target_hist_per_camera: Some(target_hist_per_camera),
                laser_hist_per_camera: None,
            },
            // Manifest is populated by callers that also wrote images for
            // the dataset (e.g. the puzzle 130×130 example); the pipeline
            // itself never has image paths to fill in.
            image_manifest: None,
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
        assert!(result.unwrap_err().to_string().contains("need 2"));
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

    fn make_dummy_output() -> RigHandeyeOutput {
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

        RigHandeyeOutput::Pinhole(PinholeHandEyeEstimate {
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
        })
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
        let dummy_view = RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: Iso3::identity(),
            },
            obs: RigViewObs {
                cameras: vec![Some(
                    CorrespondenceView::new(
                        vec![Pt3::new(0.0, 0.0, 0.0); 4],
                        vec![Pt2::new(0.0, 0.0); 4],
                    )
                    .unwrap(),
                )],
            },
        };
        let dummy_input = RigDataset::new(vec![dummy_view], 1).unwrap();
        let export = RigHandeyeProblem::export(&dummy_input, &output, &config).unwrap();

        assert!(matches!(export.handeye_mode, HandEyeMode::EyeInHand));
        assert!(export.gripper_se3_rig.is_some());
        assert!(export.base_se3_target.is_some());
        assert!(export.rig_se3_base.is_none());
        assert!(export.gripper_se3_target.is_none());
        assert_eq!(
            export.rig_se3_target.len(),
            1,
            "rig_se3_target must have one entry per input view"
        );
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
        let dummy_view = RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: Iso3::identity(),
            },
            obs: RigViewObs {
                cameras: vec![Some(
                    CorrespondenceView::new(
                        vec![Pt3::new(0.0, 0.0, 0.0); 4],
                        vec![Pt2::new(0.0, 0.0); 4],
                    )
                    .unwrap(),
                )],
            },
        };
        let dummy_input = RigDataset::new(vec![dummy_view], 1).unwrap();
        let export = RigHandeyeProblem::export(&dummy_input, &output, &config).unwrap();

        assert!(matches!(export.handeye_mode, HandEyeMode::EyeToHand));
        assert!(export.rig_se3_base.is_some());
        assert!(export.gripper_se3_target.is_some());
        assert!(export.gripper_se3_rig.is_none());
        assert!(export.base_se3_target.is_none());
    }

    #[test]
    fn export_image_manifest_defaults_absent_from_wire() {
        // ADR 0014: when the pipeline emits an export the manifest is
        // None and `skip_serializing_if` keeps the legacy JSON byte-stable.
        let output = make_dummy_output();
        let config = RigHandeyeConfig::default();
        let dummy_view = RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: Iso3::identity(),
            },
            obs: RigViewObs {
                cameras: vec![Some(
                    CorrespondenceView::new(
                        vec![Pt3::new(0.0, 0.0, 0.0); 4],
                        vec![Pt2::new(0.0, 0.0); 4],
                    )
                    .unwrap(),
                )],
            },
        };
        let dummy_input = RigDataset::new(vec![dummy_view], 1).unwrap();
        let export = RigHandeyeProblem::export(&dummy_input, &output, &config).unwrap();
        assert!(export.image_manifest.is_none());

        let json = serde_json::to_string(&export).unwrap();
        assert!(
            !json.contains("image_manifest"),
            "absent manifest must not appear on the wire"
        );
        let restored: RigHandeyeExport = serde_json::from_str(&json).unwrap();
        assert!(restored.image_manifest.is_none());
    }

    #[test]
    fn export_image_manifest_roundtrip_with_tiled_roi() {
        // The puzzle 130×130 rig writes a single 4320×540 PNG per pose
        // and references each of its six 720×540 camera tiles via ROI;
        // mirror that shape in the test to lock the wire format.
        //
        // Note: per the `image_manifest` coordinate convention, residual
        // pixel coords are *ROI-local* — the ROI is a render-time crop
        // hint only. This test pins the wire format; the convention
        // itself is exercised by the viewer (see app/src/components).
        use vision_calibration_core::{FrameRef, ImageManifest, PixelRect};

        let output = make_dummy_output();
        let config = RigHandeyeConfig::default();
        let dummy_view = RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: Iso3::identity(),
            },
            obs: RigViewObs {
                cameras: vec![Some(
                    CorrespondenceView::new(
                        vec![Pt3::new(0.0, 0.0, 0.0); 4],
                        vec![Pt2::new(0.0, 0.0); 4],
                    )
                    .unwrap(),
                )],
            },
        };
        let dummy_input = RigDataset::new(vec![dummy_view], 1).unwrap();
        let mut export = RigHandeyeProblem::export(&dummy_input, &output, &config).unwrap();
        export.image_manifest = Some(ImageManifest {
            root: std::path::PathBuf::from("."),
            frames: (0..6)
                .map(|cam| FrameRef {
                    pose: 0,
                    camera: cam,
                    path: std::path::PathBuf::from("target_0.png"),
                    roi: Some(PixelRect {
                        x: (cam as u32) * 720,
                        y: 0,
                        w: 720,
                        h: 540,
                    }),
                })
                .collect(),
        });

        let json = serde_json::to_string(&export).unwrap();
        let restored: RigHandeyeExport = serde_json::from_str(&json).unwrap();
        let manifest = restored
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.frames.len(), 6);
        assert_eq!(manifest.frames[0].camera, 0);
        assert_eq!(manifest.frames[5].roi.unwrap().x, 5 * 720);
    }
}
