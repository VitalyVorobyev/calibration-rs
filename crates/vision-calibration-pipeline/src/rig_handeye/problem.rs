//! [`ProblemType`] implementation for multi-camera rig hand-eye calibration.
//!
//! This module provides the `RigHandeyeProblem` type that implements
//! the v2 session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::stated_covariance;
use vision_calibration_core::{
    BrownConrady5, CameraParams, FeatureResidualHistogram, FxFyCxCy, ImageManifest, Iso3,
    PerFeatureResiduals, Real, RigDataset, ScheimpflugParams, Uncertainty, build_feature_histogram,
    compute_rig_target_residuals,
};
use vision_calibration_optim::{
    HandeyeEstimate, HandeyeMode, RobotPoseMeta, handeye_observer_se3_target,
};
#[cfg(test)]
use vision_calibration_optim::{HandeyeParams, RobustLoss, SolveReport};

use crate::common::ExportKind;
use crate::common::config::{
    CameraConfig, CameraOverride, HandeyeInitConfig, IntrinsicsInitConfig, RigConfig,
    RobotPoseConfig, SolverConfig, rig_camera_for, validate_rig_camera_indices,
    validate_rig_cameras,
};
use crate::rig_family::rig_camera_models;
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

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
/// Grouped by stage. Every camera has the model of `camera` unless
/// `camera_overrides` names it, so one rig can mix lenses and sensors.
///
/// The reference-camera gauge fix (`rig.reference_camera_idx`) alone removes
/// the full 6-DOF rig gauge; an extra per-view pose constraint would be
/// redundant and mildly pessimizing (as for `RigExtrinsicsConfig`).
/// View-0's rig-from-target pose is always free in the rig BA.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigHandeyeConfig {
    /// Per-camera linear-initialization stage settings. An unset
    /// `fix_tangential` holds the tangential terms of each Scheimpflug camera
    /// and keeps them free for every other camera.
    pub intrinsics: IntrinsicsInitConfig,
    /// Optional manual seeds for the per-camera intrinsics stage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manual_init: Option<IntrinsicsManualInit>,
    /// The camera model of every rig camera and the parameters its solves
    /// hold fixed. A tilted (Scheimpflug) sensor's per-camera solve runs a
    /// multi-start tilt sweep; seed its tilt through `manual_init`.
    #[serde(default)]
    pub camera: CameraConfig,
    /// Cameras whose model differs from `camera`, at most one entry per
    /// camera index.
    #[serde(default)]
    pub camera_overrides: Vec<CameraOverride>,
    /// Rig frame options (reference camera, rig-BA scope).
    pub rig: RigConfig,
    /// Hand-eye linear initialization options.
    pub handeye_init: HandeyeInitConfig,
    /// Non-linear solve stage settings.
    pub solver: SolverConfig,
    /// Final hand-eye bundle-adjustment options.
    pub handeye_optimize: HandeyeOptimizeConfig,
}

impl RigHandeyeConfig {
    /// The configuration of camera `index`: its override, or `camera`.
    pub fn camera_for(&self, index: usize) -> CameraConfig {
        rig_camera_for(&self.camera, &self.camera_overrides, index)
    }
}

/// Manual seeds for the **per-camera intrinsics stage** of rig hand-eye
/// calibration. See `rig_extrinsics::IntrinsicsManualInit` for semantics.
///
/// `per_cam_sensors` seeds the tilt of each camera with a Scheimpflug sensor
/// (a seeded tilt skips the tilt sweep); a camera with an untilted sensor
/// takes only a zero tilt.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct IntrinsicsManualInit {
    /// Per-camera intrinsics seeds. `None` runs Zhang's method per camera.
    pub per_cam_intrinsics: Option<Vec<FxFyCxCy<Real>>>,
    /// Per-camera distortion seeds.
    pub per_cam_distortion: Option<Vec<BrownConrady5<Real>>>,
    /// Per-camera sensor tilt seeds (Scheimpflug cameras; zero for the
    /// others).
    #[serde(default)]
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
}

/// Final hand-eye bundle-adjustment options; robot-pose refinement lives in
/// the shared [`RobotPoseConfig`] group.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct HandeyeOptimizeConfig {
    /// Robot-pose refinement settings for the final hand-eye BA.
    pub robot_poses: RobotPoseConfig,
    /// Refine each camera's intrinsics and distortion in the hand-eye BA,
    /// holding the camera's `fix` lists. Default: `true`, for the reason
    /// given on `rig.refine_intrinsics`: the final solve's covariance then
    /// includes the intrinsics.
    pub refine_intrinsics: bool,
    /// Refine cam_se3_rig in hand-eye BA (default: false).
    /// When true, rig extrinsics are further refined during hand-eye optimization.
    pub refine_cam_se3_rig: bool,
    /// Refine the sensor tilt of each Scheimpflug camera in hand-eye BA,
    /// holding its `fix.sensor` list (default: false). Cameras with an
    /// untilted sensor have no tilt.
    #[serde(default)]
    pub refine_sensor: bool,
}

impl Default for HandeyeOptimizeConfig {
    fn default() -> Self {
        Self {
            robot_poses: RobotPoseConfig::default(),
            refine_intrinsics: true,
            refine_cam_se3_rig: false,
            refine_sensor: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// Export format for rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RigHandeyeExport {
    /// Export-type discriminator — always [`ExportKind::RigHandeye`].
    pub kind: ExportKind,

    /// Per-camera calibrated parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,

    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    /// Transform from rig frame to camera frame.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_rig: Vec<Iso3>,

    /// Per-view rig poses: `rig_se3_target` (T_R_T), derived from the
    /// hand-eye chain (`handeye_observer_se3_target`) so downstream
    /// viewers (3D scene, epipolar overlay) can read board poses
    /// without re-implementing the chain. One entry per input view.
    /// `#[serde(default)]` lets exports without this field load;
    /// they decode with an empty Vec.
    #[serde(default)]
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub rig_se3_target: Vec<Iso3>,

    /// Hand-eye mode used to interpret mode-dependent transforms.
    pub handeye_mode: HandeyeMode,

    /// Eye-in-hand: gripper-to-rig transform `gripper_se3_rig` (T_G_R).
    ///
    /// `None` for EyeToHand mode.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub gripper_se3_rig: Option<Iso3>,

    /// Eye-to-hand: rig-to-base transform `rig_se3_base` (T_R_B).
    ///
    /// `None` for EyeInHand mode.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub rig_se3_base: Option<Iso3>,

    /// Eye-in-hand: base-to-target transform `base_se3_target` (T_B_T).
    ///
    /// `None` for EyeToHand mode.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub base_se3_target: Option<Iso3>,

    /// Eye-to-hand: gripper-to-target transform `gripper_se3_target` (T_G_T).
    ///
    /// `None` for EyeInHand mode.
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Option<vision_calibration_core::Iso3Schema>")
    )]
    pub gripper_se3_target: Option<Iso3>,

    /// Per-view robot pose corrections (if refinement enabled).
    /// Each element is [rx, ry, rz, tx, ty, tz] in se(3).
    pub robot_deltas: Option<Vec<[f64; 6]>>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,

    /// Per-camera reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,

    /// Per-feature reprojection residuals. Per-view
    /// `rig_se3_target` is derived from the handeye chain
    /// (see [`handeye_observer_se3_target`](vision_calibration_optim::handeye_observer_se3_target)),
    /// then composed with `cam_se3_rig` for projection.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,

    /// Optional image manifest (viewer-side contract). When
    /// populated, downstream viewers (the diagnose UI) can locate the source
    /// image for each `(pose, camera)` slot. Tiled multi-camera frames
    /// (e.g. six 720×540 tiles in one horizontal strip) point
    /// multiple `FrameRef`s at the same `path` with disjoint ROIs. `None`
    /// means "no images shipped"; the calibration pipeline never reads
    /// this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
    /// Covariance of the calibrated parameters: `fx ± σ`, correlations, and
    /// the noise level the residuals imply.
    #[serde(default)]
    pub uncertainty: Uncertainty,
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
/// step_rig_init(&mut session, None)?;
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

impl ProblemState for RigHandeyeProblem {
    type State = RigHandeyeState;
}

impl ProblemType for RigHandeyeProblem {
    type Config = RigHandeyeConfig;
    type Input = RigHandeyeInput;
    type Output = HandeyeEstimate;
    type Export = RigHandeyeExport;

    fn name() -> &'static str {
        "rig_handeye_v2"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        stated_covariance(
            input
                .views
                .iter()
                .flat_map(|v| v.obs.cameras.iter().flatten()),
        )?;
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
        if config.intrinsics.iterations == 0 {
            return Err(Error::invalid_input(
                "intrinsics.iterations must be positive",
            ));
        }
        if config.handeye_init.min_motion_angle_deg <= 0.0 {
            return Err(Error::invalid_input(
                "min_motion_angle_deg must be positive",
            ));
        }
        config.handeye_optimize.robot_poses.validate()?;
        validate_rig_cameras(&config.camera, &config.camera_overrides)
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        if config.rig.reference_camera_idx >= input.num_cameras {
            return Err(Error::invalid_input(format!(
                "reference_camera_idx {} is out of range (num_cameras = {})",
                config.rig.reference_camera_idx, input.num_cameras
            )));
        }
        validate_rig_camera_indices(&config.camera_overrides, input.num_cameras)
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
        let params = &output.params;
        let cam_se3_rig: Vec<Iso3> = params.cam_to_rig.iter().map(|t| t.inverse()).collect();

        let target_pose = params
            .target_poses
            .first()
            .copied()
            .ok_or_else(|| Error::invalid_input("no target pose in output"))?;

        let mode = config.handeye_init.handeye_mode;
        let handeye = params.handeye;
        let (gripper_se3_rig, rig_se3_base, base_se3_target, gripper_se3_target) = match mode {
            HandeyeMode::EyeInHand => (Some(handeye), None, Some(target_pose), None),
            HandeyeMode::EyeToHand => (None, Some(handeye), None, Some(target_pose)),
        };

        // Per-feature residuals: derive rig_se3_target per view from the
        // handeye chain, then project through each camera's model.
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
            output.robot_deltas.as_deref(),
        );
        let target = compute_rig_target_residuals(
            &rig_camera_models(&params.cameras)?,
            input,
            &cam_se3_rig,
            &rig_se3_target,
        )?;
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

        let mut per_feature_residuals = PerFeatureResiduals::default();
        per_feature_residuals.target = target;
        per_feature_residuals.target_hist_per_camera = Some(target_hist_per_camera);
        Ok(RigHandeyeExport {
            kind: ExportKind::RigHandeye,
            cameras: params.cameras.clone(),
            cam_se3_rig,
            rig_se3_target: rig_se3_target.clone(),
            handeye_mode: mode,
            gripper_se3_rig,
            rig_se3_base,
            base_se3_target,
            gripper_se3_target,
            robot_deltas: output.robot_deltas.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
            per_feature_residuals,
            // Manifest is populated by callers that also wrote images for
            // the dataset (e.g. a tiled-rig runner); the pipeline
            // itself never has image paths to fill in.
            image_manifest: None,
            uncertainty: output.uncertainty.clone(),
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
            rig: RigConfig {
                reference_camera_idx: 5, // Out of range
                ..RigConfig::default()
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
            solver: SolverConfig {
                max_iters: 100,
                robust_loss: RobustLoss::Huber { scale: 2.5 },
                ..SolverConfig::default()
            },
            rig: RigConfig {
                reference_camera_idx: 1,
                refine_intrinsics: true,
                refine_sensor: false,
            },
            handeye_init: HandeyeInitConfig {
                handeye_mode: HandeyeMode::EyeToHand,
                ..HandeyeInitConfig::default()
            },
            handeye_optimize: HandeyeOptimizeConfig {
                robot_poses: RobotPoseConfig {
                    refine: false,
                    ..RobotPoseConfig::default()
                },
                ..HandeyeOptimizeConfig::default()
            },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigHandeyeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.solver.max_iters, 100);
        assert_eq!(restored.rig.reference_camera_idx, 1);
        assert!(restored.rig.refine_intrinsics);
        assert!(!restored.handeye_optimize.robot_poses.refine);
    }

    #[test]
    fn final_handeye_optimize_defaults_refine_intrinsics_and_keep_rig_fixed() {
        let config = RigHandeyeConfig::default();
        assert!(config.rig.refine_intrinsics);
        assert!(config.handeye_optimize.refine_intrinsics);
        assert!(!config.handeye_optimize.refine_cam_se3_rig);
        assert!(!config.handeye_optimize.refine_sensor);
        assert!(!config.rig.refine_sensor);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigHandeyeProblem::name(), "rig_handeye_v2");
        assert_eq!(RigHandeyeProblem::schema_version(), 1);
    }

    fn make_dummy_output() -> HandeyeEstimate {
        let camera = vision_calibration_core::make_pinhole_camera(
            vision_calibration_core::FxFyCxCy {
                fx: 800.0,
                fy: 800.0,
                cx: 640.0,
                cy: 360.0,
            },
            vision_calibration_core::BrownConrady5::default(),
        );

        HandeyeEstimate {
            params: HandeyeParams {
                cameras: vec![vision_calibration_core::pinhole_camera_params(&camera)],
                cam_to_rig: vec![Iso3::identity()],
                handeye: Iso3::identity(),
                target_poses: vec![Iso3::identity()],
            },
            uncertainty: Default::default(),
            report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            robot_deltas: None,
            mean_reproj_error: 0.0,
            per_cam_reproj_errors: vec![0.0],
        }
    }

    #[test]
    fn export_eye_in_hand_is_explicit() {
        let output = make_dummy_output();
        let config = RigHandeyeConfig {
            handeye_init: HandeyeInitConfig {
                handeye_mode: HandeyeMode::EyeInHand,
                ..HandeyeInitConfig::default()
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

        assert!(matches!(export.handeye_mode, HandeyeMode::EyeInHand));
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
            handeye_init: HandeyeInitConfig {
                handeye_mode: HandeyeMode::EyeToHand,
                ..HandeyeInitConfig::default()
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

        assert!(matches!(export.handeye_mode, HandeyeMode::EyeToHand));
        assert!(export.rig_se3_base.is_some());
        assert!(export.gripper_se3_target.is_some());
        assert!(export.gripper_se3_rig.is_none());
        assert!(export.base_se3_target.is_none());
    }

    #[test]
    fn export_image_manifest_defaults_absent_from_wire() {
        // ADR 0014: when the pipeline emits an export the manifest is
        // None and `skip_serializing_if` keeps existing serialized JSON byte-stable.
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
        // A tiled six-camera rig writes a single 4320×540 PNG per pose
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
        let frames: Vec<FrameRef> = (0..6)
            .map(|cam| {
                let mut roi = PixelRect::default();
                roi.x = (cam as u32) * 720;
                roi.w = 720;
                roi.h = 540;
                let mut frame = FrameRef::default();
                frame.pose = 0;
                frame.camera = cam;
                frame.path = std::path::PathBuf::from("target_0.png");
                frame.roi = Some(roi);
                frame
            })
            .collect();
        let mut manifest = ImageManifest::default();
        manifest.root = std::path::PathBuf::from(".");
        manifest.frames = frames;
        export.image_manifest = Some(manifest);

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
