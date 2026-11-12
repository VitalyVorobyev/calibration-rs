//! [`ProblemType`] implementation for multi-camera rig hand-eye calibration.
//!
//! This module provides the `RigHandeyeProblem` type that implements
//! the v2 session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::stated_covariance;
use vision_calibration_core::{CameraParams, Iso3, RigDataset, Uncertainty};
use vision_calibration_optim::{
    HandeyeEstimate, RobotPoseMeta, RobotRefinement, handeye_observer_se3_target,
};
#[cfg(test)]
use vision_calibration_optim::{HandeyeParams, RobustLoss, SolveReport};

use crate::common::config::{
    CameraConfig, CameraOverride, HandeyeInitConfig, IntrinsicsInitConfig, RigConfig,
    RobotPoseConfig, SolverConfig, rig_camera_for, validate_rig_camera_indices,
    validate_rig_cameras,
};
use crate::common::{ExportDiagnostics, ExportKind, HandeyeSolution};
use crate::rig_family::{
    IntrinsicsManualInit, RigStages, RigStagesState, inverse_poses, rig_target_residuals,
};
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
    /// Per-camera linear-initialization stage settings. The linear stage
    /// fits only the distortion terms each camera's solves estimate.
    pub intrinsics: IntrinsicsInitConfig,
    /// Optional manual seeds for the per-camera intrinsics stage. They seed
    /// `step_intrinsics_init_all` and `run_calibration` alike;
    /// `step_intrinsics_init_all_with_seed` takes an explicit seed that
    /// overrides this one.
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
    /// Refine the sensor tilt of each Scheimpflug camera in the hand-eye
    /// BA, holding its `fix.sensor` list. Default: `true`, for the reason
    /// given on `rig.refine_sensor`: the final solve's covariance then
    /// includes the tilts. Cameras with an untilted sensor have no tilt.
    #[serde(default = "crate::common::config::default_true")]
    pub refine_sensor: bool,
}

impl Default for HandeyeOptimizeConfig {
    fn default() -> Self {
        Self {
            robot_poses: RobotPoseConfig::default(),
            refine_intrinsics: true,
            refine_cam_se3_rig: false,
            refine_sensor: true,
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

    /// The hand-eye and the target reference pose of the calibration's mode.
    pub handeye: HandeyeSolution,

    /// The per-view robot-pose corrections and the pose error they were
    /// weighed with: the configured `robot_poses` prior, or its estimate with
    /// `estimate_sigma`. Only present if robot refinement was enabled.
    #[serde(default)]
    pub robot_refinement: Option<RobotRefinement>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,

    /// Per-camera reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Residuals and the image manifest: evidence for viewers and
    /// debugging, not part of the result.
    #[serde(default)]
    pub diagnostics: ExportDiagnostics,
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
/// - EyeInHand export (`export.handeye`): `gripper_se3_rig` (T_G_R), `base_se3_target` (T_B_T)
/// - EyeToHand export (`export.handeye`): `rig_se3_base` (T_R_B), `gripper_se3_target` (T_G_T)
/// - Reference camera has identity extrinsics (defines rig frame)
///
/// # Example
///
/// A two-camera rig on the gripper views a static board from eight robot
/// stations; every rig frame carries the robot's `base_se3_gripper`:
///
/// ```
/// use vision_calibration_core::{RigView, RigViewObs};
/// use vision_calibration_pipeline::common::{HandeyeSolution, RobotPoseMeta};
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::rig_handeye::{
///     RigHandeyeInput, RigHandeyeProblem, step_handeye_init, step_handeye_optimize,
///     step_intrinsics_init_all, step_intrinsics_optimize_all, step_rig_init,
///     step_rig_optimize,
/// };
/// # use vision_calibration_core::synthetic::{planar, poses};
/// # use vision_calibration_core::{BrownConrady5, FxFyCxCy, Iso3, make_pinhole_camera};
/// # fn main() -> anyhow::Result<()> {
/// # let camera = make_pinhole_camera(
/// #     FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
/// #     BrownConrady5::default(),
/// # );
/// # let cam_se3_rig = poses::rig_layout(2, 0.1, 0.05);
/// # let gripper_se3_rig = poses::make_iso((0.1, -0.05, 0.02), (0.05, -0.03, 0.1));
/// # let base_se3_target = poses::make_iso((0.0, 0.0, 0.0), (-0.1, -0.1, 1.0));
/// # let board = planar::grid_points(8, 6, 0.04);
/// # let stations = poses::robot_stations(8, 7);
/// # let observe = |cam: usize, base_se3_gripper: &Iso3| {
/// #     let rig_se3_target = (base_se3_gripper * gripper_se3_rig).inverse() * base_se3_target;
/// #     planar::project_view_all(&camera, &(cam_se3_rig[cam] * rig_se3_target), &board).map(Some)
/// # };
///
/// let mut views = Vec::new();
/// for base_se3_gripper in stations {
///     let cameras = vec![observe(0, &base_se3_gripper)?, observe(1, &base_se3_gripper)?];
///     let meta = RobotPoseMeta { base_se3_gripper };
///     views.push(RigView { obs: RigViewObs { cameras }, meta });
/// }
/// let mut session = CalibrationSession::<RigHandeyeProblem>::new();
/// session.set_input(RigHandeyeInput::new(views, 2)?)?;
///
/// step_intrinsics_init_all(&mut session, None)?;
/// step_intrinsics_optimize_all(&mut session, None)?;
/// step_rig_init(&mut session, None)?;
/// step_rig_optimize(&mut session, None)?;
/// step_handeye_init(&mut session, None)?;
/// step_handeye_optimize(&mut session, None)?;
///
/// let export = session.export()?;
/// let HandeyeSolution::EyeInHand { gripper_se3_rig: estimate, .. } = export.handeye else {
///     panic!("an eye-in-hand export");
/// };
/// assert!((estimate.translation.vector - gripper_se3_rig.translation.vector).norm() < 1e-4);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RigHandeyeProblem;

impl ProblemState for RigHandeyeProblem {
    type State = RigHandeyeState;
}

impl RigStages for RigHandeyeProblem {
    type Meta = RobotPoseMeta;

    fn dataset(input: &RigHandeyeInput) -> &RigDataset<RobotPoseMeta> {
        input
    }
    fn camera_for(config: &RigHandeyeConfig, index: usize) -> CameraConfig {
        config.camera_for(index)
    }
    fn intrinsics_config(config: &RigHandeyeConfig) -> &IntrinsicsInitConfig {
        &config.intrinsics
    }
    fn rig_config(config: &RigHandeyeConfig) -> &RigConfig {
        &config.rig
    }
    fn solver_config(config: &RigHandeyeConfig) -> &SolverConfig {
        &config.solver
    }
    fn stages(state: &RigHandeyeState) -> &RigStagesState {
        &state.rig
    }
    fn stages_mut(state: &mut RigHandeyeState) -> &mut RigStagesState {
        &mut state.rig
    }
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
        2
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

        // One camera is a rig too: the joint laser workflow's warm start
        // calibrates a single laser head as one (`rig_handeye_laserline`).
        if input.num_cameras == 0 {
            return Err(Error::InsufficientData { need: 1, got: 0 });
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
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        let params = &output.params;
        let cam_se3_rig = inverse_poses(&params.rig_se3_cam);

        let target_pose = params
            .target_poses
            .first()
            .copied()
            .ok_or_else(|| Error::invalid_input("no target pose in output"))?;

        let mode = output.mode;
        let handeye = params.handeye;
        let solution = HandeyeSolution::new(mode, handeye, target_pose);

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
            output
                .robot_refinement
                .as_ref()
                .map(|r| r.deltas.as_slice()),
        );
        let per_feature_residuals =
            rig_target_residuals(&params.cameras, input, &cam_se3_rig, &rig_se3_target)?;
        Ok(RigHandeyeExport {
            kind: ExportKind::RigHandeye,
            cameras: params.cameras.clone(),
            cam_se3_rig,
            rig_se3_target: rig_se3_target.clone(),
            handeye: solution,
            robot_refinement: output.robot_refinement.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
            // Callers that have the images fill `image_manifest`; the
            // pipeline never has image paths.
            diagnostics: ExportDiagnostics::new(per_feature_residuals),
            uncertainty: output.uncertainty.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{CorrespondenceView, Pt2, Pt3, RigDataset, RigView, RigViewObs};
    use vision_calibration_optim::HandeyeMode;

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
    fn validate_input_accepts_one_camera() {
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
        assert!(RigHandeyeProblem::validate_input(&input).is_ok());
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
                robot_poses: RobotPoseConfig::Exact,
                ..HandeyeOptimizeConfig::default()
            },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: RigHandeyeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.solver.max_iters, 100);
        assert_eq!(restored.rig.reference_camera_idx, 1);
        assert!(restored.rig.refine_intrinsics);
        assert!(matches!(
            restored.handeye_optimize.robot_poses,
            RobotPoseConfig::Exact
        ));
    }

    #[test]
    fn final_handeye_optimize_defaults_refine_cameras_and_keep_rig_fixed() {
        let config = RigHandeyeConfig::default();
        assert!(config.rig.refine_intrinsics);
        assert!(config.handeye_optimize.refine_intrinsics);
        assert!(!config.handeye_optimize.refine_cam_se3_rig);
        assert!(config.handeye_optimize.refine_sensor);
        assert!(config.rig.refine_sensor);
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(RigHandeyeProblem::name(), "rig_handeye_v2");
        assert_eq!(RigHandeyeProblem::schema_version(), 2);
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
            mode: HandeyeMode::EyeInHand,
            params: HandeyeParams {
                cameras: vec![vision_calibration_core::pinhole_camera_params(&camera)],
                rig_se3_cam: vec![Iso3::identity()],
                handeye: Iso3::identity(),
                target_poses: vec![Iso3::identity()],
            },
            uncertainty: Default::default(),
            report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            robot_refinement: None,
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

        assert!(matches!(export.handeye, HandeyeSolution::EyeInHand { .. }));
        assert_solution_round_trips(&export);
        assert_eq!(
            export.rig_se3_target.len(),
            1,
            "rig_se3_target must have one entry per input view"
        );
    }

    #[test]
    fn export_eye_to_hand_is_explicit() {
        let mut output = make_dummy_output();
        output.mode = HandeyeMode::EyeToHand;
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

        assert!(matches!(export.handeye, HandeyeSolution::EyeToHand { .. }));
        assert_solution_round_trips(&export);
    }

    /// The export's JSON holds the mode's two poses under `handeye`, tagged
    /// by `mode`, and reads back equal; the shape from before the tag (a
    /// `handeye_mode` and four optional poses) is refused.
    fn assert_solution_round_trips(export: &RigHandeyeExport) {
        let json = serde_json::to_value(export).unwrap();
        let wire = &json["handeye"];
        assert_eq!(
            wire["mode"],
            match export.handeye.mode() {
                HandeyeMode::EyeInHand => "eye_in_hand",
                HandeyeMode::EyeToHand => "eye_to_hand",
            }
        );
        assert_eq!(wire.as_object().unwrap().len(), 3, "{wire}");
        assert!(json.get("handeye_mode").is_none());
        let back: RigHandeyeExport = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(back.handeye, export.handeye);

        let mut old = json;
        let object = old.as_object_mut().unwrap();
        object.remove("handeye");
        object.insert("handeye_mode".into(), "eye_in_hand".into());
        object.insert(
            "gripper_se3_rig".into(),
            serde_json::to_value(Iso3::identity()).unwrap(),
        );
        let err = serde_json::from_value::<RigHandeyeExport>(old).unwrap_err();
        assert!(err.to_string().contains("missing field `handeye`"), "{err}");
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
        assert!(export.diagnostics.image_manifest.is_none());

        let json = serde_json::to_string(&export).unwrap();
        assert!(
            !json.contains("image_manifest"),
            "absent manifest must not appear on the wire"
        );
        let restored: RigHandeyeExport = serde_json::from_str(&json).unwrap();
        assert!(restored.diagnostics.image_manifest.is_none());
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
        use vision_calibration_core::{FrameRef, PixelRect};

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
                frame.view = 0;
                frame.camera = cam;
                frame.path = std::path::PathBuf::from("target_0.png");
                frame.roi = Some(roi);
                frame
            })
            .collect();
        let mut manifest = vision_calibration_core::ImageManifest::default();
        manifest.root = std::path::PathBuf::from(".");
        manifest.frames = frames;
        export.diagnostics.image_manifest = Some(manifest);

        let json = serde_json::to_string(&export).unwrap();
        let restored: RigHandeyeExport = serde_json::from_str(&json).unwrap();
        let manifest = restored
            .diagnostics
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.frames.len(), 6);
        assert_eq!(manifest.frames[0].camera, 0);
        assert_eq!(manifest.frames[5].roi.unwrap().x, 5 * 720);
    }
}
