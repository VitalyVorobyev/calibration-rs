//! [`ProblemType`] implementation for single-camera hand-eye calibration.
//!
//! This module provides the `SingleCamHandeyeProblem` type that implements
//! the session API's `ProblemType` trait.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::stated_covariance;
use vision_calibration_core::{
    CameraParams, Handeye, Iso3, RigView, RigViewObs, Uncertainty, View,
    compute_planar_target_residuals_views,
};
use vision_calibration_optim::{
    HandeyeDataset, HandeyeEstimate, HandeyeMode, HandeyeSolveOptions, RobotPoseMeta,
    RobotRefinement, RobustLoss, handeye_observer_se3_target,
};

use crate::common::config::{
    CameraConfig, HandeyeInitConfig, IntrinsicsInitConfig, IntrinsicsOptimizeConfig,
    RobotPoseConfig, SolverConfig,
};
use crate::common::{
    ExportDiagnostics, ProblemKind, VersionedExport, one_camera_target_residuals,
    read_schema_version,
};
use crate::planar_family::check_camera_model;
use crate::rig_family::{HandeyePoses, HandeyeStages, handeye_solve_opts};
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::SingleCamHandeyeState;

// ─────────────────────────────────────────────────────────────────────────────
// Input Types
// ─────────────────────────────────────────────────────────────────────────────

/// A single view with robot pose and 2D-3D correspondences.
pub type SingleCamHandeyeView = View<RobotPoseMeta>;

/// Input for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct SingleCamHandeyeInput {
    /// Per-view observations with robot poses.
    pub views: Vec<SingleCamHandeyeView>,
}

impl SingleCamHandeyeInput {
    /// Create a new input from views.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if `views` is empty, or
    /// [`Error::InvalidInput`] if any view has fewer than 4 observations.
    pub fn new(views: Vec<SingleCamHandeyeView>) -> Result<Self, Error> {
        if views.is_empty() {
            return Err(Error::InsufficientData { need: 1, got: 0 });
        }
        for (i, view) in views.iter().enumerate() {
            if view.obs.len() < 4 {
                return Err(Error::invalid_input(format!(
                    "view {} has too few points (need >= 4, got {})",
                    i,
                    view.obs.len()
                )));
            }
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
///
/// Grouped by stage, in the order the steps run: the camera model, the
/// intrinsics linear init and refinement, the hand-eye linear init and
/// bundle adjustment, and the solver engine. Every group's own `Default` is
/// the plain workspace default (no per-problem override is needed here,
/// unlike `PlanarIntrinsicsConfig`'s iteration cap).
///
/// Two stages solve the target corners, each with its own
/// `reprojection_loss`: `intrinsics_optimize` and `handeye_optimize`. A
/// robust calibration sets both; the final stage's loss is the estimator.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct SingleCamHandeyeConfig {
    /// The camera model and the parameters its solves (the intrinsics stage
    /// and the hand-eye bundle adjustment) hold fixed. Any distortion kind
    /// with an identity sensor; a Scheimpflug sensor needs a distortion
    /// model. A tilted sensor's intrinsics stage runs a multi-start tilt
    /// sweep; seed its tilt through the intrinsics manual init.
    #[serde(default)]
    pub camera: CameraConfig,

    /// Linear-initialization stage settings (`step_intrinsics_init`). The
    /// linear stage fits only the distortion terms the camera's solves
    /// estimate.
    pub intrinsics_init: IntrinsicsInitConfig,

    /// Intrinsics refinement stage settings (`step_intrinsics_optimize`).
    pub intrinsics_optimize: IntrinsicsOptimizeConfig,

    /// Hand-eye linear-initialization stage settings (`step_handeye_init`).
    pub handeye_init: HandeyeInitConfig,

    /// Hand-eye bundle-adjustment stage settings (`step_handeye_optimize`).
    pub handeye_optimize: SingleCamHandeyeOptimizeConfig,

    /// Solver engine settings, for both non-linear stages.
    pub solver: SolverConfig,
}

/// The hand-eye bundle adjustment of a single-camera hand-eye calibration
/// (`step_handeye_optimize`): the camera, the hand-eye transform, the
/// target pose and, by default, the robot poses refined together, the
/// camera's `fix` lists held.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct SingleCamHandeyeOptimizeConfig {
    /// Robust loss on the target-corner reprojection residuals; its scale is
    /// in pixels. Default `none`.
    pub reprojection_loss: RobustLoss,
    /// Robot-pose refinement.
    pub robot_poses: RobotPoseConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Export
// ─────────────────────────────────────────────────────────────────────────────

/// The solution of a single-camera hand-eye calibration: the hand-eye of its
/// mode and the pose of the static target in the frame that mode fixes it in.
///
/// Each mode holds exactly the two poses it estimates, so an export with the
/// wrong pose for its mode cannot be written or read. It is the rig
/// solution ([`HandeyeSolution`](crate::common::HandeyeSolution)) of a rig
/// that is the camera, with the camera named where the rig is. In JSON it is
/// internally tagged by `mode`:
///
/// ```json
/// {"mode": "eye_in_hand", "gripper_se3_cam": {…}, "base_se3_target": {…}}
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SingleCamHandeyeSolution {
    /// The camera rides on the gripper; the target stands in the base frame.
    EyeInHand {
        /// The camera's pose on the gripper (`gripper_se3_cam`, camera
        /// frame → gripper frame).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        gripper_se3_cam: Iso3,
        /// The target's pose in the robot's base frame (`base_se3_target`).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        base_se3_target: Iso3,
    },
    /// The camera stands in the base frame; the target rides on the gripper.
    EyeToHand {
        /// The base's pose in the camera (`cam_se3_base`, base frame →
        /// camera frame).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        cam_se3_base: Iso3,
        /// The target's pose on the gripper (`gripper_se3_target`).
        #[cfg_attr(
            feature = "schemars",
            schemars(with = "vision_calibration_core::Iso3Schema")
        )]
        gripper_se3_target: Iso3,
    },
}

impl SingleCamHandeyeSolution {
    /// The solution of `mode` from its hand-eye pose (`gripper_se3_cam`
    /// eye-in-hand, `cam_se3_base` eye-to-hand) and its target pose
    /// (`base_se3_target`, `gripper_se3_target`).
    pub fn new(mode: HandeyeMode, handeye: Iso3, target: Iso3) -> Self {
        match mode {
            HandeyeMode::EyeInHand => Self::EyeInHand {
                gripper_se3_cam: handeye,
                base_se3_target: target,
            },
            HandeyeMode::EyeToHand => Self::EyeToHand {
                cam_se3_base: handeye,
                gripper_se3_target: target,
            },
        }
    }

    /// The mode the solution is in.
    pub fn mode(&self) -> HandeyeMode {
        match self {
            Self::EyeInHand { .. } => HandeyeMode::EyeInHand,
            Self::EyeToHand { .. } => HandeyeMode::EyeToHand,
        }
    }

    /// The camera's hand-eye, as a rig's: the camera is the rig.
    pub fn handeye(&self) -> Handeye {
        match *self {
            Self::EyeInHand {
                gripper_se3_cam, ..
            } => Handeye::EyeInHand {
                gripper_se3_rig: gripper_se3_cam,
            },
            Self::EyeToHand { cam_se3_base, .. } => Handeye::EyeToHand {
                rig_se3_base: cam_se3_base,
            },
        }
    }

    /// The target's pose of the mode: `base_se3_target` eye-in-hand,
    /// `gripper_se3_target` eye-to-hand.
    pub fn target_pose(&self) -> &Iso3 {
        match self {
            Self::EyeInHand {
                base_se3_target, ..
            } => base_se3_target,
            Self::EyeToHand {
                gripper_se3_target, ..
            } => gripper_se3_target,
        }
    }
}

/// Export format for single-camera hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct SingleCamHandeyeExport {
    /// Export-type discriminator — always [`ProblemKind::SingleCamHandeye`].
    pub kind: ProblemKind,

    /// Format version of this export, [`VersionedExport::SCHEMA_VERSION`].
    /// Reading an export of another version is an error.
    #[serde(deserialize_with = "read_schema_version::<_, SingleCamHandeyeExport>")]
    pub schema_version: u32,

    /// Calibrated camera, in the configured model.
    pub camera: CameraParams,

    /// The hand-eye and the target pose of the calibration's mode.
    pub handeye: SingleCamHandeyeSolution,

    /// The per-view robot-pose corrections and the pose error they were
    /// weighed with: the configured `handeye_optimize.robot_poses` prior, or its estimate with
    /// `estimate_sigma`. Only present if robot refinement was enabled.
    #[serde(default)]
    pub robot_refinement: Option<RobotRefinement>,

    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,

    /// Per-camera reprojection errors (pixels). Single element for single-camera.
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

impl VersionedExport for SingleCamHandeyeExport {
    const SCHEMA_VERSION: u32 = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
// ProblemType Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Single-camera hand-eye calibration problem.
///
/// Calibrates a single camera mounted on a robot arm, including:
/// - The camera, in the configured model (intrinsics, distortion, sensor)
/// - Hand-eye transform (gripper-to-camera or camera-to-base)
/// - Target pose in base frame
///
/// # Example
///
/// A camera on the gripper views a static board from eight robot stations.
/// Each view pairs the board observation with the robot's
/// `base_se3_gripper`:
///
/// ```
/// use vision_calibration_pipeline::common::RobotPoseMeta;
/// use vision_calibration_pipeline::session::CalibrationSession;
/// use vision_calibration_pipeline::single_cam_handeye::{
///     SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeSolution,
///     SingleCamHandeyeView,
///     step_handeye_init, step_handeye_optimize, step_intrinsics_init,
///     step_intrinsics_optimize,
/// };
/// # use vision_calibration_core::synthetic::{planar, poses};
/// # use vision_calibration_core::{BrownConrady5, FxFyCxCy, make_pinhole_camera};
/// # fn main() -> anyhow::Result<()> {
/// # let camera = make_pinhole_camera(
/// #     FxFyCxCy { fx: 800.0, fy: 780.0, cx: 640.0, cy: 360.0 },
/// #     BrownConrady5::default(),
/// # );
/// # let gripper_se3_cam = poses::make_iso((0.1, -0.05, 0.02), (0.05, -0.03, 0.1));
/// # let base_se3_target = poses::make_iso((0.0, 0.0, 0.0), (-0.1, -0.1, 1.0));
/// # let board = planar::grid_points(6, 5, 0.05);
/// # let stations = poses::robot_stations(8, 7);
/// # let observe = |base_se3_gripper: &vision_calibration_core::Iso3| {
/// #     let cam_se3_target = (base_se3_gripper * gripper_se3_cam).inverse() * base_se3_target;
/// #     planar::project_view_all(&camera, &cam_se3_target, &board)
/// # };
///
/// let mut views = Vec::new();
/// for base_se3_gripper in stations {
///     let board_view = observe(&base_se3_gripper)?;
///     views.push(SingleCamHandeyeView::new(board_view, RobotPoseMeta { base_se3_gripper }));
/// }
/// let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
/// session.set_input(SingleCamHandeyeInput::new(views)?)?;
///
/// step_intrinsics_init(&mut session, None)?;
/// step_intrinsics_optimize(&mut session, None)?;
/// step_handeye_init(&mut session, None)?; // closed-form (Tsai-Lenz by default)
/// step_handeye_optimize(&mut session, None)?; // joint bundle adjustment
///
/// let export = session.export()?;
/// let SingleCamHandeyeSolution::EyeInHand { gripper_se3_cam: estimate, .. } = export.handeye else {
///     panic!("an eye-in-hand export");
/// };
/// assert!((estimate.translation.vector - gripper_se3_cam.translation.vector).norm() < 1e-4);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SingleCamHandeyeProblem;

impl ProblemState for SingleCamHandeyeProblem {
    type State = SingleCamHandeyeState;
}

impl HandeyePoses for SingleCamHandeyeSolution {
    fn from_poses(mode: HandeyeMode, handeye: Iso3, target: Iso3) -> Self {
        Self::new(mode, handeye, target)
    }
    fn mode(&self) -> HandeyeMode {
        SingleCamHandeyeSolution::mode(self)
    }
    fn poses(&self) -> (Iso3, Iso3) {
        (*self.handeye().pose(), *self.target_pose())
    }
    fn pose_names(mode: HandeyeMode) -> (&'static str, &'static str) {
        match mode {
            HandeyeMode::EyeInHand => ("gripper_se3_cam", "base_se3_target"),
            HandeyeMode::EyeToHand => ("cam_se3_base", "gripper_se3_target"),
        }
    }
}

/// The hand-eye stages of a rig that is the camera: one camera, at the
/// identity in the rig frame.
impl HandeyeStages for SingleCamHandeyeProblem {
    type Solution = SingleCamHandeyeSolution;

    fn handeye_init_config(config: &SingleCamHandeyeConfig) -> &HandeyeInitConfig {
        &config.handeye_init
    }
    fn solver_config(config: &SingleCamHandeyeConfig) -> &SolverConfig {
        &config.solver
    }
    fn base_se3_gripper(input: &SingleCamHandeyeInput) -> Vec<Iso3> {
        input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper)
            .collect()
    }
    fn observer_se3_target(state: &SingleCamHandeyeState) -> Result<&[Iso3], Error> {
        if !state.has_intrinsics_optimized() {
            return Err(Error::not_available(
                "intrinsics optimization (call step_intrinsics_optimize first)",
            ));
        }
        state
            .optimized_cam_se3_target
            .as_deref()
            .ok_or_else(|| Error::not_available("optimized target poses"))
    }
    fn initial_handeye(state: &SingleCamHandeyeState) -> Option<SingleCamHandeyeSolution> {
        state.initial_handeye
    }
    fn initial_handeye_mut(
        state: &mut SingleCamHandeyeState,
    ) -> &mut Option<SingleCamHandeyeSolution> {
        &mut state.initial_handeye
    }
    fn handeye_cameras(
        config: &SingleCamHandeyeConfig,
        state: &SingleCamHandeyeState,
    ) -> Result<Vec<CameraParams>, Error> {
        let camera = state
            .optimized_camera
            .clone()
            .ok_or_else(|| Error::not_available("optimized camera"))?;
        check_camera_model(&camera, config.camera.model, "step_intrinsics_init")?;
        Ok(vec![camera])
    }
    fn rig_se3_cam(_state: &SingleCamHandeyeState) -> Result<Vec<Iso3>, Error> {
        Ok(vec![Iso3::identity()])
    }
    fn handeye_dataset(
        input: &SingleCamHandeyeInput,
        mode: HandeyeMode,
    ) -> Result<HandeyeDataset, Error> {
        let views = input
            .views
            .iter()
            .map(|v| RigView {
                meta: RobotPoseMeta {
                    base_se3_gripper: v.meta.base_se3_gripper,
                },
                obs: RigViewObs {
                    cameras: vec![Some(v.obs.clone())],
                },
            })
            .collect();
        Ok(HandeyeDataset::new(views, 1, mode)?)
    }
    fn handeye_solve_options(
        config: &SingleCamHandeyeConfig,
        _num_cameras: usize,
    ) -> HandeyeSolveOptions {
        // The camera holds its `fix` lists; its extrinsics stay the identity.
        handeye_solve_opts(
            config.handeye_optimize.reprojection_loss,
            vec![config.camera.fix.clone()],
            vec![true],
            &config.handeye_optimize.robot_poses,
        )
    }
    fn record_handeye_optimize(
        state: &mut SingleCamHandeyeState,
        final_cost: f64,
        mean_reproj_error: f64,
    ) {
        state.handeye_final_cost = Some(final_cost);
        state.handeye_reproj_error = Some(mean_reproj_error);
    }
}

impl ProblemType for SingleCamHandeyeProblem {
    type Config = SingleCamHandeyeConfig;
    type Input = SingleCamHandeyeInput;
    type Output = HandeyeEstimate;
    type Export = SingleCamHandeyeExport;

    fn name() -> &'static str {
        "single_cam_handeye_v2"
    }

    fn schema_version() -> u32 {
        4
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        stated_covariance(input.views.iter().map(|v| &v.obs))?;
        if input.num_views() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: input.num_views(),
            });
        }

        for (i, view) in input.views.iter().enumerate() {
            if view.obs.len() < 4 {
                return Err(Error::invalid_input(format!(
                    "view {} has too few points (need >= 4 for homography, got {})",
                    i,
                    view.obs.len()
                )));
            }
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.solver.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.intrinsics_init.iterations == 0 {
            return Err(Error::invalid_input(
                "intrinsics_init.iterations must be positive",
            ));
        }
        if config.handeye_init.min_motion_angle_deg <= 0.0 {
            return Err(Error::invalid_input(
                "min_motion_angle_deg must be positive",
            ));
        }
        config.handeye_optimize.robot_poses.validate()?;
        config.camera.validate_single_camera()
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
        // Extract the single camera (index 0 for single-cam setup).
        let camera = output
            .params
            .cameras
            .first()
            .cloned()
            .ok_or_else(|| Error::invalid_input("no camera in output"))?;
        let model = camera
            .build()
            .map_err(|e| Error::invalid_input(format!("camera model: {e}")))?;

        // Target pose (single fixed target pose; interpretation depends on mode)
        let target_pose = output
            .params
            .target_poses
            .first()
            .cloned()
            .ok_or_else(|| Error::invalid_input("no target pose in output"))?;

        let solution =
            SingleCamHandeyeSolution::new(output.mode, output.params.handeye, target_pose);

        // Per-feature residuals: derive cam_se3_target per view from the
        // handeye chain (camera == observer for single-cam) and project.
        let robot_poses: Vec<Iso3> = input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper)
            .collect();
        let cam_se3_target = handeye_observer_se3_target(
            output.mode,
            &output.params.handeye,
            &target_pose,
            &robot_poses,
            output
                .robot_refinement
                .as_ref()
                .map(|r| r.deltas.as_slice()),
        );
        let target = compute_planar_target_residuals_views(&model, &input.views, &cam_se3_target)?;
        let per_feature_residuals = one_camera_target_residuals(target);
        Ok(SingleCamHandeyeExport {
            kind: ProblemKind::SingleCamHandeye,
            schema_version: SingleCamHandeyeExport::SCHEMA_VERSION,
            camera,
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
    use vision_calibration_core::{CorrespondenceView, Pt2, Pt3};
    use vision_calibration_core::{DistortionKind, SensorKind};
    use vision_calibration_optim::RobustLoss;

    fn make_minimal_view() -> SingleCamHandeyeView {
        View::new(
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
            .unwrap(),
            RobotPoseMeta {
                base_se3_gripper: Iso3::identity(),
            },
        )
    }

    fn make_minimal_input() -> SingleCamHandeyeInput {
        SingleCamHandeyeInput {
            views: vec![
                make_minimal_view(),
                make_minimal_view(),
                make_minimal_view(),
            ],
        }
    }

    #[test]
    fn validate_input_requires_3_views() {
        let input = make_minimal_input();
        let result = SingleCamHandeyeProblem::validate_input(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_input_rejects_too_few_views() {
        let input = SingleCamHandeyeInput {
            views: vec![make_minimal_view(), make_minimal_view()],
        };
        let result = SingleCamHandeyeProblem::validate_input(&input);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("need 3"));
    }

    #[test]
    fn validate_config_requires_positive_iters() {
        let config = SingleCamHandeyeConfig {
            solver: SolverConfig {
                max_iters: 0,
                ..Default::default()
            },
            ..Default::default()
        };
        let result = SingleCamHandeyeProblem::validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("max_iters"));
    }

    #[test]
    fn validate_config_accepts_valid() {
        let config = SingleCamHandeyeConfig::default();
        let result = SingleCamHandeyeProblem::validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_config_checks_the_camera() {
        let tilted_without_distortion = SingleCamHandeyeConfig {
            camera: CameraConfig::new(vision_calibration_core::CameraModelSpec::pinhole(
                DistortionKind::None,
                SensorKind::Scheimpflug,
            )),
            ..Default::default()
        };
        let err = SingleCamHandeyeProblem::validate_config(&tilted_without_distortion)
            .unwrap_err()
            .to_string();
        assert!(err.contains("distortion model"), "{err}");

        let mut bad_fix = SingleCamHandeyeConfig::default();
        bad_fix.camera.fix.sensor = Some(vec!["tilt_x".into()]);
        let err = SingleCamHandeyeProblem::validate_config(&bad_fix)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tilt_x"), "{err}");
    }

    #[test]
    fn config_json_roundtrip() {
        let config = SingleCamHandeyeConfig {
            solver: SolverConfig {
                max_iters: 100,
                ..Default::default()
            },
            intrinsics_init: IntrinsicsInitConfig {
                fix_k3: false,
                ..Default::default()
            },
            intrinsics_optimize: IntrinsicsOptimizeConfig {
                reprojection_loss: RobustLoss::Huber { scale: 2.5 },
            },
            handeye_init: HandeyeInitConfig {
                handeye_mode: HandeyeMode::EyeToHand,
                ..Default::default()
            },
            ..Default::default()
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: SingleCamHandeyeConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.solver.max_iters, 100);
        assert!(!restored.intrinsics_init.fix_k3);
        assert_eq!(
            restored.intrinsics_optimize.reprojection_loss,
            RobustLoss::Huber { scale: 2.5 }
        );
        assert_eq!(
            restored.handeye_optimize.reprojection_loss,
            RobustLoss::None
        );
        assert!(matches!(
            restored.handeye_init.handeye_mode,
            HandeyeMode::EyeToHand
        ));
    }

    #[test]
    fn solution_wire_form_is_tagged_by_mode() {
        let (handeye, target) = (
            Iso3::translation(0.1, 0.0, 0.2),
            Iso3::translation(0.0, 0.5, 1.0),
        );
        for (mode, keys) in [
            (
                HandeyeMode::EyeInHand,
                ["gripper_se3_cam", "base_se3_target"],
            ),
            (
                HandeyeMode::EyeToHand,
                ["cam_se3_base", "gripper_se3_target"],
            ),
        ] {
            let solution = SingleCamHandeyeSolution::new(mode, handeye, target);
            assert_eq!(solution.mode(), mode);
            assert_eq!(solution.target_pose(), &target);
            assert_eq!(solution.handeye().pose(), &handeye);
            let json = serde_json::to_value(solution).unwrap();
            assert_eq!(json["mode"], serde_json::to_value(mode).unwrap());
            assert_eq!(json.as_object().unwrap().len(), 3, "{json}");
            for key in keys {
                assert!(json.get(key).is_some(), "{key} in {json}");
            }
            let back: SingleCamHandeyeSolution = serde_json::from_value(json.clone()).unwrap();
            assert_eq!(back, solution);

            // The other mode's pose is not the mode's: refused, not guessed.
            let mut wrong = json;
            wrong["mode"] = match mode {
                HandeyeMode::EyeInHand => "eye_to_hand".into(),
                HandeyeMode::EyeToHand => "eye_in_hand".into(),
            };
            let err = serde_json::from_value::<SingleCamHandeyeSolution>(wrong).unwrap_err();
            assert!(err.to_string().contains("missing field"), "{err}");
        }
    }

    #[test]
    fn problem_name_and_version() {
        assert_eq!(SingleCamHandeyeProblem::name(), "single_cam_handeye_v2");
        assert_eq!(SingleCamHandeyeProblem::schema_version(), 4);
    }
}
