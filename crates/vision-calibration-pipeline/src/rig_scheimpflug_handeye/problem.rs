//! [`ProblemType`] implementation for Scheimpflug rig hand-eye calibration (EyeInHand only).

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{Iso3, PinholeCamera, RigDataset, ScheimpflugParams};
use vision_calibration_optim::{
    HandEyeScheimpflugEstimate, RobotPoseMeta, RobustLoss, ScheimpflugFixMask,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigScheimpflugHandeyeState;

/// Input: rig dataset with per-view robot pose metadata.
pub type RigScheimpflugHandeyeInput = RigDataset<RobotPoseMeta>;

/// Per-camera intrinsics config.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeIntrinsicsConfig {
    /// Number of iterations for iterative intrinsics estimation.
    pub init_iterations: usize,
    /// Fix k3 during intrinsics calibration.
    pub fix_k3: bool,
    /// Fix tangential distortion (p1, p2).
    pub fix_tangential: bool,
    /// Enforce zero skew.
    pub zero_skew: bool,
    /// Initial Scheimpflug tilt_x (radians).
    pub init_tilt_x: f64,
    /// Initial Scheimpflug tilt_y (radians).
    pub init_tilt_y: f64,
    /// Scheimpflug mask during per-camera refinement.
    pub fix_scheimpflug: ScheimpflugFixMask,

    /// Optional per-camera initial intrinsics that bypass Zhang's method.
    ///
    /// If `Some`, its length must equal `num_cameras`. Each `Some` entry
    /// overrides the linear init for that camera; `None` entries fall back to
    /// Zhang's method (with `fallback_to_shared_init` applying if Zhang fails).
    ///
    /// Useful when a single homogeneous rig shares the same optical design
    /// across all cameras — a known focal-length / principal-point prior lets
    /// the non-linear refinement step do all the heavy lifting and sidesteps
    /// Zhang's sensitivity to borderline view geometry.
    pub initial_cameras: Option<Vec<PinholeCamera>>,

    /// Optional per-camera initial Scheimpflug tilts. Same semantics as
    /// `initial_cameras`. `None` entries fall back to `init_tilt_x/y`.
    pub initial_sensors: Option<Vec<ScheimpflugParams>>,

    /// When Zhang's method fails for a camera, fall back to a successful
    /// camera's intrinsics + sensor as the seed. Defaults to `true` — a
    /// typical homogeneous rig has identical optics, so reusing a successful
    /// neighbor's solution is a safe starting point for non-linear refinement.
    pub fallback_to_shared_init: bool,
}

impl Default for RigScheimpflugHandeyeIntrinsicsConfig {
    fn default() -> Self {
        Self {
            init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            init_tilt_x: 0.0,
            init_tilt_y: 0.0,
            fix_scheimpflug: ScheimpflugFixMask::default(),
            initial_cameras: None,
            initial_sensors: None,
            fallback_to_shared_init: true,
        }
    }
}

/// Rig-specific options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeRigConfig {
    /// Reference camera index.
    pub reference_camera_idx: usize,
    /// Re-refine intrinsics in rig BA.
    pub refine_intrinsics_in_rig_ba: bool,
    /// Re-refine Scheimpflug in rig BA.
    pub refine_scheimpflug_in_rig_ba: bool,
    /// Fix first rig pose for gauge freedom.
    pub fix_first_rig_pose: bool,
}

impl Default for RigScheimpflugHandeyeRigConfig {
    fn default() -> Self {
        Self {
            reference_camera_idx: 0,
            refine_intrinsics_in_rig_ba: false,
            refine_scheimpflug_in_rig_ba: false,
            fix_first_rig_pose: true,
        }
    }
}

/// Hand-eye linear initialization options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeInitConfig {
    /// Minimum motion angle (degrees) for linear hand-eye initialization.
    pub min_motion_angle_deg: f64,
}

impl Default for RigScheimpflugHandeyeInitConfig {
    fn default() -> Self {
        Self {
            min_motion_angle_deg: 5.0,
        }
    }
}

/// Solver options shared by rig and hand-eye BA stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeSolverConfig {
    /// Maximum optimizer iterations.
    pub max_iters: usize,
    /// Verbosity level.
    pub verbosity: usize,
    /// Robust loss function.
    pub robust_loss: RobustLoss,
}

impl Default for RigScheimpflugHandeyeSolverConfig {
    fn default() -> Self {
        Self {
            max_iters: 80,
            verbosity: 0,
            robust_loss: RobustLoss::None,
        }
    }
}

/// Hand-eye bundle adjustment options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeBaConfig {
    /// Refine robot poses in hand-eye BA.
    pub refine_robot_poses: bool,
    /// Robot rotation prior sigma (radians).
    pub robot_rot_sigma: f64,
    /// Robot translation prior sigma (meters).
    pub robot_trans_sigma: f64,
    /// Refine cam_se3_rig in hand-eye BA.
    pub refine_cam_se3_rig_in_handeye_ba: bool,
    /// Refine Scheimpflug in hand-eye BA.
    pub refine_scheimpflug_in_handeye_ba: bool,
}

impl Default for RigScheimpflugHandeyeBaConfig {
    fn default() -> Self {
        Self {
            refine_robot_poses: true,
            robot_rot_sigma: 0.5_f64.to_radians(),
            robot_trans_sigma: 0.001,
            refine_cam_se3_rig_in_handeye_ba: false,
            refine_scheimpflug_in_handeye_ba: false,
        }
    }
}

/// Configuration for Scheimpflug rig hand-eye calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeConfig {
    /// Per-camera intrinsics options.
    pub intrinsics: RigScheimpflugHandeyeIntrinsicsConfig,
    /// Rig and gauge options.
    pub rig: RigScheimpflugHandeyeRigConfig,
    /// Hand-eye linear initialization options.
    pub handeye_init: RigScheimpflugHandeyeInitConfig,
    /// Shared solver settings.
    pub solver: RigScheimpflugHandeyeSolverConfig,
    /// Final hand-eye BA options.
    pub handeye_ba: RigScheimpflugHandeyeBaConfig,
}

/// Export format for Scheimpflug rig hand-eye calibration (EyeInHand).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugHandeyeExport {
    /// Per-camera calibrated intrinsics + distortion.
    pub cameras: Vec<PinholeCamera>,
    /// Per-camera Scheimpflug sensor parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera `cam_se3_rig` (T_C_R).
    pub cam_se3_rig: Vec<Iso3>,
    /// Gripper-to-rig transform (T_G_R).
    pub gripper_se3_rig: Iso3,
    /// Base-to-target transform (T_B_T).
    pub base_se3_target: Iso3,
    /// Per-view robot pose corrections in se(3) ([rx, ry, rz, tx, ty, tz]).
    pub robot_deltas: Option<Vec<[f64; 6]>>,
    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors.
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Multi-camera Scheimpflug rig hand-eye calibration problem (EyeInHand).
#[derive(Debug)]
pub struct RigScheimpflugHandeyeProblem;

impl ProblemType for RigScheimpflugHandeyeProblem {
    type Config = RigScheimpflugHandeyeConfig;
    type Input = RigScheimpflugHandeyeInput;
    type State = RigScheimpflugHandeyeState;
    type Output = HandEyeScheimpflugEstimate;
    type Export = RigScheimpflugHandeyeExport;

    fn name() -> &'static str {
        "rig_scheimpflug_handeye_v1"
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
        for (i, view) in input.views.iter().enumerate() {
            if view.obs.cameras.len() != input.num_cameras {
                return Err(Error::invalid_input(format!(
                    "view {i} has {} cameras, expected {}",
                    view.obs.cameras.len(),
                    input.num_cameras
                )));
            }
            if !view.obs.cameras.iter().any(|c| c.is_some()) {
                return Err(Error::invalid_input(format!(
                    "view {i} has no observations from any camera"
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

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export, Error> {
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
            .ok_or_else(|| Error::invalid_input("no target pose in output"))?;

        Ok(RigScheimpflugHandeyeExport {
            cameras: output.params.cameras.clone(),
            sensors: output.params.sensors.clone(),
            cam_se3_rig,
            gripper_se3_rig: output.params.handeye,
            base_se3_target: target_pose,
            robot_deltas: output.robot_deltas.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
        })
    }
}
