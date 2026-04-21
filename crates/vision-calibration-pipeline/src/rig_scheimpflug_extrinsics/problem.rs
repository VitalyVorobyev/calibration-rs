//! [`ProblemType`] implementation for Scheimpflug rig extrinsics calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{Iso3, NoMeta, PinholeCamera, RigDataset, ScheimpflugParams};
use vision_calibration_optim::{RigExtrinsicsScheimpflugEstimate, RobustLoss, ScheimpflugFixMask};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigScheimpflugExtrinsicsState;

/// Input: multi-camera rig dataset without per-view metadata.
pub type RigScheimpflugExtrinsicsInput = RigDataset<NoMeta>;

/// Configuration for Scheimpflug rig extrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugExtrinsicsConfig {
    // Per-camera intrinsics options
    /// Number of iterations for iterative intrinsics estimation.
    pub intrinsics_init_iterations: usize,
    /// Fix k3 during intrinsics calibration.
    pub fix_k3: bool,
    /// Fix tangential distortion (p1, p2).
    pub fix_tangential: bool,
    /// Enforce zero skew.
    pub zero_skew: bool,
    /// Initial Scheimpflug tilt around X (radians).
    pub init_tilt_x: f64,
    /// Initial Scheimpflug tilt around Y (radians).
    pub init_tilt_y: f64,
    /// Mask for Scheimpflug parameters during per-camera intrinsics refinement.
    pub fix_scheimpflug_in_intrinsics: ScheimpflugFixMask,

    // Rig options
    /// Reference camera index for rig frame (identity extrinsics).
    pub reference_camera_idx: usize,

    // Optimization options
    /// Maximum iterations for optimization.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
    /// Robust loss function for outlier handling.
    pub robust_loss: RobustLoss,

    // Rig BA options
    /// Re-refine intrinsics in rig BA (default: false).
    pub refine_intrinsics_in_rig_ba: bool,
    /// Re-refine Scheimpflug in rig BA (default: false).
    pub refine_scheimpflug_in_rig_ba: bool,
    /// Fix first rig pose for gauge freedom (default: true, fixes view 0).
    pub fix_first_rig_pose: bool,
}

impl Default for RigScheimpflugExtrinsicsConfig {
    fn default() -> Self {
        Self {
            intrinsics_init_iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            init_tilt_x: 0.0,
            init_tilt_y: 0.0,
            fix_scheimpflug_in_intrinsics: ScheimpflugFixMask::default(),
            reference_camera_idx: 0,
            max_iters: 80,
            verbosity: 0,
            robust_loss: RobustLoss::None,
            refine_intrinsics_in_rig_ba: false,
            refine_scheimpflug_in_rig_ba: false,
            fix_first_rig_pose: true,
        }
    }
}

/// Export format for Scheimpflug rig extrinsics calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigScheimpflugExtrinsicsExport {
    /// Per-camera calibrated intrinsics + distortion (pinhole core).
    pub cameras: Vec<PinholeCamera>,
    /// Per-camera Scheimpflug sensor parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera extrinsics: `cam_se3_rig` (T_C_R).
    pub cam_se3_rig: Vec<Iso3>,
    /// Per-view rig poses: `rig_se3_target` (T_R_T).
    pub rig_se3_target: Vec<Iso3>,
    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors (pixels).
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Multi-camera Scheimpflug rig extrinsics calibration problem.
#[derive(Debug)]
pub struct RigScheimpflugExtrinsicsProblem;

impl ProblemType for RigScheimpflugExtrinsicsProblem {
    type Config = RigScheimpflugExtrinsicsConfig;
    type Input = RigScheimpflugExtrinsicsInput;
    type State = RigScheimpflugExtrinsicsState;
    type Output = RigExtrinsicsScheimpflugEstimate;
    type Export = RigScheimpflugExtrinsicsExport;

    fn name() -> &'static str {
        "rig_scheimpflug_extrinsics_v1"
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
        if config.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.intrinsics_init_iterations == 0 {
            return Err(Error::invalid_input(
                "intrinsics_init_iterations must be positive",
            ));
        }
        Ok(())
    }

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        if config.reference_camera_idx >= input.num_cameras {
            return Err(Error::invalid_input(format!(
                "reference_camera_idx {} is out of range (num_cameras = {})",
                config.reference_camera_idx, input.num_cameras
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
        Ok(RigScheimpflugExtrinsicsExport {
            cameras: output.params.cameras.clone(),
            sensors: output.params.sensors.clone(),
            cam_se3_rig,
            rig_se3_target: output.params.rig_from_target.clone(),
            mean_reproj_error: output.mean_reproj_error,
            per_cam_reproj_errors: output.per_cam_reproj_errors.clone(),
        })
    }
}
