//! [`ProblemType`] implementation for rig-level laserline calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{BrownConrady5, FxFyCxCySkew, Iso3, Real, ScheimpflugParams};
use vision_calibration_optim::{
    LaserPlane, LaserlineResidualType, LaserlineStats, RigLaserlineDataset, RigLaserlineEstimate,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigLaserlineDeviceState;

/// Upstream rig calibration (frozen starting point).
///
/// This is a plain struct so the example can construct it directly from an
/// upstream `RigScheimpflugHandeyeExport`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigUpstreamCalibration {
    /// Per-camera intrinsics.
    pub intrinsics: Vec<FxFyCxCySkew<Real>>,
    /// Per-camera distortion.
    pub distortion: Vec<BrownConrady5<Real>>,
    /// Per-camera Scheimpflug sensor parameters.
    pub sensors: Vec<ScheimpflugParams>,
    /// Per-camera extrinsics `cam_se3_rig` (T_C_R).
    pub cam_se3_rig: Vec<Iso3>,
    /// Per-view rig poses `rig_se3_target` (T_R_T).
    pub rig_se3_target: Vec<Iso3>,
}

/// Input for rig laserline calibration: dataset + frozen upstream calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigLaserlineDeviceInput {
    /// Per-view, per-camera observations.
    pub dataset: RigLaserlineDataset,
    /// Upstream rig calibration (frozen).
    pub upstream: RigUpstreamCalibration,
    /// Optional per-camera initial plane in camera frame. If `None`, defaults
    /// to a generic `z=-0.2m` plane.
    pub initial_planes_cam: Option<Vec<LaserPlane>>,
}

/// Configuration for rig laserline calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigLaserlineDeviceConfig {
    /// Maximum solver iterations.
    pub max_iters: Option<usize>,
    /// Verbosity level.
    pub verbosity: Option<usize>,
    /// Laser residual type (point-to-plane vs line-distance).
    pub laser_residual_type: LaserlineResidualType,
}

/// Export format for rig laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigLaserlineDeviceExport {
    /// Per-camera laser planes in rig frame.
    pub laser_planes_rig: Vec<LaserPlane>,
    /// Per-camera laser planes in their own camera frames.
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Per-camera stats (reprojection + laser residuals).
    pub per_camera_stats: Vec<LaserlineStats>,
}

/// Rig laserline calibration problem.
#[derive(Debug)]
pub struct RigLaserlineDeviceProblem;

impl ProblemType for RigLaserlineDeviceProblem {
    type Config = RigLaserlineDeviceConfig;
    type Input = RigLaserlineDeviceInput;
    type State = RigLaserlineDeviceState;
    type Output = RigLaserlineEstimate;
    type Export = RigLaserlineDeviceExport;

    fn name() -> &'static str {
        "rig_laserline_device_v1"
    }

    fn schema_version() -> u32 {
        1
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        let n = input.dataset.num_cameras;
        if input.upstream.intrinsics.len() != n
            || input.upstream.distortion.len() != n
            || input.upstream.sensors.len() != n
            || input.upstream.cam_se3_rig.len() != n
        {
            return Err(Error::invalid_input(format!(
                "upstream per-camera lengths must equal num_cameras ({n})"
            )));
        }
        if input.upstream.rig_se3_target.len() != input.dataset.num_views() {
            return Err(Error::invalid_input(format!(
                "upstream rig_se3_target has {} entries, expected {}",
                input.upstream.rig_se3_target.len(),
                input.dataset.num_views()
            )));
        }
        if let Some(planes) = &input.initial_planes_cam
            && planes.len() != n
        {
            return Err(Error::invalid_input(format!(
                "initial_planes_cam has {} entries, expected {n}",
                planes.len()
            )));
        }
        Ok(())
    }

    fn validate_config(_config: &Self::Config) -> Result<(), Error> {
        Ok(())
    }

    fn validate_input_config(_input: &Self::Input, _config: &Self::Config) -> Result<(), Error> {
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn on_config_change() -> InvalidationPolicy {
        InvalidationPolicy::KEEP_ALL
    }

    fn export(output: &Self::Output, _config: &Self::Config) -> Result<Self::Export, Error> {
        Ok(RigLaserlineDeviceExport {
            laser_planes_rig: output.laser_planes_rig.clone(),
            laser_planes_cam: output.laser_planes_cam.clone(),
            per_camera_stats: output.per_camera_stats.clone(),
        })
    }
}
