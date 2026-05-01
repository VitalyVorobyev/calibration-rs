//! [`ProblemType`] implementation for rig-level laserline calibration.

use crate::Error;
use crate::rig_handeye::RigHandeyeExport;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, FeatureResidualHistogram, FxFyCxCySkew, Iso3, NoMeta,
    PerFeatureResiduals, Pinhole, Real, RigDataset, RigView, RigViewObs, ScheimpflugParams,
    build_feature_histogram, compute_rig_target_residuals,
};
use vision_calibration_optim::{
    LaserPlane, LaserlineResidualType, LaserlineStats, RigLaserlineDataset, RigLaserlineEstimate,
    compute_rig_laserline_feature_residuals,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::RigLaserlineDeviceState;

/// Upstream rig calibration (frozen starting point).
///
/// Mirrors a Scheimpflug hand-eye result's per-camera intrinsics + distortion +
/// Scheimpflug sensors + cam→rig poses, plus the per-view rig→target poses the
/// downstream laserline dataset spans.
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

impl RigHandeyeExport {
    /// Build a [`RigUpstreamCalibration`] from this hand-eye result, supplying
    /// the per-view rig→target poses that this export does not carry.
    ///
    /// Each entry of `rig_se3_target` must correspond to one view of the
    /// downstream laserline dataset. For a fixed target observed from a moving
    /// rig, all entries are typically equal to the canonical target pose
    /// recorded in this export (`base_se3_target` for `EyeInHand`); pass
    /// `vec![target_pose; num_views]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if `self.sensors` is `None` (pinhole
    /// rig). The downstream laserline calibration currently requires
    /// Scheimpflug sensor parameters; pinhole rigs are not yet supported.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use vision_calibration_pipeline::rig_handeye::RigHandeyeExport;
    /// # use vision_calibration_pipeline::rig_laserline_device::RigUpstreamCalibration;
    /// # use vision_calibration_core::Iso3;
    /// # let handeye_export: RigHandeyeExport = unimplemented!();
    /// # let num_views: usize = unimplemented!();
    /// # let target_pose: Iso3 = Iso3::identity();
    /// let upstream: RigUpstreamCalibration = handeye_export
    ///     .to_upstream_calibration(vec![target_pose; num_views])
    ///     .expect("Scheimpflug rig handeye export");
    /// ```
    pub fn to_upstream_calibration(
        &self,
        rig_se3_target: Vec<Iso3>,
    ) -> Result<RigUpstreamCalibration, Error> {
        let sensors = self.sensors.as_ref().ok_or_else(|| {
            Error::invalid_input(
                "to_upstream_calibration requires a Scheimpflug rig handeye export \
                 (sensors field populated); pinhole rigs are not yet supported \
                 by rig_laserline_device",
            )
        })?;
        Ok(RigUpstreamCalibration {
            intrinsics: self.cameras.iter().map(|c| c.k).collect(),
            distortion: self.cameras.iter().map(|c| c.dist).collect(),
            sensors: sensors.clone(),
            cam_se3_rig: self.cam_se3_rig.clone(),
            rig_se3_target,
        })
    }
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
    /// Per-feature reprojection + laser residuals (ADR 0012). Multi-camera
    /// rig: `target` covers per-corner reprojection (when present) and
    /// `laser` covers per-pixel laser distances. Both per-camera histograms
    /// are populated.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
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

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        let n = input.dataset.num_cameras;
        let upstream = &input.upstream;

        // Build a RigDataset<NoMeta> from the laser dataset's target slot for
        // target residual computation.
        let target_dataset = RigDataset::<NoMeta> {
            num_cameras: n,
            views: input
                .dataset
                .views
                .iter()
                .map(|v| RigView {
                    meta: NoMeta,
                    obs: RigViewObs {
                        cameras: v.cameras.clone(),
                    },
                })
                .collect(),
        };
        let scheimpflug_cameras: Vec<_> = (0..n)
            .map(|c| {
                Camera::new(
                    Pinhole,
                    upstream.distortion[c],
                    upstream.sensors[c].compile(),
                    upstream.intrinsics[c],
                )
            })
            .collect();
        let target = compute_rig_target_residuals(
            &scheimpflug_cameras,
            &target_dataset,
            &upstream.cam_se3_rig,
            &upstream.rig_se3_target,
        )?;
        let target_hist_per_camera: Vec<FeatureResidualHistogram> = (0..n)
            .map(|cam_idx| {
                build_feature_histogram(
                    target
                        .iter()
                        .filter(|r| r.camera == cam_idx)
                        .filter_map(|r| r.error_px),
                )
            })
            .collect();

        let laser = compute_rig_laserline_feature_residuals(
            &input.dataset,
            &upstream.intrinsics,
            &upstream.distortion,
            &upstream.sensors,
            &upstream.cam_se3_rig,
            &upstream.rig_se3_target,
            &output.laser_planes_cam,
        )?;
        let laser_hist_per_camera: Vec<FeatureResidualHistogram> = (0..n)
            .map(|cam_idx| {
                build_feature_histogram(
                    laser
                        .iter()
                        .filter(|r| r.camera == cam_idx)
                        .filter_map(|r| r.residual_px),
                )
            })
            .collect();

        Ok(RigLaserlineDeviceExport {
            laser_planes_rig: output.laser_planes_rig.clone(),
            laser_planes_cam: output.laser_planes_cam.clone(),
            per_camera_stats: output.per_camera_stats.clone(),
            per_feature_residuals: PerFeatureResiduals {
                target,
                laser,
                target_hist_per_camera: Some(target_hist_per_camera),
                laser_hist_per_camera: Some(laser_hist_per_camera),
            },
        })
    }
}
