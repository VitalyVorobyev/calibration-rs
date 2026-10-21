//! [`ProblemType`] implementation for rig-level laserline calibration.

use crate::Error;
use crate::rig_handeye::RigHandeyeExport;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, FeatureResidualHistogram, ImageManifest, Iso3, NoMeta, PerFeatureResiduals,
    RigDataset, RigView, RigViewObs, Uncertainty, build_feature_histogram,
    compute_rig_target_residuals,
};
use vision_calibration_optim::{
    LaserPlane, LaserlineResidualType, LaserlineStats, RigLaserlineDataset, RigLaserlineEstimate,
    compute_rig_laserline_feature_residuals,
};

use crate::common::ExportKind;
use crate::common::config::SolverConfig;
use crate::rig_family::rig_camera_models;
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::RigLaserlineDeviceState;

/// Upstream rig calibration (frozen starting point).
///
/// Mirrors a rig hand-eye result's per-camera parameters + cam→rig poses,
/// plus the per-view rig→target poses the downstream laserline dataset
/// spans. Each camera is evaluated in its own model: any distortion kind,
/// and an identity or Scheimpflug sensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct RigUpstreamCalibration {
    /// Per-camera parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera extrinsics `cam_se3_rig` (T_C_R).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_rig: Vec<Iso3>,
    /// Per-view rig poses `rig_se3_target` (T_R_T).
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
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
    /// Returns [`Error::InvalidInput`] if the export's camera and
    /// `cam_se3_rig` counts differ.
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
    ///     .expect("consistent rig handeye export");
    /// ```
    pub fn to_upstream_calibration(
        &self,
        rig_se3_target: Vec<Iso3>,
    ) -> Result<RigUpstreamCalibration, Error> {
        if self.cam_se3_rig.len() != self.cameras.len() {
            return Err(Error::invalid_input(format!(
                "cam_se3_rig count {} != camera count {}",
                self.cam_se3_rig.len(),
                self.cameras.len()
            )));
        }
        Ok(RigUpstreamCalibration {
            cameras: self.cameras.clone(),
            cam_se3_rig: self.cam_se3_rig.clone(),
            rig_se3_target,
        })
    }
}

/// Input for rig laserline calibration: dataset + frozen upstream calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
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
///
/// Grouped by stage. `solver.max_iters` defaults to 200: this stage
/// refines only per-camera laser-plane parameters against an already-frozen
/// rig geometry (1 DOF per view per camera), so iterations are nearly free.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct RigLaserlineDeviceConfig {
    /// Non-linear solve stage settings. `robust_loss` is **not consulted**
    /// by this problem — the underlying per-camera laserline solver picks
    /// its own calibration/laser robust losses (frozen upstream geometry,
    /// so only the laser-plane fit actually moves).
    pub solver: SolverConfig,
    /// Laser residual type (point-to-plane vs line-distance).
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for RigLaserlineDeviceConfig {
    fn default() -> Self {
        Self {
            solver: SolverConfig {
                max_iters: 200,
                verbosity: 0,
                ..SolverConfig::default()
            },
            laser_residual_type: LaserlineResidualType::default(),
        }
    }
}

/// Export format for rig laserline calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct RigLaserlineDeviceExport {
    /// Export-type discriminator — always [`ExportKind::RigLaserlineDevice`].
    pub kind: ExportKind,
    /// Per-camera laser planes in rig frame.
    pub laser_planes_rig: Vec<LaserPlane>,
    /// Per-camera laser planes in their own camera frames.
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Per-camera stats (reprojection + laser residuals).
    pub per_camera_stats: Vec<LaserlineStats>,

    /// Frozen upstream cameras, echoed so the export is self-contained for
    /// downstream viewers (3D rig scene, epipolar) — same field name as
    /// `RigHandeyeExport`. Empty when absent in the JSON (`serde(default)`).
    #[serde(default)]
    pub cameras: Vec<CameraParams>,
    /// Frozen upstream per-camera extrinsics `T_C_R`.
    #[serde(default)]
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub cam_se3_rig: Vec<Iso3>,
    /// Frozen upstream per-view rig poses `T_R_T`.
    #[serde(default)]
    #[cfg_attr(
        feature = "schemars",
        schemars(with = "Vec<vision_calibration_core::Iso3Schema>")
    )]
    pub rig_se3_target: Vec<Iso3>,
    /// Mean target reprojection error (pixels) against the frozen
    /// upstream. Diagnostic echo — this problem type does not optimize
    /// reprojection.
    #[serde(default)]
    pub mean_reproj_error: f64,
    /// Per-feature reprojection + laser residuals. Multi-camera
    /// rig: `target` covers per-corner reprojection (when present) and
    /// `laser` covers per-pixel laser distances. Both per-camera histograms
    /// are populated.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,

    /// Optional image manifest (viewer-side contract). When
    /// populated, downstream viewers (the diagnose / 3D / epipolar UIs)
    /// can locate the source image for each `(pose, camera)` slot.
    /// Tiled multi-camera frames (e.g. six 720×540 tiles in one
    /// horizontal strip) point multiple `FrameRef`s at the same
    /// `path` with disjoint ROIs. `None` means "no images shipped"; the
    /// calibration pipeline never reads this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_manifest: Option<ImageManifest>,
    /// Covariance of the laser planes in their camera frames
    /// (`cam1.plane.nx`), conditional on the upstream calibration. Each
    /// camera's plane is solved on its own, so each camera has its own
    /// noise levels.
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

/// Rig laserline calibration problem.
#[derive(Debug)]
pub struct RigLaserlineDeviceProblem;

impl ProblemState for RigLaserlineDeviceProblem {
    type State = RigLaserlineDeviceState;
}

impl ProblemType for RigLaserlineDeviceProblem {
    type Config = RigLaserlineDeviceConfig;
    type Input = RigLaserlineDeviceInput;
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
        if input.upstream.cameras.len() != n || input.upstream.cam_se3_rig.len() != n {
            return Err(Error::invalid_input(format!(
                "upstream per-camera lengths must equal num_cameras ({n})"
            )));
        }
        if let Some(i) = input
            .upstream
            .cameras
            .iter()
            .position(|c| c.spec().is_none())
        {
            return Err(Error::invalid_input(format!(
                "upstream camera {i} has a general homography sensor; the laser residuals need \
                 a camera model (an identity or Scheimpflug sensor)"
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

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.solver.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
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
        let target = compute_rig_target_residuals(
            &rig_camera_models(&upstream.cameras)?,
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
            &upstream.cameras,
            &upstream.cam_se3_rig,
            &upstream.rig_se3_target,
            &output.laser_planes_cam,
        )?;
        let laser_hist_per_camera: Vec<FeatureResidualHistogram> = (0..n)
            .map(|cam_idx| {
                build_feature_histogram(
                    laser
                        .features
                        .iter()
                        .filter(|r| r.camera == cam_idx)
                        .filter_map(|r| r.residual_px),
                )
            })
            .collect();

        // Mean over all finite per-feature reprojection errors — the
        // viewer-facing headline number (against the frozen upstream).
        let (err_sum, err_count) = target
            .iter()
            .filter_map(|r| r.error_px)
            .fold((0.0, 0usize), |(s, c), e| (s + e, c + 1));
        let mean_reproj_error = if err_count > 0 {
            err_sum / err_count as f64
        } else {
            0.0
        };

        let mut per_feature_residuals = PerFeatureResiduals::default();
        per_feature_residuals.target = target;
        per_feature_residuals.laser = laser.features;
        per_feature_residuals.laser_lines = laser.lines;
        per_feature_residuals.target_hist_per_camera = Some(target_hist_per_camera);
        per_feature_residuals.laser_hist_per_camera = Some(laser_hist_per_camera);
        Ok(RigLaserlineDeviceExport {
            kind: ExportKind::RigLaserlineDevice,
            laser_planes_rig: output.laser_planes_rig.clone(),
            laser_planes_cam: output.laser_planes_cam.clone(),
            per_camera_stats: output.per_camera_stats.clone(),
            cameras: upstream.cameras.clone(),
            cam_se3_rig: upstream.cam_se3_rig.clone(),
            rig_se3_target: upstream.rig_se3_target.clone(),
            mean_reproj_error,
            per_feature_residuals,
            // Manifest is populated by callers that wrote images for the
            // dataset; the pipeline never has image paths to fill in.
            image_manifest: None,
            uncertainty: output.uncertainty.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_export() -> RigLaserlineDeviceExport {
        // Construct an export directly — RigLaserlineDeviceExport is
        // `#[non_exhaustive]` so external callers can't, but in-module
        // tests can. This keeps the manifest tests independent of the
        // upstream calibration setup needed to drive `export()`.
        RigLaserlineDeviceExport {
            kind: ExportKind::RigLaserlineDevice,
            laser_planes_rig: Vec::new(),
            laser_planes_cam: Vec::new(),
            per_camera_stats: Vec::new(),
            cameras: Vec::new(),
            cam_se3_rig: Vec::new(),
            rig_se3_target: Vec::new(),
            mean_reproj_error: 0.0,
            per_feature_residuals: PerFeatureResiduals::default(),
            image_manifest: None,
            uncertainty: Uncertainty::default(),
        }
    }

    #[test]
    fn export_image_manifest_defaults_absent_from_wire() {
        // ADR 0014: when the pipeline emits an export the manifest is
        // None and `skip_serializing_if` keeps existing serialized JSON byte-stable.
        let export = make_minimal_export();
        assert!(export.image_manifest.is_none());

        let json = serde_json::to_string(&export).unwrap();
        assert!(
            !json.contains("image_manifest"),
            "absent manifest must not appear on the wire"
        );
        let restored: RigLaserlineDeviceExport = serde_json::from_str(&json).unwrap();
        assert!(restored.image_manifest.is_none());
    }

    #[test]
    fn export_image_manifest_roundtrip_with_tiled_roi() {
        // A tiled six-camera rig writes a single 4320×540 PNG per pose
        // and references each of its six 720×540 camera tiles via ROI;
        // mirror that shape here so any future change to PixelRect or
        // FrameRef serialization is caught at PR time.
        use vision_calibration_core::{FrameRef, ImageManifest, PixelRect};

        let mut export = make_minimal_export();
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
        let restored: RigLaserlineDeviceExport = serde_json::from_str(&json).unwrap();
        let manifest = restored
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.frames.len(), 6);
        assert_eq!(manifest.frames[0].camera, 0);
        assert_eq!(manifest.frames[5].roi.unwrap().x, 5 * 720);
    }
}
