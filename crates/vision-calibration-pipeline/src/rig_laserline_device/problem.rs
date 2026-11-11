//! [`ProblemType`] implementation for rig-level laserline calibration.

use crate::Error;
use crate::rig_handeye::RigHandeyeExport;
use serde::{Deserialize, Serialize};
use vision_calibration_core::LaserPlane;
use vision_calibration_core::stated_covariance;
use vision_calibration_core::{
    CameraParams, Iso3, NoMeta, RigDataset, RigView, RigViewObs, Uncertainty,
};
use vision_calibration_optim::{
    LaserlineResidualType, LaserlineStats, RigLaserlineDataset, RigLaserlineEstimate,
    compute_rig_laserline_feature_residuals, handeye_upstream_uncertainty,
};

use crate::common::config::{SolverConfig, check_laser_residual};
use crate::common::{ExportDiagnostics, ExportKind};
use crate::rig_family::{per_camera_histograms, rig_target_residuals};
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::RigLaserlineDeviceState;

/// Upstream rig calibration (frozen starting point).
///
/// Mirrors a rig hand-eye result's per-camera parameters + cam→rig poses,
/// plus the per-view rig→target poses the downstream laserline dataset
/// spans. Each camera is evaluated in its own model: any distortion kind,
/// and an identity or Scheimpflug sensor.
///
/// The planes are fitted with these parameters held at their values; their
/// `uncertainty` decides whether the planes' covariance carries theirs.
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
    /// The covariance of the parameters above, which the laser planes'
    /// covariance then carries: the cameras' parameters (`cam1.fx`, no
    /// prefix for a one-camera rig), their extrinsics
    /// (`cam1.cam_se3_rig.rx … tz`, a right perturbation of `cam_se3_rig`)
    /// and the views' target poses (`view3.rig_se3_target.rx … tz` among
    /// the shared parameters, correlated with the rest, or
    /// `rig_se3_target.rx … tz` in view 3's own block). Parameters it does
    /// not list count as exact. Unavailable, the default, holds all of them
    /// exact: the planes' covariance is then conditional on this
    /// calibration. [`RigHandeyeExport::to_upstream_calibration`] fills it
    /// from the hand-eye calibration's.
    #[serde(default)]
    pub uncertainty: Uncertainty,
}

impl RigHandeyeExport {
    /// Build a [`RigUpstreamCalibration`] from this hand-eye result, with
    /// one rig→target pose per view of the downstream laserline dataset.
    ///
    /// Each entry of `rig_se3_target` is the target's pose at one view of the
    /// laserline dataset, through this calibration's hand-eye chain: this
    /// export's own `rig_se3_target` when the laser views are its views, or
    /// the chain at the laser views' robot poses (`EyeInHand`: `T_R_T =
    /// T_G_R⁻¹ · T_B_G⁻¹ · T_B_T`; `EyeToHand`: `T_R_T = T_R_B · T_B_G ·
    /// T_G_T`).
    ///
    /// The upstream's `uncertainty` is this calibration's, in the
    /// upstream's terms: the cameras and extrinsics as they are, and the
    /// hand-eye and target poses as each view's `rig_se3_target`, correlated
    /// across views and with the cameras. The robot poses, with any per-view
    /// corrections this calibration estimated, count as exact.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the export's camera and
    /// `cam_se3_rig` counts differ or the covariance lists part of a pose.
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
        let uncertainty = handeye_upstream_uncertainty(
            &self.uncertainty,
            self.handeye.mode(),
            self.handeye.handeye().pose(),
            self.cameras.len(),
            &rig_se3_target,
        )?;
        Ok(RigUpstreamCalibration {
            cameras: self.cameras.clone(),
            cam_se3_rig: self.cam_se3_rig.clone(),
            rig_se3_target,
            uncertainty,
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
    /// Optional per-camera initial plane in camera frame. If `None`, each
    /// camera's plane is fitted in closed form from its laser pixels.
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
    /// Residuals and the image manifest: evidence for viewers and
    /// debugging, not part of the result.
    #[serde(default)]
    pub diagnostics: ExportDiagnostics,
    /// Covariance of the laser planes in their camera frames
    /// (`cam1.plane.nx … cam1.plane.d`). When the upstream calibration
    /// carries its covariance, the planes' carries its share, and the
    /// cameras' parameters and extrinsics are listed with it, correlated
    /// with the planes: a laser point's σ then has the whole calibration in
    /// it. The views' target poses enter the planes' share but are not
    /// listed. Without one, the covariance is conditional on the upstream
    /// calibration (its parameters are among `fixed`) and the planes of
    /// different cameras are uncorrelated. Each camera's plane is solved on
    /// its own, so each camera has its own noise levels.
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
        stated_covariance(
            input
                .dataset
                .views
                .iter()
                .flat_map(|v| v.cameras.iter().flatten()),
        )?;
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

    fn validate_input_config(input: &Self::Input, config: &Self::Config) -> Result<(), Error> {
        for (i, camera) in input.upstream.cameras.iter().enumerate() {
            if let Some(model) = camera.spec() {
                check_laser_residual(
                    config.laser_residual_type,
                    model,
                    &format!("laser_residual_type (camera {i})"),
                )?;
            }
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
        let mut per_feature_residuals = rig_target_residuals(
            &upstream.cameras,
            &target_dataset,
            &upstream.cam_se3_rig,
            &upstream.rig_se3_target,
        )?;

        let laser = compute_rig_laserline_feature_residuals(
            &input.dataset,
            &upstream.cameras,
            &upstream.cam_se3_rig,
            &upstream.rig_se3_target,
            &output.laser_planes_cam,
        )?;
        per_feature_residuals.laser_hist_per_camera = Some(per_camera_histograms(n, |cam| {
            laser
                .features
                .iter()
                .filter(move |r| r.camera == cam)
                .filter_map(|r| r.residual_px)
        }));

        // Mean over all finite per-feature reprojection errors — the
        // viewer-facing headline number (against the frozen upstream).
        let (err_sum, err_count) = per_feature_residuals
            .target
            .iter()
            .filter_map(|r| r.error_px)
            .fold((0.0, 0usize), |(s, c), e| (s + e, c + 1));
        if err_count == 0 {
            return Err(Error::numerical(
                "no target feature projects through the calibrated cameras, so the \
                 export has no reprojection error to report",
            ));
        }
        let mean_reproj_error = err_sum / err_count as f64;

        per_feature_residuals.laser = laser.features;
        per_feature_residuals.laser_lines = laser.lines;
        Ok(RigLaserlineDeviceExport {
            kind: ExportKind::RigLaserlineDevice,
            laser_planes_rig: output.laser_planes_rig.clone(),
            laser_planes_cam: output.laser_planes_cam.clone(),
            per_camera_stats: output.per_camera_stats.clone(),
            cameras: upstream.cameras.clone(),
            cam_se3_rig: upstream.cam_se3_rig.clone(),
            rig_se3_target: upstream.rig_se3_target.clone(),
            mean_reproj_error,
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
            diagnostics: ExportDiagnostics::default(),
            uncertainty: Uncertainty::default(),
        }
    }

    #[test]
    fn export_image_manifest_defaults_absent_from_wire() {
        // ADR 0014: when the pipeline emits an export the manifest is
        // None and `skip_serializing_if` keeps existing serialized JSON byte-stable.
        let export = make_minimal_export();
        assert!(export.diagnostics.image_manifest.is_none());

        let json = serde_json::to_string(&export).unwrap();
        assert!(
            !json.contains("image_manifest"),
            "absent manifest must not appear on the wire"
        );
        let restored: RigLaserlineDeviceExport = serde_json::from_str(&json).unwrap();
        assert!(restored.diagnostics.image_manifest.is_none());
    }

    #[test]
    fn export_image_manifest_roundtrip_with_tiled_roi() {
        // A tiled six-camera rig writes a single 4320×540 PNG per pose
        // and references each of its six 720×540 camera tiles via ROI;
        // mirror that shape here so any future change to PixelRect or
        // FrameRef serialization is caught at PR time.
        use vision_calibration_core::{FrameRef, PixelRect};

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
        let mut manifest = vision_calibration_core::ImageManifest::default();
        manifest.root = std::path::PathBuf::from(".");
        manifest.frames = frames;
        export.diagnostics.image_manifest = Some(manifest);

        let json = serde_json::to_string(&export).unwrap();
        let restored: RigLaserlineDeviceExport = serde_json::from_str(&json).unwrap();
        let manifest = restored
            .diagnostics
            .image_manifest
            .expect("manifest survives roundtrip");
        assert_eq!(manifest.frames.len(), 6);
        assert_eq!(manifest.frames[0].camera, 0);
        assert_eq!(manifest.frames[5].roi.unwrap().x, 5 * 720);
    }
}
