//! [`ProblemType`] implementation for single laserline device calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::stated_covariance;
use vision_calibration_core::{CameraFix, CameraModelSpec, SensorKind};
use vision_calibration_core::{
    PerFeatureResiduals, Uncertainty, build_feature_histogram,
    compute_planar_target_residuals_views,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    LaserlineDataset, LaserlineEstimate, LaserlineParams, LaserlineResidualType,
    LaserlineSolveOptions, LaserlineStats, compute_laserline_feature_residuals,
};

use crate::common::config::{CameraConfig, IntrinsicsInitConfig, SolverConfig};
use crate::common::{ExportDiagnostics, ExportKind};
use crate::session::{InvalidationPolicy, ProblemState, ProblemType};

use super::state::LaserlineDeviceState;

/// Laserline device calibration problem (single camera + laser plane).
#[derive(Debug)]
pub struct LaserlineDeviceProblem;

/// Input type for laserline device calibration.
pub type LaserlineDeviceInput = LaserlineDataset;

/// Configuration for laserline device calibration.
///
/// Settings are grouped by stage; the camera model and its fixed parameters
/// live in [`CameraConfig`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct LaserlineDeviceConfig {
    /// Per-camera linear-initialization stage settings. The linear stage
    /// fits only the distortion terms the bundle adjustment estimates
    /// ([`Self::effective_fix`]).
    pub init: IntrinsicsInitConfig,
    /// The camera model and the parameters the bundle adjustment holds
    /// fixed. Any distortion kind, and an identity or Scheimpflug sensor;
    /// the line-distance laser residual needs a projection that keeps lines
    /// straight. A Scheimpflug sensor's tilt is held by default at the
    /// manual-init `sensor` seed (zero when unseeded): a device's mount tilt
    /// is measured separately. An explicit `fix.sensor` list estimates it
    /// instead (`[]` refines both angles; unseeded, the tilt then comes from
    /// the linear tilt sweep and a staged multi-start).
    #[serde(default)]
    pub camera: CameraConfig,
    /// Non-linear solve stage settings.
    ///
    /// `robust_loss` is **not consulted** by this problem: laser-carrying
    /// stages track calibration and laser residuals as independent families
    /// with their own robust losses (`optimize.calib_loss`,
    /// `optimize.laser_loss`). `max_iters`, `verbosity` and `backend` apply.
    pub solver: SolverConfig,
    /// Bundle-adjustment options.
    pub optimize: DeviceOptimizeConfig,
}

/// Bundle-adjustment options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
#[serde(deny_unknown_fields)]
pub struct DeviceOptimizeConfig {
    /// Robust loss for calibration reprojection residuals.
    pub calib_loss: vision_calibration_optim::RobustLoss,
    /// Robust loss for laser residuals. Its scale is in the residual's unit:
    /// pixels for the line distance (the default residual), meters for
    /// point-to-plane.
    pub laser_loss: vision_calibration_optim::RobustLoss,
    /// Global weight for calibration residuals.
    pub calib_weight: f64,
    /// Global weight for laser residuals.
    pub laser_weight: f64,
    /// Indices of target poses to hold fixed. None by default: each view's
    /// pose is measured by its own corners, so there is no gauge to fix. A
    /// held pose keeps its initial value, and the linear init is not robust
    /// to outlying corners: holding one keeps the solve off the minimum when
    /// some corners are outliers.
    pub fix_poses: Vec<usize>,
    /// Fix laser plane parameters during optimization.
    pub fix_plane: bool,
    /// Laser residual type.
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for DeviceOptimizeConfig {
    fn default() -> Self {
        Self {
            calib_loss: vision_calibration_optim::RobustLoss::Huber { scale: 1.0 },
            laser_loss: vision_calibration_optim::RobustLoss::Huber { scale: 1.0 },
            calib_weight: 1.0,
            laser_weight: 1.0,
            fix_poses: Vec::new(),
            fix_plane: false,
            laser_residual_type: LaserlineResidualType::LineDistNormalized,
        }
    }
}

impl LaserlineDeviceConfig {
    /// Whether the device's Scheimpflug tilt is held, not estimated: the
    /// default for a laserline device, whose mount tilt is a hardware
    /// property measured separately and given through the manual-init
    /// `sensor` seed (zero when unseeded). An explicit `camera.fix.sensor`
    /// list opts into estimating the tilt (`[]` refines both angles).
    pub fn holds_tilt(&self) -> bool {
        self.camera.model.sensor == SensorKind::Scheimpflug && self.camera.fix.sensor.is_none()
    }

    /// Convert to vision-calibration-linear initialization options. The
    /// linear stage fits only the distortion terms the bundle adjustment
    /// estimates ([`Self::effective_fix`]): with the tilt estimated, the
    /// default list holds the tangential terms, which trade off against it;
    /// with the tilt held, they are free.
    pub(crate) fn init_opts(&self) -> IterativeIntrinsicsOptions {
        let camera = CameraConfig {
            model: self.camera.model,
            fix: self.effective_fix(),
        };
        self.init.iterative_opts(None, &camera)
    }

    /// The fix lists the bundle adjustment applies. A held tilt (see
    /// [`Self::holds_tilt`]) fixes `tilt_x` and `tilt_y`, and an unset
    /// distortion list takes the untilted default: with the tilt known, the
    /// tangential terms do not trade off against it.
    pub fn effective_fix(&self) -> CameraFix {
        let mut fix = self.camera.fix.clone();
        if self.holds_tilt() {
            let names = self.camera.model.sensor.param_names();
            fix.sensor = Some(names.iter().map(|n| n.to_string()).collect());
            if fix.distortion.is_none() {
                fix.distortion = CameraModelSpec {
                    sensor: SensorKind::Identity,
                    ..self.camera.model
                }
                .default_fix()
                .distortion;
            }
        }
        fix
    }

    /// Convert to vision-calibration-optim solve options; the camera's fix
    /// lists are [`Self::effective_fix`].
    pub(crate) fn solve_opts(&self) -> LaserlineSolveOptions {
        LaserlineSolveOptions {
            calib_loss: self.optimize.calib_loss,
            calib_weight: self.optimize.calib_weight,
            laser_loss: self.optimize.laser_loss,
            laser_weight: self.optimize.laser_weight,
            fix: self.effective_fix(),
            fix_poses: self.optimize.fix_poses.clone(),
            fix_plane: self.optimize.fix_plane,
            laser_residual_type: self.optimize.laser_residual_type,
        }
    }
}

/// The laserline device's estimate: the bundle solve's estimate plus the
/// residual statistics the export reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceEstimate {
    /// Optimized camera/laser parameters and backend report.
    pub estimate: LaserlineEstimate,
    /// Aggregated reprojection and laser residual statistics.
    pub stats: LaserlineStats,
}

/// Export type for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct LaserlineDeviceExport {
    /// Export-type discriminator — always [`ExportKind::LaserlineDevice`].
    pub kind: ExportKind,
    /// The calibrated camera, the laser plane (camera frame) and the
    /// per-view target poses.
    pub params: LaserlineParams,
    /// Laserline statistics payload.
    pub stats: LaserlineStats,
    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors (single element for single-camera workflows).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Covariance of the calibrated parameters (camera and laser plane), with
    /// the noise level the residuals imply.
    #[serde(default)]
    pub uncertainty: Uncertainty,
    /// Residuals and the image manifest: evidence for viewers and
    /// debugging, not part of the result.
    #[serde(default)]
    pub diagnostics: ExportDiagnostics,
}

impl ProblemState for LaserlineDeviceProblem {
    type State = LaserlineDeviceState;
}

impl ProblemType for LaserlineDeviceProblem {
    type Config = LaserlineDeviceConfig;
    type Input = LaserlineDeviceInput;
    type Output = LaserlineDeviceEstimate;
    type Export = LaserlineDeviceExport;

    fn name() -> &'static str {
        "laserline_device_v1"
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
        stated_covariance(input.iter().map(|v| &v.obs))?;
        if input.len() < 3 {
            return Err(Error::InsufficientData {
                need: 3,
                got: input.len(),
            });
        }

        for (i, view) in input.iter().enumerate() {
            if view.obs.len() < 4 {
                return Err(Error::invalid_input(format!(
                    "view {} has too few points (need >= 4 for homography, got {})",
                    i,
                    view.obs.len()
                )));
            }
            view.meta
                .validate()
                .map_err(|e| Error::invalid_input(format!("view {}: {}", i, e)))?;
        }

        Ok(())
    }

    fn validate_config(config: &Self::Config) -> Result<(), Error> {
        if config.solver.max_iters == 0 {
            return Err(Error::invalid_input("max_iters must be positive"));
        }
        if config.init.iterations == 0 {
            return Err(Error::invalid_input("init.iterations must be positive"));
        }
        if config.optimize.calib_weight <= 0.0 {
            return Err(Error::invalid_input("calib_weight must be positive"));
        }
        if config.optimize.laser_weight <= 0.0 {
            return Err(Error::invalid_input("laser_weight must be positive"));
        }
        config.camera.validate_single_camera()
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        // Evaluate the target residuals in the calibrated camera's full
        // model (distortion and sensor), the model the optimizer used.
        let params = &output.estimate.params;
        let camera = params
            .camera
            .build()
            .map_err(|e| Error::invalid_input(format!("camera model: {e}")))?;
        let target = compute_planar_target_residuals_views(&camera, input, &params.poses)?;
        let target_hist = build_feature_histogram(target.iter().filter_map(|r| r.error_px));

        let laser = compute_laserline_feature_residuals(input, params)?;
        let laser_hist =
            build_feature_histogram(laser.features.iter().filter_map(|r| r.residual_px));

        let mut per_feature_residuals = PerFeatureResiduals::default();
        per_feature_residuals.target = target;
        per_feature_residuals.laser = laser.features;
        per_feature_residuals.laser_lines = laser.lines;
        per_feature_residuals.target_hist_per_camera = Some(vec![target_hist]);
        per_feature_residuals.laser_hist_per_camera = Some(vec![laser_hist]);
        Ok(LaserlineDeviceExport {
            kind: ExportKind::LaserlineDevice,
            params: output.estimate.params.clone(),
            stats: output.stats.clone(),
            mean_reproj_error: output.stats.mean_reproj_error,
            per_cam_reproj_errors: vec![output.stats.mean_reproj_error],
            uncertainty: output.estimate.uncertainty.clone(),
            // Callers that have the images fill `image_manifest`; the
            // pipeline never has image paths.
            diagnostics: ExportDiagnostics::new(per_feature_residuals),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, Camera, CameraParams, CorrespondenceView, DistortionParams, FxFyCxCy, Iso3,
        Pinhole, ProjectionParams, Pt2, Pt3, ScheimpflugParams, SensorParams, View,
    };
    use vision_calibration_optim::{LaserPlane, LaserlineMeta, LaserlineParams, SolveReport};

    fn bc_camera(
        intrinsics: FxFyCxCy<f64>,
        distortion: BrownConrady5<f64>,
        sensor: SensorParams,
    ) -> CameraParams {
        CameraParams {
            projection: ProjectionParams::Pinhole,
            distortion: DistortionParams::BrownConrady5 { params: distortion },
            sensor,
            intrinsics,
        }
    }

    fn make_synthetic_scenario() -> (LaserlineDataset, LaserlineDeviceEstimate) {
        // Camera + target plane setup matching the laser feature residual
        // synthetic test: target plane at z=1 in camera frame, laser plane at
        // y=0.1 in camera frame. Pixels lie exactly on the projected line +
        // exactly on the projected target board.
        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };
        let distortion = BrownConrady5::default();
        let camera = vision_calibration_core::make_pinhole_camera(intrinsics, distortion);

        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 1.0, 0.0), -0.1);

        let mut views = Vec::new();
        let mut poses = Vec::new();
        for view_idx in 0..3 {
            // Each view's target plane lives at depth z = 1.0 + 0.05*i in
            // camera frame. Generate laser pixels by projecting points along
            // the per-view intersection line (y = 0.1, z = view_z).
            let view_z = 1.0 + 0.05 * view_idx as f64;
            let pose = Iso3::from_parts(
                nalgebra::Translation3::new(0.0, 0.0, view_z),
                nalgebra::UnitQuaternion::identity(),
            );
            poses.push(pose);
            let target_pixels: Vec<Pt2> = board
                .iter()
                .map(|p| {
                    let p_cam = pose * p;
                    camera.project_point_c(&p_cam.coords).unwrap()
                })
                .collect();
            let laser_pixels: Vec<Pt2> = [-0.10, -0.05, 0.0, 0.05, 0.10]
                .iter()
                .map(|&x| {
                    camera
                        .project_point_c(&nalgebra::Vector3::new(x, 0.1, view_z))
                        .unwrap()
                })
                .collect();
            views.push(View::new(
                CorrespondenceView::new(board.clone(), target_pixels).unwrap(),
                LaserlineMeta {
                    laser_pixels,
                    laser_weights: Vec::new(),
                },
            ));
        }

        let params = LaserlineParams::new(
            bc_camera(intrinsics, distortion, SensorParams::Identity),
            poses,
            plane,
        )
        .unwrap();
        let estimate = LaserlineEstimate {
            params,
            report: SolveReport {
                final_cost: 0.0,
                num_iters: 0,
            },
            uncertainty: Default::default(),
        };
        let stats = LaserlineStats {
            mean_reproj_error: 0.0,
            mean_laser_error: 0.0,
            per_view_reproj_errors: vec![0.0, 0.0, 0.0],
            per_view_laser_errors: vec![0.0, 0.0, 0.0],
        };
        (views, LaserlineDeviceEstimate { estimate, stats })
    }

    #[test]
    fn export_per_feature_residuals_zero_for_perfect_data() {
        let (dataset, output) = make_synthetic_scenario();
        let export =
            LaserlineDeviceProblem::export(&dataset, &output, &LaserlineDeviceConfig::default())
                .expect("export");

        // 3 views × 4 target features = 12; 3 views × 5 laser pixels = 15.
        assert_eq!(export.diagnostics.per_feature_residuals.target.len(), 12);
        assert_eq!(export.diagnostics.per_feature_residuals.laser.len(), 15);
        for r in &export.diagnostics.per_feature_residuals.target {
            assert!(r.error_px.unwrap() < 1e-9);
        }
        for r in &export.diagnostics.per_feature_residuals.laser {
            let res_m = r.residual_m.expect("ray must hit target plane");
            let res_px = r.residual_px.expect("line endpoints must be recoverable");
            assert!(res_m.abs() < 1e-6, "residual_m {res_m}");
            assert!(res_px < 1e-6, "residual_px {res_px}");
        }
        let target_hist = &export
            .diagnostics
            .per_feature_residuals
            .target_hist_per_camera
            .as_ref()
            .unwrap()[0];
        assert_eq!(target_hist.count, 12);
        assert_eq!(target_hist.counts, [12, 0, 0, 0, 0]);
        let laser_hist = &export
            .diagnostics
            .per_feature_residuals
            .laser_hist_per_camera
            .as_ref()
            .unwrap()[0];
        assert_eq!(laser_hist.count, 15);
        assert_eq!(laser_hist.counts, [15, 0, 0, 0, 0]);
    }

    #[test]
    fn export_uses_scheimpflug_sensor_for_target_residuals() {
        // Regression test: target reprojection
        // residuals must use the calibrated Scheimpflug sensor, not a plain
        // pinhole. The setup constructs ground-truth pixels with a non-zero
        // Scheimpflug tilt; if the export rebuilds a sensor-less PinholeCamera
        // (the bug), the resulting `error_px` values will be much larger than
        // 1e-6 px because the ground-truth pixels embed the tilt that the
        // bare pinhole projection ignores.
        let intrinsics = FxFyCxCy {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
        };
        let distortion = BrownConrady5::default();
        let sensor = ScheimpflugParams {
            tilt_x: 0.05,
            tilt_y: -0.03,
        };
        // Project through the full Scheimpflug-tilted camera to generate
        // ground-truth pixels.
        let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);

        let board = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let pose = Iso3::from_parts(
            nalgebra::Translation3::new(0.0, 0.0, 1.0),
            nalgebra::UnitQuaternion::identity(),
        );
        let pixels: Vec<Pt2> = board
            .iter()
            .map(|p| {
                let p_cam = pose * p;
                camera.project_point_c(&p_cam.coords).unwrap()
            })
            .collect();
        let dataset = vec![View::new(
            CorrespondenceView::new(board.clone(), pixels).unwrap(),
            LaserlineMeta {
                laser_pixels: vec![Pt2::new(320.0, 240.0)],
                laser_weights: Vec::new(),
            },
        )];
        let plane = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -1.0);
        let params = LaserlineParams::new(
            bc_camera(
                intrinsics,
                distortion,
                SensorParams::Scheimpflug { params: sensor },
            ),
            vec![pose],
            plane,
        )
        .unwrap();
        let output = LaserlineDeviceEstimate {
            estimate: LaserlineEstimate {
                params,
                report: SolveReport {
                    final_cost: 0.0,
                    num_iters: 0,
                },
                uncertainty: Default::default(),
            },
            stats: LaserlineStats {
                mean_reproj_error: 0.0,
                mean_laser_error: 0.0,
                per_view_reproj_errors: vec![0.0],
                per_view_laser_errors: vec![0.0],
            },
        };

        let export =
            LaserlineDeviceProblem::export(&dataset, &output, &LaserlineDeviceConfig::default())
                .expect("export");
        assert_eq!(export.diagnostics.per_feature_residuals.target.len(), 4);
        for r in &export.diagnostics.per_feature_residuals.target {
            let err = r.error_px.expect("residual missing");
            assert!(
                err < 1e-6,
                "feature {} error_px {} should be near zero with correct sensor model",
                r.feature,
                err
            );
        }
    }

    #[test]
    fn validate_config_checks_the_camera() {
        use vision_calibration_core::{CameraModelSpec, DistortionKind};
        let config = LaserlineDeviceConfig {
            camera: CameraConfig::new(CameraModelSpec::pinhole(
                DistortionKind::None,
                SensorKind::Scheimpflug,
            )),
            ..LaserlineDeviceConfig::default()
        };
        let err = LaserlineDeviceProblem::validate_config(&config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("distortion model"), "{err}");
        let mut config = LaserlineDeviceConfig::default();
        config.camera.fix.sensor = Some(vec!["tilt_x".into()]);
        let err = LaserlineDeviceProblem::validate_config(&config)
            .unwrap_err()
            .to_string();
        assert!(err.contains("tilt_x"), "{err}");
    }

    #[test]
    fn the_linear_stage_fits_only_what_the_bundle_adjustment_estimates() {
        use vision_calibration_core::DistortionKind;
        let tangential_held =
            |config: &LaserlineDeviceConfig| config.init_opts().distortion_opts.fix_tangential;
        // A frontal camera: free by default, held when its fix list holds them.
        let mut config = LaserlineDeviceConfig::default();
        assert!(!tangential_held(&config));
        config.camera.fix.distortion = Some(vec!["k3".into(), "p1".into(), "p2".into()]);
        assert!(tangential_held(&config));
        // A tilted device holds its tilt by default, so the tangential terms
        // follow the untilted default list; estimating the tilt holds them.
        let mut config = LaserlineDeviceConfig {
            camera: CameraConfig::new(CameraModelSpec::pinhole(
                DistortionKind::BrownConrady5,
                SensorKind::Scheimpflug,
            )),
            ..LaserlineDeviceConfig::default()
        };
        assert!(!tangential_held(&config));
        config.camera.fix.sensor = Some(Vec::new());
        assert!(tangential_held(&config));
    }

    #[test]
    fn export_per_feature_residuals_json_roundtrip() {
        let (dataset, output) = make_synthetic_scenario();
        let export =
            LaserlineDeviceProblem::export(&dataset, &output, &LaserlineDeviceConfig::default())
                .expect("export");
        let json = serde_json::to_string(&export).unwrap();
        let restored: LaserlineDeviceExport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.diagnostics.per_feature_residuals.target.len(),
            export.diagnostics.per_feature_residuals.target.len()
        );
        assert_eq!(
            restored.diagnostics.per_feature_residuals.laser.len(),
            export.diagnostics.per_feature_residuals.laser.len()
        );
        assert_eq!(
            restored
                .diagnostics
                .per_feature_residuals
                .target_hist_per_camera,
            export
                .diagnostics
                .per_feature_residuals
                .target_hist_per_camera
        );
        assert_eq!(
            restored
                .diagnostics
                .per_feature_residuals
                .laser_hist_per_camera,
            export
                .diagnostics
                .per_feature_residuals
                .laser_hist_per_camera
        );
    }
}
