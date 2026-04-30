//! [`ProblemType`] implementation for single laserline device calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    Camera, PerFeatureResiduals, Pinhole, ScheimpflugParams, TargetFeatureResidual,
    build_feature_histogram,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, LaserlineDataset, LaserlineEstimate, LaserlineResidualType,
    LaserlineSolveOptions, LaserlineStats, compute_laserline_feature_residuals,
};

use crate::session::{InvalidationPolicy, ProblemType};

use super::state::LaserlineDeviceState;

/// Laserline device calibration problem (single camera + laser plane).
#[derive(Debug)]
pub struct LaserlineDeviceProblem;

/// Input type for laserline device calibration.
pub type LaserlineDeviceInput = LaserlineDataset;

/// Configuration for laserline device calibration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceConfig {
    /// Initialization options.
    pub init: LaserlineDeviceInitConfig,
    /// Shared solver options.
    pub solver: LaserlineDeviceSolverConfig,
    /// Bundle-adjustment options.
    pub optimize: LaserlineDeviceOptimizeConfig,
}

/// Initialization options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceInitConfig {
    /// Number of iterations for iterative intrinsics estimation.
    pub iterations: usize,
    /// Fix k3 during initialization (recommended for typical lenses).
    pub fix_k3: bool,
    /// Fix tangential distortion during initialization.
    pub fix_tangential: bool,
    /// Enforce zero skew during initialization.
    pub zero_skew: bool,
    /// Initial Scheimpflug sensor parameters (use zeros for pinhole/identity).
    pub sensor_init: ScheimpflugParams,
}

impl Default for LaserlineDeviceInitConfig {
    fn default() -> Self {
        Self {
            iterations: 2,
            fix_k3: true,
            fix_tangential: false,
            zero_skew: true,
            sensor_init: ScheimpflugParams::default(),
        }
    }
}

/// Shared solver options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceSolverConfig {
    /// Maximum iterations for the optimizer.
    pub max_iters: usize,
    /// Verbosity level (0 = silent, 1 = summary, 2+ = detailed).
    pub verbosity: usize,
}

impl Default for LaserlineDeviceSolverConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            verbosity: 0,
        }
    }
}

/// Bundle-adjustment options for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceOptimizeConfig {
    /// Robust loss for calibration reprojection residuals.
    pub calib_loss: vision_calibration_optim::RobustLoss,
    /// Robust loss for laser residuals.
    pub laser_loss: vision_calibration_optim::RobustLoss,
    /// Global weight for calibration residuals.
    pub calib_weight: f64,
    /// Global weight for laser residuals.
    pub laser_weight: f64,
    /// Fix camera intrinsics during optimization.
    pub fix_intrinsics: bool,
    /// Fix distortion parameters during optimization.
    pub fix_distortion: bool,
    /// Fix k3 distortion parameter during optimization.
    pub fix_k3: bool,
    /// Fix Scheimpflug sensor parameters during optimization.
    pub fix_sensor: bool,
    /// Indices of poses to fix (e.g., \[0\] to fix first pose).
    pub fix_poses: Vec<usize>,
    /// Fix laser plane parameters during optimization.
    pub fix_plane: bool,
    /// Laser residual type.
    pub laser_residual_type: LaserlineResidualType,
}

impl Default for LaserlineDeviceOptimizeConfig {
    fn default() -> Self {
        Self {
            calib_loss: vision_calibration_optim::RobustLoss::Huber { scale: 1.0 },
            laser_loss: vision_calibration_optim::RobustLoss::Huber { scale: 0.01 },
            calib_weight: 1.0,
            laser_weight: 1.0,
            fix_intrinsics: false,
            fix_distortion: false,
            fix_k3: true,
            fix_sensor: true,
            fix_poses: vec![0],
            fix_plane: false,
            laser_residual_type: LaserlineResidualType::LineDistNormalized,
        }
    }
}

impl LaserlineDeviceConfig {
    /// Convert to vision-calibration-linear initialization options.
    pub fn init_opts(&self) -> IterativeIntrinsicsOptions {
        IterativeIntrinsicsOptions {
            iterations: self.init.iterations,
            distortion_opts: DistortionFitOptions {
                fix_k3: self.init.fix_k3,
                fix_tangential: self.init.fix_tangential,
                iters: 8,
            },
            zero_skew: self.init.zero_skew,
        }
    }

    /// Convert to vision-calibration-optim solve options.
    pub fn solve_opts(&self) -> LaserlineSolveOptions {
        LaserlineSolveOptions {
            calib_loss: self.optimize.calib_loss,
            calib_weight: self.optimize.calib_weight,
            laser_loss: self.optimize.laser_loss,
            laser_weight: self.optimize.laser_weight,
            fix_intrinsics: self.optimize.fix_intrinsics,
            fix_distortion: self.optimize.fix_distortion,
            fix_k3: self.optimize.fix_k3,
            fix_sensor: self.optimize.fix_sensor,
            fix_poses: self.optimize.fix_poses.clone(),
            fix_plane: self.optimize.fix_plane,
            laser_residual_type: self.optimize.laser_residual_type,
        }
    }

    /// Convert to backend solver options.
    pub fn backend_opts(&self) -> BackendSolveOptions {
        BackendSolveOptions {
            max_iters: self.solver.max_iters,
            verbosity: self.solver.verbosity,
            ..Default::default()
        }
    }
}

/// Pipeline output including optimized parameters and summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaserlineDeviceOutput {
    /// Optimized camera/laser parameters and backend report.
    pub estimate: LaserlineEstimate,
    /// Aggregated reprojection and laser residual statistics.
    pub stats: LaserlineStats,
}

/// Export type for laserline device calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceExport {
    /// Pipeline output including optimized parameters and summary statistics.
    pub estimate: LaserlineEstimate,
    /// Laserline statistics payload.
    pub stats: LaserlineStats,
    /// Mean reprojection error (pixels).
    pub mean_reproj_error: f64,
    /// Per-camera reprojection errors (single element for single-camera workflows).
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-feature reprojection + laser residuals (ADR 0012). Single-camera:
    /// `target_hist_per_camera` and `laser_hist_per_camera` each carry
    /// `Some(vec![one_entry])`.
    #[serde(default)]
    pub per_feature_residuals: PerFeatureResiduals,
}

impl ProblemType for LaserlineDeviceProblem {
    type Config = LaserlineDeviceConfig;
    type Input = LaserlineDeviceInput;
    type State = LaserlineDeviceState;
    type Output = LaserlineDeviceOutput;
    type Export = LaserlineDeviceExport;

    fn name() -> &'static str {
        "laserline_device_v1"
    }

    fn validate_input(input: &Self::Input) -> Result<(), Error> {
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
            return Err(Error::invalid_input("init_iterations must be positive"));
        }
        if config.optimize.calib_weight <= 0.0 {
            return Err(Error::invalid_input("calib_weight must be positive"));
        }
        if config.optimize.laser_weight <= 0.0 {
            return Err(Error::invalid_input("laser_weight must be positive"));
        }
        Ok(())
    }

    fn on_input_change() -> InvalidationPolicy {
        InvalidationPolicy::CLEAR_COMPUTED
    }

    fn export(
        input: &Self::Input,
        output: &Self::Output,
        _config: &Self::Config,
    ) -> Result<Self::Export, Error> {
        // Reconstruct the full calibrated camera (pinhole core + Brown-Conrady
        // distortion + Scheimpflug-tilted sensor) so target reprojection
        // residuals match the projection model the optimizer used. Skipping
        // the sensor here would mis-report errors for any non-identity
        // Scheimpflug calibration.
        //
        // Inlined projection loop because the core helper
        // `compute_planar_target_residuals_views` is currently typed against
        // PinholeCamera. PR #34 (follow-ups) generifies it over a
        // CameraProject trait so this can be replaced with one helper call.
        let params = &output.estimate.params;
        let camera = Camera::new(
            Pinhole,
            params.distortion,
            params.sensor.compile(),
            params.intrinsics,
        );
        if params.poses.len() != input.len() {
            return Err(Error::invalid_input(format!(
                "camera_se3_target count {} != view count {}",
                params.poses.len(),
                input.len()
            )));
        }
        let mut target = Vec::new();
        for (view_idx, view) in input.iter().enumerate() {
            let pose = params.poses[view_idx];
            for (feature_idx, (p3d, p2d)) in view
                .obs
                .points_3d
                .iter()
                .zip(view.obs.points_2d.iter())
                .enumerate()
            {
                let p_cam = pose * p3d;
                let (projected_px, error_px) = match camera.project_point_c(&p_cam.coords) {
                    Some(proj) => (Some([proj.x, proj.y]), Some((proj - *p2d).norm())),
                    None => (None, None),
                };
                target.push(TargetFeatureResidual {
                    pose: view_idx,
                    camera: 0,
                    feature: feature_idx,
                    target_xyz_m: [p3d.x, p3d.y, p3d.z],
                    observed_px: [p2d.x, p2d.y],
                    projected_px,
                    error_px,
                });
            }
        }
        let target_hist = build_feature_histogram(target.iter().filter_map(|r| r.error_px));

        let laser = compute_laserline_feature_residuals(input, params)?;
        let laser_hist = build_feature_histogram(laser.iter().filter_map(|r| r.residual_px));

        Ok(LaserlineDeviceExport {
            estimate: output.estimate.clone(),
            stats: output.stats.clone(),
            mean_reproj_error: output.stats.mean_reproj_error,
            per_cam_reproj_errors: vec![output.stats.mean_reproj_error],
            per_feature_residuals: PerFeatureResiduals {
                target,
                laser,
                target_hist_per_camera: Some(vec![target_hist]),
                laser_hist_per_camera: Some(vec![laser_hist]),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Pt2, Pt3, View,
    };
    use vision_calibration_optim::{LaserPlane, LaserlineMeta, LaserlineParams, SolveReport};

    fn make_synthetic_scenario() -> (LaserlineDataset, LaserlineDeviceOutput) {
        // Camera + target plane setup matching the laser feature residual
        // synthetic test: target plane at z=1 in camera frame, laser plane at
        // y=0.1 in camera frame. Pixels lie exactly on the projected line +
        // exactly on the projected target board.
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
        };
        let distortion = BrownConrady5::default();
        let sensor = ScheimpflugParams::default();
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

        let params = LaserlineParams::new(intrinsics, distortion, sensor, poses, plane).unwrap();
        let estimate = LaserlineEstimate {
            params,
            report: SolveReport { final_cost: 0.0 },
        };
        let stats = LaserlineStats {
            mean_reproj_error: 0.0,
            mean_laser_error: 0.0,
            per_view_reproj_errors: vec![0.0, 0.0, 0.0],
            per_view_laser_errors: vec![0.0, 0.0, 0.0],
        };
        (views, LaserlineDeviceOutput { estimate, stats })
    }

    #[test]
    fn export_per_feature_residuals_zero_for_perfect_data() {
        let (dataset, output) = make_synthetic_scenario();
        let export =
            LaserlineDeviceProblem::export(&dataset, &output, &LaserlineDeviceConfig::default())
                .expect("export");

        // 3 views × 4 target features = 12; 3 views × 5 laser pixels = 15.
        assert_eq!(export.per_feature_residuals.target.len(), 12);
        assert_eq!(export.per_feature_residuals.laser.len(), 15);
        for r in &export.per_feature_residuals.target {
            assert!(r.error_px.unwrap() < 1e-9);
        }
        for r in &export.per_feature_residuals.laser {
            let res_m = r.residual_m.expect("ray must hit target plane");
            let res_px = r.residual_px.expect("line endpoints must be recoverable");
            assert!(res_m.abs() < 1e-6, "residual_m {res_m}");
            assert!(res_px < 1e-6, "residual_px {res_px}");
        }
        let target_hist = &export
            .per_feature_residuals
            .target_hist_per_camera
            .as_ref()
            .unwrap()[0];
        assert_eq!(target_hist.count, 12);
        assert_eq!(target_hist.counts, [12, 0, 0, 0, 0]);
        let laser_hist = &export
            .per_feature_residuals
            .laser_hist_per_camera
            .as_ref()
            .unwrap()[0];
        assert_eq!(laser_hist.count, 15);
        assert_eq!(laser_hist.counts, [15, 0, 0, 0, 0]);
    }

    #[test]
    fn export_uses_scheimpflug_sensor_for_target_residuals() {
        // Regression test for Codex P1 on PR #33: target reprojection
        // residuals must use the calibrated Scheimpflug sensor, not a plain
        // pinhole. The setup constructs ground-truth pixels with a non-zero
        // Scheimpflug tilt; if the export rebuilds a sensor-less PinholeCamera
        // (the bug), the resulting `error_px` values will be much larger than
        // 1e-6 px because the ground-truth pixels embed the tilt that the
        // bare pinhole projection ignores.
        let intrinsics = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 320.0,
            cy: 240.0,
            skew: 0.0,
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
        let params =
            LaserlineParams::new(intrinsics, distortion, sensor, vec![pose], plane).unwrap();
        let output = LaserlineDeviceOutput {
            estimate: LaserlineEstimate {
                params,
                report: SolveReport { final_cost: 0.0 },
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
        assert_eq!(export.per_feature_residuals.target.len(), 4);
        for r in &export.per_feature_residuals.target {
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
    fn export_per_feature_residuals_json_roundtrip() {
        let (dataset, output) = make_synthetic_scenario();
        let export =
            LaserlineDeviceProblem::export(&dataset, &output, &LaserlineDeviceConfig::default())
                .expect("export");
        let json = serde_json::to_string(&export).unwrap();
        let restored: LaserlineDeviceExport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.per_feature_residuals.target.len(),
            export.per_feature_residuals.target.len()
        );
        assert_eq!(
            restored.per_feature_residuals.laser.len(),
            export.per_feature_residuals.laser.len()
        );
        assert_eq!(
            restored.per_feature_residuals.target_hist_per_camera,
            export.per_feature_residuals.target_hist_per_camera
        );
        assert_eq!(
            restored.per_feature_residuals.laser_hist_per_camera,
            export.per_feature_residuals.laser_hist_per_camera
        );
    }
}
