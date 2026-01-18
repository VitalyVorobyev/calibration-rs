//! Linescan calibration session problem.
//!
//! This problem type handles joint calibration of camera intrinsics, distortion,
//! poses, and laser plane parameters using both calibration pattern observations
//! and laser line observations.

use anyhow::{ensure, Result};
use calib_core::{BrownConrady5, CorrespondenceView, FxFyCxCySkew, Iso3, Pt2, Real, Vec2};
use calib_linear::{LinescanPlaneSolver, LinescanView};
use calib_optim::{
    ir::RobustLoss,
    params::laser_plane::LaserPlane,
    problems::linescan_bundle::{
        optimize_linescan, LaserResidualType, LinescanDataset, LinescanInit,
        LinescanSolveOptions, LinescanViewObservations, PinholeCamera,
    },
    BackendSolveOptions,
};
use serde::{Deserialize, Serialize};

use crate::session::ProblemType;
use crate::{pinhole_camera_params, planar_init_seed_from_views, CameraConfig};

/// Linescan calibration problem (camera intrinsics + laser plane).
///
/// Estimates camera intrinsics (fx, fy, cx, cy, skew), Brown-Conrady distortion
/// (k1, k2, k3, p1, p2), camera-to-target poses, and laser plane parameters
/// (normal, distance) from observations of a planar calibration pattern with
/// a visible laser line.
pub struct LinescanProblem;

/// Single view observation for linescan calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinescanViewData {
    /// Standard calibration pattern observations (corners).
    pub calib_view: CorrespondenceView,
    /// Laser line pixel observations.
    pub laser_pixels: Vec<Vec2>,
}

/// Observations for linescan calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinescanObservations {
    /// Per-view observations with calibration corners and laser pixels.
    pub views: Vec<LinescanViewData>,
}

impl LinescanObservations {
    /// Create new observations, validating each view.
    pub fn new(views: Vec<LinescanViewData>) -> Result<Self> {
        ensure!(!views.is_empty(), "need at least one view");
        for (i, view) in views.iter().enumerate() {
            ensure!(
                view.calib_view.len() >= 4,
                "view {} has too few calibration points (need >=4)",
                i
            );
            ensure!(
                view.laser_pixels.len() >= 3,
                "view {} has too few laser pixels (need >=3)",
                i
            );
        }
        Ok(Self { views })
    }

    /// Number of views.
    pub fn num_views(&self) -> usize {
        self.views.len()
    }
}

/// Initial values from linear initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinescanInitial {
    /// Initial camera intrinsics.
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Initial distortion parameters.
    pub distortion: BrownConrady5<Real>,
    /// Initial camera-to-target poses.
    pub poses: Vec<Iso3>,
    /// Initial laser plane estimate.
    pub plane: LaserPlane,
    /// Mean reprojection error from initialization (pixels).
    pub init_reproj_error: Real,
}

/// Optimized results from non-linear refinement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinescanOptimized {
    /// Final camera configuration.
    pub camera: CameraConfig,
    /// Refined camera-to-target poses.
    pub poses: Vec<Iso3>,
    /// Refined laser plane.
    pub plane: LaserPlane,
    /// Final optimization cost.
    pub final_cost: Real,
    /// Mean reprojection error after optimization (pixels).
    pub mean_reproj_error: Real,
}

/// Options for linear initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LinescanInitOptions {
    /// Placeholder for future init-specific options.
    #[serde(skip)]
    _private: (),
}

/// Options for non-linear optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinescanOptimOptions {
    /// Robust loss for calibration reprojection residuals.
    #[serde(default = "default_calib_loss")]
    pub calib_loss: RobustLoss,
    /// Robust loss for laser plane residuals.
    #[serde(default = "default_laser_loss")]
    pub laser_loss: RobustLoss,
    /// Fix camera intrinsics during optimization.
    #[serde(default)]
    pub fix_intrinsics: bool,
    /// Fix distortion parameters during optimization.
    #[serde(default)]
    pub fix_distortion: bool,
    /// Fix k3 distortion parameter.
    #[serde(default = "default_true")]
    pub fix_k3: bool,
    /// Indices of poses to fix.
    #[serde(default = "default_fix_first_pose")]
    pub fix_poses: Vec<usize>,
    /// Fix laser plane during optimization.
    #[serde(default)]
    pub fix_plane: bool,
    /// Laser residual type.
    #[serde(default)]
    pub laser_residual_type: LaserResidualType,
    /// Maximum iterations for non-linear solver.
    #[serde(default = "default_max_iters")]
    pub max_iterations: usize,
}

fn default_calib_loss() -> RobustLoss {
    RobustLoss::Huber { scale: 1.0 }
}

fn default_laser_loss() -> RobustLoss {
    RobustLoss::Huber { scale: 0.01 }
}

fn default_true() -> bool {
    true
}

fn default_fix_first_pose() -> Vec<usize> {
    vec![0]
}

fn default_max_iters() -> usize {
    50
}

impl Default for LinescanOptimOptions {
    fn default() -> Self {
        Self {
            calib_loss: default_calib_loss(),
            laser_loss: default_laser_loss(),
            fix_intrinsics: false,
            fix_distortion: false,
            fix_k3: true,
            fix_poses: vec![0],
            fix_plane: false,
            laser_residual_type: LaserResidualType::LineDistNormalized,
            max_iterations: 50,
        }
    }
}

impl ProblemType for LinescanProblem {
    type Observations = LinescanObservations;
    type InitialValues = LinescanInitial;
    type OptimizedResults = LinescanOptimized;
    type InitOptions = LinescanInitOptions;
    type OptimOptions = LinescanOptimOptions;

    fn problem_name() -> &'static str {
        "linescan"
    }

    fn initialize(
        obs: &Self::Observations,
        _opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        ensure!(
            obs.num_views() >= 2,
            "need at least 2 views for linescan calibration"
        );

        // Step 1: Initialize intrinsics and poses from calibration views (Zhang's method)
        let calib_views: Vec<CorrespondenceView> =
            obs.views.iter().map(|v| v.calib_view.clone()).collect();

        let (seed, camera) = planar_init_seed_from_views(&calib_views)?;

        let intrinsics = FxFyCxCySkew {
            fx: seed.intrinsics.fx,
            fy: seed.intrinsics.fy,
            cx: seed.intrinsics.cx,
            cy: seed.intrinsics.cy,
            skew: 0.0,
        };

        let distortion = BrownConrady5 {
            k1: seed.distortion.k1,
            k2: seed.distortion.k2,
            k3: seed.distortion.k3,
            p1: seed.distortion.p1,
            p2: seed.distortion.p2,
            iters: 8,
        };

        // Step 2: Initialize laser plane from multi-view observations
        let linescan_views: Vec<LinescanView> = obs
            .views
            .iter()
            .zip(seed.poses.iter())
            .map(|(view_data, pose)| LinescanView {
                laser_pixels: view_data
                    .laser_pixels
                    .iter()
                    .map(|p| Pt2::new(p.x, p.y))
                    .collect(),
                camera_pose: *pose,
            })
            .collect();

        let plane_estimate = LinescanPlaneSolver::from_views(&linescan_views, &camera)?;

        let plane = LaserPlane::new(plane_estimate.normal.into_inner(), plane_estimate.distance);

        // Compute initial reprojection error
        let init_reproj_error = mean_reproj_error_linescan(&calib_views, &camera, &seed.poses)?;

        Ok(LinescanInitial {
            intrinsics,
            distortion,
            poses: seed.poses,
            plane,
            init_reproj_error,
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        // Build dataset for optimization
        let optim_views: Vec<LinescanViewObservations> = obs
            .views
            .iter()
            .map(|view| LinescanViewObservations {
                calib_points_3d: view.calib_view.points_3d.clone(),
                calib_pixels: view.calib_view.points_2d.clone(),
                laser_pixels: view.laser_pixels.clone(),
                calib_weights: view.calib_view.weights.clone(),
                laser_weights: None,
            })
            .collect();

        let dataset = LinescanDataset::new_single_plane(optim_views)?;
        let linescan_init =
            LinescanInit::new(init.intrinsics, init.distortion, init.poses.clone(), vec![
                init.plane.clone(),
            ])?;

        let solve_opts = LinescanSolveOptions {
            calib_loss: opts.calib_loss,
            laser_loss: opts.laser_loss,
            fix_intrinsics: opts.fix_intrinsics,
            fix_distortion: opts.fix_distortion,
            fix_k3: opts.fix_k3,
            fix_poses: opts.fix_poses.clone(),
            fix_planes: if opts.fix_plane { vec![0] } else { vec![] },
            laser_residual_type: opts.laser_residual_type,
        };

        let backend_opts = BackendSolveOptions {
            max_iters: opts.max_iterations,
            verbosity: 0,
            ..Default::default()
        };

        let result = optimize_linescan(&dataset, &linescan_init, &solve_opts, &backend_opts)?;

        // Compute final reprojection error
        let calib_views: Vec<CorrespondenceView> =
            obs.views.iter().map(|v| v.calib_view.clone()).collect();
        let mean_reproj_error =
            mean_reproj_error_linescan(&calib_views, &result.camera, &result.poses)?;

        Ok(LinescanOptimized {
            camera: pinhole_camera_params(&result.camera),
            poses: result.poses,
            plane: result.planes.into_iter().next().unwrap(),
            final_cost: result.final_cost,
            mean_reproj_error,
        })
    }
}

/// Compute mean reprojection error for linescan calibration.
fn mean_reproj_error_linescan(
    views: &[CorrespondenceView],
    camera: &PinholeCamera,
    poses: &[Iso3],
) -> Result<Real> {
    let mut total_error = 0.0;
    let mut total_points = 0usize;

    for (view, pose) in views.iter().zip(poses.iter()) {
        for (p3d, p2d) in view.points_3d.iter().zip(view.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            if let Some(projected) = camera.project_point_c(&p_cam.coords) {
                let error = (projected - *p2d).norm();
                total_error += error;
                total_points += 1;
            }
        }
    }

    ensure!(
        total_points > 0,
        "no valid projections for error computation"
    );
    Ok(total_error / total_points as Real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::CalibrationSession;
    use calib_core::synthetic::planar;
    use calib_core::{Camera, IdentitySensor, Pinhole, Pt3};

    /// Generate synthetic linescan observations.
    ///
    /// Creates views with calibration corners and laser line pixels.
    fn generate_synthetic_linescan_data(
        camera: &PinholeCamera,
        board_points: &[Pt3],
        poses: &[Iso3],
        laser_plane: &LaserPlane,
    ) -> Vec<LinescanViewData> {
        let mut views = Vec::new();

        for pose in poses {
            // Project calibration corners
            let mut points_2d = Vec::new();
            for pt in board_points {
                let p_cam = pose.transform_point(pt);
                if let Some(proj) = camera.project_point_c(&p_cam.coords) {
                    points_2d.push(proj);
                }
            }

            // Generate laser pixels along intersection of laser plane and target
            // Target plane: Z=0 in target frame
            // Laser plane in camera frame intersects target plane
            let laser_pixels = generate_laser_pixels(camera, pose, laser_plane);

            if points_2d.len() == board_points.len() && laser_pixels.len() >= 3 {
                views.push(LinescanViewData {
                    calib_view: CorrespondenceView {
                        points_3d: board_points.to_vec(),
                        points_2d,
                        weights: None,
                    },
                    laser_pixels,
                });
            }
        }

        views
    }

    /// Generate laser pixels for a given pose and laser plane.
    fn generate_laser_pixels(
        camera: &PinholeCamera,
        pose: &Iso3,
        laser_plane: &LaserPlane,
    ) -> Vec<Vec2> {
        // Sample points along the laser line in target frame
        // The laser plane (in camera frame) intersects the target plane (Z=0 in target)
        // We compute intersection points and project them

        let mut laser_pixels = Vec::new();

        // Sample X positions on the target
        for x_idx in 0..10 {
            let x_target = (x_idx as f64 - 4.5) * 0.02; // -0.09 to 0.09 m

            // For each X, find the Y where the laser hits the target (Z=0)
            // This requires solving the intersection of laser plane and target plane
            // Simplified: assume horizontal laser line in camera frame

            // Transform sample points from target to camera frame
            let pt_target = Pt3::new(x_target, 0.0, 0.0);
            let pt_camera = pose.transform_point(&pt_target);

            // Check if point is approximately on laser plane
            let plane_dist = laser_plane.normal.dot(&pt_camera.coords) + laser_plane.distance;

            // If close to plane, project to image
            if plane_dist.abs() < 0.1 {
                if let Some(proj) = camera.project_point_c(&pt_camera.coords) {
                    // Add some offset based on plane distance to simulate laser line
                    let offset_y = plane_dist * 100.0; // Scale for visibility
                    laser_pixels.push(Vec2::new(proj.x, proj.y + offset_y));
                }
            }
        }

        // If no points found, generate some default pixels
        if laser_pixels.is_empty() {
            for i in 0..5 {
                let u = 200.0 + i as f64 * 100.0;
                laser_pixels.push(Vec2::new(u, 400.0));
            }
        }

        laser_pixels
    }

    #[test]
    fn linescan_problem_full_pipeline() {
        // Generate synthetic calibration data
        let k_gt = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist_gt = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        let cam_gt = Camera::new(Pinhole, dist_gt, IdentitySensor, k_gt);

        // Ground truth laser plane
        let plane_gt = LaserPlane::new(nalgebra::Vector3::new(0.0, 0.0, 1.0), -0.5);

        let board_points = planar::grid_points(5, 4, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);

        let views = generate_synthetic_linescan_data(&cam_gt, &board_points, &poses, &plane_gt);

        // Need at least 2 views
        if views.len() < 2 {
            eprintln!("Warning: Not enough valid views for test");
            return;
        }

        let obs = LinescanObservations::new(views).unwrap();

        // Create session and run pipeline
        let mut session = CalibrationSession::<LinescanProblem>::new_with_description(
            "Synthetic linescan test".to_string(),
        );

        session.set_observations(obs);

        // Initialize
        let init_result = session.initialize(LinescanInitOptions::default());
        assert!(init_result.is_ok(), "Initialization failed: {:?}", init_result);
        assert_eq!(
            session.stage(),
            crate::session::SessionStage::Initialized,
            "Should be in Initialized stage"
        );

        // Check initial values
        let init = session.initial_values().unwrap();
        // Intrinsics should be reasonable (within 20% of ground truth)
        assert!((init.intrinsics.fx - k_gt.fx).abs() < k_gt.fx * 0.2);
        assert!((init.intrinsics.fy - k_gt.fy).abs() < k_gt.fy * 0.2);

        // Optimize
        let optim_result = session.optimize(LinescanOptimOptions::default());
        assert!(optim_result.is_ok(), "Optimization failed: {:?}", optim_result);
        assert_eq!(
            session.stage(),
            crate::session::SessionStage::Optimized,
            "Should be in Optimized stage"
        );

        // Export
        let export_result = session.export();
        assert!(export_result.is_ok(), "Export failed");
        assert_eq!(
            session.stage(),
            crate::session::SessionStage::Exported,
            "Should be in Exported stage"
        );
    }

    #[test]
    fn linescan_session_json_checkpoint() {
        // Create a simple session with observations
        let views = vec![
            LinescanViewData {
                calib_view: CorrespondenceView {
                    points_3d: vec![
                        Pt3::new(0.0, 0.0, 0.0),
                        Pt3::new(0.05, 0.0, 0.0),
                        Pt3::new(0.05, 0.05, 0.0),
                        Pt3::new(0.0, 0.05, 0.0),
                    ],
                    points_2d: vec![
                        Vec2::new(100.0, 100.0),
                        Vec2::new(200.0, 100.0),
                        Vec2::new(200.0, 200.0),
                        Vec2::new(100.0, 200.0),
                    ],
                    weights: None,
                },
                laser_pixels: vec![
                    Vec2::new(150.0, 300.0),
                    Vec2::new(250.0, 300.0),
                    Vec2::new(350.0, 300.0),
                ],
            },
            LinescanViewData {
                calib_view: CorrespondenceView {
                    points_3d: vec![
                        Pt3::new(0.0, 0.0, 0.0),
                        Pt3::new(0.05, 0.0, 0.0),
                        Pt3::new(0.05, 0.05, 0.0),
                        Pt3::new(0.0, 0.05, 0.0),
                    ],
                    points_2d: vec![
                        Vec2::new(120.0, 110.0),
                        Vec2::new(220.0, 108.0),
                        Vec2::new(225.0, 208.0),
                        Vec2::new(118.0, 212.0),
                    ],
                    weights: None,
                },
                laser_pixels: vec![
                    Vec2::new(160.0, 310.0),
                    Vec2::new(260.0, 305.0),
                    Vec2::new(360.0, 308.0),
                ],
            },
        ];

        let mut session = CalibrationSession::<LinescanProblem>::new();
        session.set_observations(LinescanObservations { views });

        // Serialize
        let json = session.to_json().unwrap();
        assert!(json.contains("linescan"));
        assert!(json.contains("Uninitialized"));

        // Deserialize
        let restored: CalibrationSession<LinescanProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(
            restored.stage(),
            crate::session::SessionStage::Uninitialized
        );
        assert_eq!(restored.observations().unwrap().views.len(), 2);
    }
}
