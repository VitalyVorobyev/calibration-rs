pub mod handeye_single;
pub mod helpers;
pub mod session;

// Re-export key building block modules from calib-linear for custom workflows
// Users can access functions like: calib_pipeline::homography::dlt_homography()
pub use calib_linear::{
    distortion_fit, epipolar, extrinsics, handeye as handeye_linear, homography,
    iterative_intrinsics, linescan, planar_pose, pnp, triangulation, zhang_intrinsics,
};

// Re-export optimization problem builders from calib-optim
pub use calib_optim::{
    backend::BackendSolveOptions,
    handeye,
    ir::HandEyeMode,
    planar_intrinsics::{
        optimize_planar_intrinsics as optimize_planar_intrinsics_raw, PinholeCamera, PlanarDataset,
        PlanarIntrinsicsInit, PlanarIntrinsicsSolveOptions, PlanarViewObservations, RobustLoss,
    },
};

// Re-export problem modules as they become public in calib-optim.

use anyhow::{ensure, Context, Result};
use calib_core::{
    BrownConrady5, Camera, CameraParams, DistortionParams, FxFyCxCySkew, IdentitySensor,
    IntrinsicsParams, Iso3, Mat3, Pinhole, ProjectionParams, Pt2, Pt3, Real, SensorParams, Vec2,
};
// Note: These are now re-exported above for public API, not imported here
use calib_optim::planar_intrinsics::optimize_planar_intrinsics;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsConfig {
    /// Robust loss to use for residuals.
    pub robust_loss: Option<RobustLossConfig>,
    /// Maximum LM iterations (if `None`, use solver default).
    pub max_iters: Option<usize>,
    /// Fix fx during optimization.
    pub fix_fx: bool,
    /// Fix fy during optimization.
    pub fix_fy: bool,
    /// Fix cx during optimization.
    pub fix_cx: bool,
    /// Fix cy during optimization.
    pub fix_cy: bool,
    /// Fix poses by index.
    pub fix_poses: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustLossConfig {
    None,
    Huber { scale: Real },
    Cauchy { scale: Real },
    Arctan { scale: Real },
}

impl Default for PlanarIntrinsicsConfig {
    fn default() -> Self {
        Self {
            robust_loss: Some(RobustLossConfig::None),
            max_iters: None,
            fix_fx: false,
            fix_fy: false,
            fix_cx: false,
            fix_cy: false,
            fix_poses: None,
        }
    }
}

impl RobustLossConfig {
    pub fn to_loss(&self) -> RobustLoss {
        match *self {
            RobustLossConfig::None => RobustLoss::None,
            RobustLossConfig::Huber { scale } => RobustLoss::Huber { scale },
            RobustLossConfig::Cauchy { scale } => RobustLoss::Cauchy { scale },
            RobustLossConfig::Arctan { scale } => RobustLoss::Arctan { scale },
        }
    }
}

impl PlanarIntrinsicsConfig {
    pub fn solve_options(&self) -> PlanarIntrinsicsSolveOptions {
        PlanarIntrinsicsSolveOptions {
            robust_loss: self
                .robust_loss
                .as_ref()
                .map(RobustLossConfig::to_loss)
                .unwrap_or(RobustLoss::None),
            fix_fx: self.fix_fx,
            fix_fy: self.fix_fy,
            fix_cx: self.fix_cx,
            fix_cy: self.fix_cy,
            fix_poses: self.fix_poses.clone().unwrap_or_default(),
            ..Default::default()
        }
    }

    pub fn solver_options(&self) -> BackendSolveOptions {
        let mut opts = BackendSolveOptions::default();
        if let Some(max_iters) = self.max_iters {
            opts.max_iters = max_iters;
        }
        opts
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarViewData {
    pub points_3d: Vec<Pt3>,
    pub points_2d: Vec<Vec2>,
    pub weights: Option<Vec<Real>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInput {
    pub views: Vec<PlanarViewData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsReport {
    pub camera: CameraParams,
    pub final_cost: Real,
}

pub(crate) fn make_pinhole_camera(
    k: FxFyCxCySkew<Real>,
    dist: BrownConrady5<Real>,
) -> PinholeCamera {
    Camera::new(Pinhole, dist, IdentitySensor, k)
}

pub(crate) fn pinhole_camera_params(camera: &PinholeCamera) -> CameraParams {
    CameraParams {
        projection: ProjectionParams::Pinhole,
        distortion: DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1: camera.dist.k1,
                k2: camera.dist.k2,
                k3: camera.dist.k3,
                p1: camera.dist.p1,
                p2: camera.dist.p2,
                iters: camera.dist.iters,
            },
        },
        sensor: SensorParams::Identity,
        intrinsics: IntrinsicsParams::FxFyCxCySkew {
            params: FxFyCxCySkew {
                fx: camera.k.fx,
                fy: camera.k.fy,
                cx: camera.k.cx,
                cy: camera.k.cy,
                skew: camera.k.skew,
            },
        },
    }
}

fn board_and_pixel_points(view: &PlanarViewData) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();

    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();

    (board_2d, pixel_2d)
}

pub(crate) fn k_matrix_from_intrinsics(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

pub(crate) fn planar_homographies_from_views(views: &[PlanarViewData]) -> Result<Vec<Mat3>> {
    use calib_linear::homography::dlt_homography;

    let mut homographies = Vec::with_capacity(views.len());
    for (idx, view) in views.iter().enumerate() {
        let (board_2d, pixel_2d) = board_and_pixel_points(view);
        let h = dlt_homography(&board_2d, &pixel_2d).with_context(|| {
            format!(
                "failed to compute homography for view {} (need >=4 well-conditioned points)",
                idx
            )
        })?;
        homographies.push(h);
    }
    Ok(homographies)
}

pub(crate) fn poses_from_homographies(kmtx: &Mat3, homographies: &[Mat3]) -> Result<Vec<Iso3>> {
    use calib_linear::planar_pose::estimate_planar_pose_from_h;

    homographies
        .iter()
        .enumerate()
        .map(|(idx, h)| {
            estimate_planar_pose_from_h(kmtx, h)
                .with_context(|| format!("failed to recover pose for view {}", idx))
        })
        .collect()
}

fn iterative_init_guess(
    views: &[PlanarViewData],
) -> Option<(FxFyCxCySkew<Real>, BrownConrady5<Real>)> {
    use calib_linear::iterative_intrinsics::{
        estimate_intrinsics_iterative, IterativeCalibView, IterativeIntrinsicsOptions,
    };

    if views.len() < 3 {
        return None;
    }

    let calib_views: Vec<IterativeCalibView> = views
        .iter()
        .map(|v| {
            let (board_2d, pixel_2d) = board_and_pixel_points(v);
            IterativeCalibView::new(board_2d, pixel_2d)
        })
        .collect();

    let opts = IterativeIntrinsicsOptions::default();
    match estimate_intrinsics_iterative(&calib_views, opts) {
        Ok(res) => Some((
            res.intrinsics,
            BrownConrady5 {
                iters: res.distortion.iters,
                ..res.distortion
            },
        )),
        Err(_) => None,
    }
}

pub(crate) fn planar_init_seed_from_views(
    views: &[PlanarViewData],
) -> Result<(PlanarIntrinsicsInit, PinholeCamera)> {
    use calib_linear::zhang_intrinsics::estimate_intrinsics_from_homographies;

    ensure!(
        views.len() >= 3,
        "need at least 3 views for planar initialization (got {})",
        views.len()
    );

    let homographies = planar_homographies_from_views(views)?;

    // Primary path: Zhang closed-form intrinsics (no distortion)
    let mut intrinsics = estimate_intrinsics_from_homographies(&homographies)
        .context("zhang intrinsics initialization failed")?;
    let mut distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: 8,
    };

    // Fallback: iterative intrinsics to capture distortion if Zhang is unstable
    if let Some((intr, dist)) = iterative_init_guess(views) {
        intrinsics = intr;
        distortion = dist;
    }

    // Compute pose seeds from homographies and intrinsics
    let kmtx = k_matrix_from_intrinsics(&intrinsics);
    let poses0 = poses_from_homographies(&kmtx, &homographies)?;

    let init = PlanarIntrinsicsInit::new(intrinsics, distortion, poses0)?;

    let camera = make_pinhole_camera(intrinsics, distortion);

    Ok((init, camera))
}

pub(crate) fn build_planar_dataset(input: &PlanarIntrinsicsInput) -> Result<PlanarDataset> {
    let mut observations = Vec::new();
    for (idx, view) in input.views.iter().enumerate() {
        ensure!(
            view.points_3d.len() == view.points_2d.len(),
            "view {} has mismatched 3D/2D points ({} vs {})",
            idx,
            view.points_3d.len(),
            view.points_2d.len()
        );
        ensure!(
            view.points_3d.len() >= 4,
            "view {} needs at least 4 points (got {})",
            idx,
            view.points_3d.len()
        );
        if let Some(weights) = view.weights.as_ref() {
            ensure!(
                weights.len() == view.points_3d.len(),
                "view {} weights must match point count",
                idx
            );
            ensure!(
                weights.iter().all(|w| *w >= 0.0),
                "view {} has negative weights",
                idx
            );
        }

        let obs = if let Some(weights) = view.weights.as_ref() {
            PlanarViewObservations::new_with_weights(
                view.points_3d.clone(),
                view.points_2d.clone(),
                weights.clone(),
            )?
        } else {
            PlanarViewObservations::new(view.points_3d.clone(), view.points_2d.clone())?
        };
        observations.push(obs);
    }

    PlanarDataset::new(observations).context("invalid planar dataset")
}

pub(crate) fn optimize_planar_intrinsics_with_init(
    dataset: PlanarDataset,
    init: PlanarIntrinsicsInit,
    config: &PlanarIntrinsicsConfig,
) -> Result<calib_optim::planar_intrinsics::PlanarIntrinsicsResult> {
    optimize_planar_intrinsics(
        dataset,
        init,
        config.solve_options(),
        config.solver_options(),
    )
}

pub fn run_planar_intrinsics(
    input: &PlanarIntrinsicsInput,
    config: &PlanarIntrinsicsConfig,
) -> Result<PlanarIntrinsicsReport> {
    ensure!(
        !input.views.is_empty(),
        "need at least one view for calibration"
    );

    let dataset = build_planar_dataset(input)?;
    let (init, _) = planar_init_seed_from_views(&input.views)?;
    let result = optimize_planar_intrinsics_with_init(dataset, init, config)?;

    let camera_cfg = pinhole_camera_params(&result.camera);

    Ok(PlanarIntrinsicsReport {
        camera: camera_cfg,
        final_cost: result.final_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handeye::{
        optimize_handeye, CameraViewObservations, HandEyeDataset, HandEyeInit, HandEyeSolveOptions,
        RigViewObservations,
    };
    use crate::helpers::{initialize_planar_intrinsics, optimize_planar_intrinsics_from_init};
    use calib_linear::distortion_fit::DistortionFitOptions;
    use calib_linear::handeye::estimate_handeye_dlt;
    use calib_linear::iterative_intrinsics::IterativeIntrinsicsOptions;
    use calib_optim::ir::HandEyeMode;
    use nalgebra::Translation3;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn zhang_initialization_recovers_intrinsics_seed() {
        let k_gt = FxFyCxCySkew {
            fx: 1250.0,
            fy: 1220.0,
            cx: 640.0,
            cy: 400.0,
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
        let cam_gt = make_pinhole_camera(k_gt, dist_gt);

        // Simple grid
        let nx = 6;
        let ny = 5;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for view_idx in 0..4 {
            let angle = 0.08 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.05 * view_idx as f64);
            let pose = Iso3::from_parts(translation.into(), rotation);

            let mut points_2d = Vec::new();
            for pw in &board_points {
                let pc = pose.transform_point(pw);
                let proj = cam_gt.project_point(&pc).unwrap();
                points_2d.push(Vec2::new(proj.x, proj.y));
            }

            views.push(PlanarViewData {
                points_3d: board_points.clone(),
                points_2d,
                weights: None,
            });
        }

        let (seed, _) = planar_init_seed_from_views(&views).expect("init should succeed");
        assert!((seed.intrinsics.fx - k_gt.fx).abs() < 30.0);
        assert!((seed.intrinsics.fy - k_gt.fy).abs() < 30.0);
        assert!((seed.intrinsics.cx - k_gt.cx).abs() < 25.0);
        assert!((seed.intrinsics.cy - k_gt.cy).abs() < 25.0);
    }

    fn intrinsics_from_params(cfg: &CameraParams) -> FxFyCxCySkew<Real> {
        match &cfg.intrinsics {
            IntrinsicsParams::FxFyCxCySkew { params } => *params,
        }
    }

    #[test]
    fn planar_intrinsics_pipeline_synthetic_recovers_intrinsics() {
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
        let cam_gt = make_pinhole_camera(k_gt, dist_gt);

        let nx = 5;
        let ny = 4;
        let spacing = 0.05_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for view_idx in 0..3 {
            let angle = 0.1 * (view_idx as f64);
            let axis = Vector3::new(0.0, 1.0, 0.0);
            let rotation = UnitQuaternion::from_scaled_axis(axis * angle);
            let translation = Vector3::new(0.0, 0.0, 0.6 + 0.1 * view_idx as f64);
            let pose = Iso3::from_parts(translation.into(), rotation);

            let mut points_2d = Vec::new();
            for pw in &board_points {
                let pc = pose.transform_point(pw);
                let proj = cam_gt.project_point(&pc).unwrap();
                points_2d.push(Vec2::new(proj.x, proj.y));
            }

            views.push(PlanarViewData {
                points_3d: board_points.clone(),
                points_2d,
                weights: None,
            });
        }

        let input = PlanarIntrinsicsInput { views };
        let config = PlanarIntrinsicsConfig::default();

        let report = run_planar_intrinsics(&input, &config).expect("pipeline should succeed");
        assert!(
            report.final_cost < 1e-6,
            "final cost too high: {}",
            report.final_cost
        );

        let ki = intrinsics_from_params(&report.camera);
        assert!((ki.fx - k_gt.fx).abs() < 20.0);
        assert!((ki.fy - k_gt.fy).abs() < 20.0);
        assert!((ki.cx - k_gt.cx).abs() < 20.0);
        assert!((ki.cy - k_gt.cy).abs() < 20.0);
    }

    #[test]
    fn handeye_pipeline_synthetic_recovers_handeye() {
        let k_gt = FxFyCxCySkew {
            fx: 820.0,
            fy: 800.0,
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
        let cam_gt = make_pinhole_camera(k_gt, dist_gt);

        // Hand-eye parameter is gripper-from-camera (T_GC) for EyeInHand.
        let handeye_gt = Iso3::from_parts(
            Translation3::new(0.02, -0.015, 0.12),
            UnitQuaternion::from_euler_angles(0.06, -0.02, 0.05),
        );

        let robot_poses = vec![
            Iso3::from_parts(
                Translation3::new(0.0, 0.0, -1.0),
                UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
            ),
            Iso3::from_parts(
                Translation3::new(0.08, -0.05, -0.95),
                UnitQuaternion::from_euler_angles(0.12, 0.05, 0.1),
            ),
            Iso3::from_parts(
                Translation3::new(-0.07, 0.06, -1.08),
                UnitQuaternion::from_euler_angles(-0.1, 0.14, -0.12),
            ),
            Iso3::from_parts(
                Translation3::new(0.05, 0.09, -1.05),
                UnitQuaternion::from_euler_angles(0.2, -0.08, 0.06),
            ),
            Iso3::from_parts(
                Translation3::new(-0.06, -0.04, -0.92),
                UnitQuaternion::from_euler_angles(-0.16, 0.1, 0.12),
            ),
        ];

        let nx = 6;
        let ny = 5;
        let spacing = 0.04_f64;
        let mut board_points = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                board_points.push(Pt3::new(i as f64 * spacing, j as f64 * spacing, 0.0));
            }
        }

        let mut views = Vec::new();
        for robot_pose in &robot_poses {
            let cam_to_target = handeye_gt.inverse() * robot_pose.inverse();
            let mut points_2d = Vec::new();
            for pw in &board_points {
                let pc = cam_to_target.transform_point(pw);
                let proj = cam_gt.project_point(&pc).unwrap();
                points_2d.push(Vec2::new(proj.x, proj.y));
            }

            views.push(PlanarViewData {
                points_3d: board_points.clone(),
                points_2d,
                weights: None,
            });
        }

        let init_opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
            zero_skew: true,
        };
        let init = initialize_planar_intrinsics(&views, &init_opts).unwrap();

        let solve_opts = PlanarIntrinsicsSolveOptions {
            fix_k3: true,
            ..Default::default()
        };
        let optim =
            optimize_planar_intrinsics_from_init(&views, &init, &solve_opts, &Default::default())
                .unwrap();

        let cam_in_target: Vec<Iso3> = optim.poses.iter().map(|pose| pose.inverse()).collect();
        let handeye_init = estimate_handeye_dlt(&robot_poses, &cam_in_target, 1.0).unwrap();

        let target_poses: Vec<Iso3> = robot_poses
            .iter()
            .zip(&optim.poses)
            .map(|(base_to_gripper, cam_to_target)| {
                base_to_gripper.clone() * handeye_init * cam_to_target.clone()
            })
            .collect();

        let mut rig_views = Vec::new();
        for (robot_pose, view) in robot_poses.iter().zip(&views) {
            let obs = CameraViewObservations::new(view.points_3d.clone(), view.points_2d.clone())
                .unwrap();
            rig_views.push(RigViewObservations {
                cameras: vec![Some(obs)],
                robot_pose: robot_pose.clone(),
            });
        }

        let dataset = HandEyeDataset::new(rig_views, 1, HandEyeMode::EyeInHand).unwrap();
        let mut handeye_intrinsics = optim.intrinsics;
        handeye_intrinsics.skew = 0.0;
        let init = HandEyeInit {
            intrinsics: vec![handeye_intrinsics],
            distortion: vec![optim.distortion],
            cam_to_rig: vec![Iso3::identity()],
            handeye: handeye_init,
            target_poses,
        };

        let mut opts = HandEyeSolveOptions::default();
        opts.fix_fx = true;
        opts.fix_fy = true;
        opts.fix_cx = true;
        opts.fix_cy = true;
        opts.fix_k1 = true;
        opts.fix_k2 = true;
        opts.fix_k3 = true;
        opts.fix_p1 = true;
        opts.fix_p2 = true;
        opts.fix_extrinsics = vec![true];
        opts.fix_target_poses = vec![0];

        let result = optimize_handeye(dataset, init, opts, BackendSolveOptions::default()).unwrap();

        let t_err = (result.handeye.translation.vector - handeye_gt.translation.vector).norm();
        let r_final = result.handeye.rotation.to_rotation_matrix();
        let r_gt = handeye_gt.rotation.to_rotation_matrix();
        let r_diff = r_final.transpose() * r_gt;
        let angle = ((r_diff.matrix().trace() - 1.0) * 0.5)
            .clamp(-1.0, 1.0)
            .acos();

        assert!(
            t_err < 1e-3,
            "handeye translation error too large: {}",
            t_err
        );
        assert!(angle < 1e-3, "handeye rotation error too large: {}", angle);
    }

    #[test]
    fn config_json_roundtrip() {
        let config = PlanarIntrinsicsConfig {
            robust_loss: Some(RobustLossConfig::Huber { scale: 2.5 }),
            max_iters: Some(80),
            fix_fx: true,
            fix_fy: false,
            fix_cx: false,
            fix_cy: true,
            fix_poses: Some(vec![0, 2]),
        };

        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(
            json.contains("Huber") && json.contains("2.5"),
            "json missing expected content: {}",
            json
        );

        let de: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.max_iters, config.max_iters);
        assert_eq!(de.fix_fx, config.fix_fx);
        assert_eq!(de.fix_cy, config.fix_cy);
        match (de.robust_loss, config.robust_loss) {
            (
                Some(RobustLossConfig::Huber { scale: d1 }),
                Some(RobustLossConfig::Huber { scale: d2 }),
            ) => assert!((d1 - d2).abs() < 1e-12),
            other => panic!("mismatch in losses: {:?}", other),
        }
    }

    #[test]
    fn input_json_roundtrip() {
        let input = PlanarIntrinsicsInput {
            views: vec![PlanarViewData {
                points_3d: vec![
                    Pt3::new(0.0, 0.0, 0.0),
                    Pt3::new(1.0, 0.0, 0.0),
                    Pt3::new(1.0, 1.0, 0.0),
                    Pt3::new(0.0, 1.0, 0.0),
                ],
                points_2d: vec![
                    Vec2::new(100.0, 100.0),
                    Vec2::new(200.0, 100.0),
                    Vec2::new(200.0, 200.0),
                    Vec2::new(100.0, 200.0),
                ],
                weights: Some(vec![1.0, 1.0, 0.5, 0.5]),
            }],
        };

        let json = serde_json::to_string_pretty(&input).unwrap();
        let de: PlanarIntrinsicsInput = serde_json::from_str(&json).unwrap();

        assert_eq!(de.views.len(), input.views.len());
        for (view_a, view_b) in de.views.iter().zip(input.views.iter()) {
            assert_eq!(view_a.points_3d.len(), view_b.points_3d.len());
            assert_eq!(view_a.points_2d.len(), view_b.points_2d.len());
            assert_eq!(view_a.weights.as_ref().unwrap().len(), 4);
            for (a, b) in view_a.points_3d.iter().zip(view_b.points_3d.iter()) {
                assert!((a.x - b.x).abs() < 1e-12);
                assert!((a.y - b.y).abs() < 1e-12);
                assert!((a.z - b.z).abs() < 1e-12);
            }
            for (a, b) in view_a.points_2d.iter().zip(view_b.points_2d.iter()) {
                assert!((a.x - b.x).abs() < 1e-12);
                assert!((a.y - b.y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn report_json_roundtrip() {
        let cam = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5 {
                k1: 0.0,
                k2: 0.0,
                k3: 0.0,
                p1: 0.0,
                p2: 0.0,
                iters: 8,
            },
        );
        let report = PlanarIntrinsicsReport {
            camera: pinhole_camera_params(&cam),
            final_cost: 1e-8,
        };

        let json = serde_json::to_string_pretty(&report).unwrap();
        let de: PlanarIntrinsicsReport = serde_json::from_str(&json).unwrap();

        let ki_de = intrinsics_from_params(&de.camera);
        let ki_report = intrinsics_from_params(&report.camera);

        assert!((ki_de.fx - ki_report.fx).abs() < 1e-12);
        assert!((ki_de.fy - ki_report.fy).abs() < 1e-12);
        assert!((ki_de.cx - ki_report.cx).abs() < 1e-12);
        assert!((ki_de.cy - ki_report.cy).abs() < 1e-12);

        match (&de.camera.distortion, &report.camera.distortion) {
            (
                DistortionParams::BrownConrady5 { params: a },
                DistortionParams::BrownConrady5 { params: b },
            ) => {
                assert!((a.k1 - b.k1).abs() < 1e-12);
                assert!((a.k2 - b.k2).abs() < 1e-12);
                assert!((a.p1 - b.p1).abs() < 1e-12);
                assert!((a.p2 - b.p2).abs() < 1e-12);
                assert!((a.k3 - b.k3).abs() < 1e-12);
            }
            other => panic!("distortion mismatch: {:?}", other),
        }

        assert!((de.final_cost - report.final_cost).abs() < 1e-12);
    }
}
