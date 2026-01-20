use calib_core::{
    make_pinhole_camera, pinhole_camera_params, BrownConrady5, CorrespondenceView, FxFyCxCySkew,
    Iso3, Mat3, PinholeCamera, Pt2, Real,
};

use calib_linear::prelude::*;

use calib_optim::{
    optimize_planar_intrinsics, BackendSolveOptions, PlanarDataset, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions,
};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanarIntrinsicsConfig {
    #[serde(default)]
    pub solve_opts: PlanarIntrinsicsSolveOptions,
    #[serde(default)]
    pub backend_opts: BackendSolveOptions,
}

impl PlanarIntrinsicsConfig {
    pub fn solve_options(&self) -> PlanarIntrinsicsSolveOptions {
        self.solve_opts.clone()
    }

    pub fn solver_options(&self) -> BackendSolveOptions {
        self.backend_opts.clone()
    }
}

// Note: PlanarIntrinsicsReport is defined in types.rs and re-exported from mod.rs
use super::types::PlanarIntrinsicsReport;

fn board_and_pixel_points(view: &CorrespondenceView) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
    (board_2d, pixel_2d)
}

fn k_matrix_from_intrinsics(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

fn planar_homographies_from_views(views: &[CorrespondenceView]) -> Result<Vec<Mat3>> {
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

fn poses_from_homographies(kmtx: &Mat3, homographies: &[Mat3]) -> Result<Vec<Iso3>> {
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
    views: &[CorrespondenceView],
) -> Option<(FxFyCxCySkew<Real>, BrownConrady5<Real>)> {
    if views.len() < 3 {
        return None;
    }

    let calib_views: Vec<IterativeCalibView> = views
        .iter()
        .map(|v| {
            let (board_2d, pixel_2d) = board_and_pixel_points(v);
            IterativeCalibView::new(board_2d, pixel_2d)
        })
        .collect::<Result<Vec<_>>>()
        .ok()?;

    let opts = IterativeIntrinsicsOptions::default();
    match estimate_intrinsics_iterative(&calib_views, opts) {
        Ok(cam) => Some((cam.k, cam.dist)),
        Err(_) => None,
    }
}

pub fn planar_init_seed_from_views(views: &[CorrespondenceView]) -> Result<PlanarIntrinsicsParams> {
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

    let camera = make_pinhole_camera(intrinsics, distortion);
    let init = PlanarIntrinsicsParams::new(camera, poses0)?;

    Ok(init)
}

fn optimize_planar_intrinsics_with_init(
    dataset: &PlanarDataset,
    init: PlanarIntrinsicsParams,
    config: &PlanarIntrinsicsConfig,
) -> Result<calib_optim::PlanarIntrinsicsEstimate> {
    optimize_planar_intrinsics(
        dataset,
        init,
        config.solve_options(),
        config.solver_options(),
    )
}

pub fn run_planar_intrinsics(
    dataset: &PlanarDataset,
    config: &PlanarIntrinsicsConfig,
) -> Result<PlanarIntrinsicsReport> {
    ensure!(
        !dataset.views.is_empty(),
        "need at least one view for calibration"
    );

    let init = planar_init_seed_from_views(&dataset.views)?;
    let result = optimize_planar_intrinsics_with_init(dataset, init, config)?;

    let camera_cfg = pinhole_camera_params(&result.params.camera);

    // Compute mean reprojection error
    let mean_reproj_error = compute_mean_reproj_error(
        &dataset.views,
        &result.params.intrinsics(),
        &result.params.distortion(),
        result.params.poses(),
    )?;

    Ok(PlanarIntrinsicsReport {
        camera: camera_cfg,
        final_cost: result.report.final_cost,
        mean_reproj_error,
        poses: Some(result.params.poses().to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{
        synthetic::planar, CameraParams, DistortionParams, IntrinsicsFixMask, IntrinsicsParams,
        Pt3, Vec2,
    };
    use calib_optim::RobustLoss;

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

        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.08, 0.6, 0.05);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        let seed = planar_init_seed_from_views(&views).expect("init should succeed");
        let k = seed.intrinsics();
        assert!((k.fx - k_gt.fx).abs() < 30.0);
        assert!((k.fy - k_gt.fy).abs() < 30.0);
        assert!((k.cx - k_gt.cx).abs() < 25.0);
        assert!((k.cy - k_gt.cy).abs() < 25.0);
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

        let board_points = planar::grid_points(5, 4, 0.05);
        let poses = planar::poses_yaw_y_z(3, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        let input = PlanarDataset { views };
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

    // TODO: Re-enable after handeye module is updated
    // #[test]
    // fn handeye_pipeline_synthetic_recovers_handeye() { ... }

    #[test]
    fn config_json_roundtrip() {
        let mut config = PlanarIntrinsicsConfig::default();
        config.solve_opts.robust_loss = RobustLoss::Huber { scale: 2.5 };
        config.solve_opts.fix_intrinsics = IntrinsicsFixMask {
            fx: true,
            cy: true,
            ..Default::default()
        };
        config.solve_opts.fix_poses = vec![0, 2];
        config.backend_opts.max_iters = 80;

        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(
            json.contains("Huber") && json.contains("2.5"),
            "json missing expected content: {}",
            json
        );

        let de: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.backend_opts.max_iters, 80);
        assert!(de.solve_opts.fix_intrinsics.fx);
        assert!(de.solve_opts.fix_intrinsics.cy);
        assert_eq!(de.solve_opts.fix_poses, vec![0, 2]);
        match de.solve_opts.robust_loss {
            RobustLoss::Huber { scale } => assert!((scale - 2.5).abs() < 1e-12),
            other => panic!("unexpected robust_loss: {other:?}"),
        };
    }

    #[test]
    fn input_json_roundtrip() {
        let input = PlanarDataset {
            views: vec![CorrespondenceView {
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
        let de: PlanarDataset = serde_json::from_str(&json).unwrap();

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
            mean_reproj_error: 0.5,
            poses: None,
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
