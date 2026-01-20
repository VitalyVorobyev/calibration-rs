use calib_core::{CorrespondenceView, FxFyCxCySkew, Iso3, Mat3, PlanarDataset, Pt2, Real};

use calib_linear::prelude::*;

use calib_optim::{
    optimize_planar_intrinsics, BackendSolveOptions, PlanarIntrinsicsEstimate,
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
};

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlanarIntrinsicsConfig {
    pub init_opts: IterativeIntrinsicsOptions,
    #[serde(default)]
    pub optim_opts: PlanarIntrinsicsSolveOptions,
    #[serde(default)]
    pub backend_opts: BackendSolveOptions,
}

impl PlanarIntrinsicsConfig {
    pub fn optim_options(&self) -> PlanarIntrinsicsSolveOptions {
        self.optim_opts.clone()
    }

    pub fn solver_options(&self) -> BackendSolveOptions {
        self.backend_opts.clone()
    }
}

fn board_and_pixel_points(view: &CorrespondenceView) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
    (board_2d, pixel_2d)
}

fn k_matrix_from_intrinsics(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

fn planar_homographies_from_views(dataset: &PlanarDataset) -> Result<Vec<Mat3>> {
    let mut homographies = Vec::with_capacity(dataset.views.len());
    for (idx, view) in dataset.views.iter().enumerate() {
        let (board_2d, pixel_2d) = board_and_pixel_points(&view.obs);
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

pub fn planar_init_seed_from_views(
    dataset: &PlanarDataset,
    opts: IterativeIntrinsicsOptions,
) -> Result<PlanarIntrinsicsParams> {
    ensure!(
        dataset.views.len() >= 3,
        "need at least 3 views for planar initialization (got {})",
        dataset.views.len()
    );

    let camera = estimate_intrinsics_iterative(dataset, opts)?;
    let homographies = planar_homographies_from_views(dataset)?;

    // Compute pose seeds from homographies and intrinsics
    let kmtx = k_matrix_from_intrinsics(&camera.k);
    let camera_se3_target = poses_from_homographies(&kmtx, &homographies)?;

    PlanarIntrinsicsParams::new(camera, camera_se3_target)
}

pub fn run_planar_intrinsics(
    dataset: &PlanarDataset,
    config: &PlanarIntrinsicsConfig,
) -> Result<PlanarIntrinsicsEstimate> {
    ensure!(
        !dataset.views.is_empty(),
        "need at least one view for calibration"
    );

    let init = planar_init_seed_from_views(dataset, config.init_opts)?;
    optimize_planar_intrinsics(
        dataset,
        &init,
        config.optim_options(),
        config.solver_options(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{
        make_pinhole_camera, synthetic::planar, BrownConrady5, IntrinsicsFixMask, NoMeta, Pt2, Pt3,
        View,
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
        let dataset =
            PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();

        let opts = IterativeIntrinsicsOptions {
            iterations: 5,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: true,
                iters: 8,
            },
            zero_skew: true,
        };
        let seed = estimate_intrinsics_iterative(&dataset, opts).expect("init should succeed");
        let k = seed.k;
        assert!((k.fx - k_gt.fx).abs() < 30.0);
        assert!((k.fy - k_gt.fy).abs() < 30.0);
        assert!((k.cx - k_gt.cx).abs() < 25.0);
        assert!((k.cy - k_gt.cy).abs() < 25.0);
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

        let input =
            PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();
        let config = PlanarIntrinsicsConfig::default();

        let result = run_planar_intrinsics(&input, &config).expect("pipeline should succeed");
        assert!(
            result.report.final_cost < 1e-6,
            "final cost too high: {}",
            result.report.final_cost
        );

        let ki = result.params.intrinsics();
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
        config.optim_opts.robust_loss = RobustLoss::Huber { scale: 2.5 };
        config.optim_opts.fix_intrinsics = IntrinsicsFixMask {
            fx: true,
            cy: true,
            ..Default::default()
        };
        config.optim_opts.fix_poses = vec![0, 2];
        config.backend_opts.max_iters = 80;

        let json = serde_json::to_string_pretty(&config).unwrap();
        assert!(
            json.contains("Huber") && json.contains("2.5"),
            "json missing expected content: {}",
            json
        );

        let de: PlanarIntrinsicsConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(de.backend_opts.max_iters, 80);
        assert!(de.optim_opts.fix_intrinsics.fx);
        assert!(de.optim_opts.fix_intrinsics.cy);
        assert_eq!(de.optim_opts.fix_poses, vec![0, 2]);
        match de.optim_opts.robust_loss {
            RobustLoss::Huber { scale } => assert!((scale - 2.5).abs() < 1e-12),
            other => panic!("unexpected robust_loss: {other:?}"),
        };
    }

    #[test]
    fn input_json_roundtrip() {
        let input = PlanarDataset {
            views: vec![View::new(
                CorrespondenceView {
                    points_3d: vec![
                        Pt3::new(0.0, 0.0, 0.0),
                        Pt3::new(1.0, 0.0, 0.0),
                        Pt3::new(1.0, 1.0, 0.0),
                        Pt3::new(0.0, 1.0, 0.0),
                    ],
                    points_2d: vec![
                        Pt2::new(100.0, 100.0),
                        Pt2::new(200.0, 100.0),
                        Pt2::new(200.0, 200.0),
                        Pt2::new(100.0, 200.0),
                    ],
                    weights: Some(vec![1.0, 1.0, 0.5, 0.5]),
                },
                NoMeta {},
            )],
        };

        let json = serde_json::to_string_pretty(&input).unwrap();
        let de: PlanarDataset = serde_json::from_str(&json).unwrap();

        assert_eq!(de.views.len(), input.views.len());
        for (view_a, view_b) in de.views.iter().zip(input.views.iter()) {
            assert_eq!(view_a.obs.points_3d.len(), view_b.obs.points_3d.len());
            assert_eq!(view_a.obs.points_2d.len(), view_b.obs.points_2d.len());
            assert_eq!(view_a.obs.weights.as_ref().unwrap().len(), 4);
            for (a, b) in view_a.obs.points_3d.iter().zip(view_b.obs.points_3d.iter()) {
                assert!((a.x - b.x).abs() < 1e-12);
                assert!((a.y - b.y).abs() < 1e-12);
                assert!((a.z - b.z).abs() < 1e-12);
            }
            for (a, b) in view_a.obs.points_2d.iter().zip(view_b.obs.points_2d.iter()) {
                assert!((a.x - b.x).abs() < 1e-12);
                assert!((a.y - b.y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn estimate_json_roundtrip() {
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
        let estimate = PlanarIntrinsicsEstimate {
            params: PlanarIntrinsicsParams::new(cam, vec![Iso3::identity()]).unwrap(),
            report: calib_optim::SolveReport { final_cost: 1e-8 },
            mean_reproj_error: 0.5,
        };

        let json = serde_json::to_string_pretty(&estimate).unwrap();
        let de: PlanarIntrinsicsEstimate = serde_json::from_str(&json).unwrap();

        assert!((de.report.final_cost - estimate.report.final_cost).abs() < 1e-12);
        assert!((de.mean_reproj_error - estimate.mean_reproj_error).abs() < 1e-12);

        let ki_de = de.params.intrinsics();
        let ki_est = estimate.params.intrinsics();
        assert!((ki_de.fx - ki_est.fx).abs() < 1e-12);
        assert!((ki_de.fy - ki_est.fy).abs() < 1e-12);
        assert!((ki_de.cx - ki_est.cx).abs() < 1e-12);
        assert!((ki_de.cy - ki_est.cy).abs() < 1e-12);

        let (a, b) = (&de.params.camera.dist, &estimate.params.camera.dist);
        assert!((a.k1 - b.k1).abs() < 1e-12);
        assert!((a.k2 - b.k2).abs() < 1e-12);
        assert!((a.p1 - b.p1).abs() < 1e-12);
        assert!((a.p2 - b.p2).abs() < 1e-12);
        assert!((a.k3 - b.k3).abs() < 1e-12);
    }
}
