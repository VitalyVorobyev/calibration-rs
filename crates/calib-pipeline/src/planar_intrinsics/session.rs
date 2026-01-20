//! [`ProblemType`] implementation for planar intrinsics calibration.

use anyhow::{ensure, Result};
use calib_core::{CorrespondenceView, PlanarDataset, View};
use calib_optim::{optimize_planar_intrinsics, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams};

use crate::session::types::{ExportOptions, FilterOptions};
use crate::session::ProblemType;

use super::functions::{planar_init_seed_from_views, PlanarIntrinsicsConfig};

/// Planar intrinsics calibration problem (Zhang's method with distortion).
pub struct PlanarIntrinsicsProblem;

impl ProblemType for PlanarIntrinsicsProblem {
    type Observations = PlanarDataset;
    type InitialValues = PlanarIntrinsicsParams;
    type OptimizedResults = PlanarIntrinsicsEstimate;
    type ExportReport = PlanarIntrinsicsEstimate;
    type InitOptions = PlanarIntrinsicsConfig;
    type OptimOptions = PlanarIntrinsicsConfig;

    fn problem_name() -> &'static str {
        "planar_intrinsics"
    }

    fn initialize(
        obs: &Self::Observations,
        opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        planar_init_seed_from_views(obs, opts.init_opts)
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        optimize_planar_intrinsics(obs, init, opts.optim_options(), opts.solver_options())
    }

    fn filter_observations(
        obs: &Self::Observations,
        result: &Self::OptimizedResults,
        opts: &FilterOptions,
    ) -> Result<Self::Observations> {
        ensure!(
            opts.min_points_per_view >= 4,
            "min_points_per_view must be >=4 for planar intrinsics"
        );

        let poses = result.params.poses();
        ensure!(
            obs.views.len() == poses.len(),
            "pose count ({}) must match view count ({})",
            poses.len(),
            obs.views.len()
        );

        let threshold = opts.max_reproj_error.unwrap_or(f64::INFINITY);
        let camera = &result.params.camera;

        let mut filtered_views = Vec::new();
        for (view, pose) in obs.views.iter().zip(poses) {
            let mut points_3d = Vec::new();
            let mut points_2d = Vec::new();
            let mut weights = view.obs.weights.as_ref().map(|_| Vec::<f64>::new());

            for (i, (p3d, p2d)) in view
                .obs
                .points_3d
                .iter()
                .zip(view.obs.points_2d.iter())
                .enumerate()
            {
                let p_cam = pose.transform_point(p3d);
                let Some(projected) = camera.project_point_c(&p_cam.coords) else {
                    continue;
                };

                let error = (projected - *p2d).norm();
                if error <= threshold {
                    points_3d.push(*p3d);
                    points_2d.push(*p2d);
                    if let Some(ref mut w) = weights {
                        w.push(view.obs.weight(i));
                    }
                }
            }

            if points_3d.len() >= opts.min_points_per_view {
                let obs = if let Some(w) = weights {
                    CorrespondenceView::new_with_weights(points_3d, points_2d, w)?
                } else {
                    CorrespondenceView::new(points_3d, points_2d)?
                };
                filtered_views.push(View::without_meta(obs));
            } else if !opts.remove_sparse_views {
                filtered_views.push(view.clone());
            }
        }

        PlanarDataset::new(filtered_views)
    }

    fn export(
        result: &Self::OptimizedResults,
        _opts: &ExportOptions,
    ) -> Result<Self::ExportReport> {
        Ok(result.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::CalibrationSession;
    use calib_core::{synthetic::planar, BrownConrady5, FxFyCxCySkew};

    fn make_test_camera() -> calib_core::PinholeCamera {
        use calib_core::make_pinhole_camera;

        make_pinhole_camera(
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
        )
    }

    fn num_points(dataset: &PlanarDataset) -> usize {
        dataset.views.iter().map(|v| v.obs.len()).sum()
    }

    #[test]
    fn planar_intrinsics_full_workflow() {
        let cam_gt = make_test_camera();

        let board_points = planar::grid_points(5, 4, 0.05);
        let poses = planar::poses_yaw_y_z(3, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        let dataset =
            PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
            "Synthetic planar intrinsics test".to_string(),
        );

        let obs_id = session.add_observations(dataset);

        // Initialize
        let config = PlanarIntrinsicsConfig::default();
        let init_id = session
            .run_init(obs_id, config.clone())
            .expect("initialization should succeed");

        let init = session.get_initial_values(init_id).unwrap();
        assert_eq!(init.poses().len(), 3);

        // Optimize
        let result_id = session
            .run_optimize(obs_id, init_id, config)
            .expect("optimization should succeed");

        let result = session.get_optimized_results(result_id).unwrap();
        assert!(
            result.report.final_cost < 1e-6,
            "final cost too high: {}",
            result.report.final_cost
        );

        // Export
        let report = session
            .run_export(
                result_id,
                ExportOptions {
                    include_poses: true,
                    include_residuals: false,
                },
            )
            .expect("export should succeed");

        assert_eq!(report.params.poses().len(), 3);
    }

    #[test]
    fn planar_intrinsics_filter_workflow() {
        let cam_gt = make_test_camera();

        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let mut views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        // Add an outlier to the first view.
        if let Some(first_view) = views.first_mut() {
            if let Some(p) = first_view.points_2d.first_mut() {
                p.x += 50.0;
            }
        }

        let dataset =
            PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();
        let original_points = num_points(&dataset);

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        let obs_id = session.add_observations(dataset);

        let config = PlanarIntrinsicsConfig::default();
        let init_id = session.run_init(obs_id, config.clone()).unwrap();
        let result_id = session.run_optimize(obs_id, init_id, config).unwrap();

        // Filter with tight threshold.
        let filtered_id = session
            .run_filter_obs(
                obs_id,
                result_id,
                FilterOptions {
                    max_reproj_error: Some(5.0),
                    min_points_per_view: 4,
                    remove_sparse_views: true,
                },
            )
            .unwrap();

        let filtered = session.get_observations(filtered_id).unwrap();
        assert!(num_points(filtered) < original_points);
    }

    #[test]
    fn session_json_checkpoint() {
        use calib_core::{NoMeta, Pt2, Pt3};

        let dataset = PlanarDataset::new(vec![View::new(
            CorrespondenceView {
                points_3d: vec![
                    Pt3::new(0.0, 0.0, 0.0),
                    Pt3::new(0.05, 0.0, 0.0),
                    Pt3::new(0.05, 0.05, 0.0),
                    Pt3::new(0.0, 0.05, 0.0),
                ],
                points_2d: vec![
                    Pt2::new(100.0, 100.0),
                    Pt2::new(200.0, 100.0),
                    Pt2::new(200.0, 200.0),
                    Pt2::new(100.0, 200.0),
                ],
                weights: None,
            },
            NoMeta {},
        )])
        .unwrap();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.add_observations(dataset);

        let json = session.to_json().unwrap();
        assert!(json.contains("planar_intrinsics"));

        let restored: CalibrationSession<PlanarIntrinsicsProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(restored.artifact_count(), 1);
    }
}
