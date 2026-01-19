//! ProblemType implementation for planar intrinsics calibration.

use anyhow::{ensure, Result};
use calib_core::{
    pinhole_camera_params, Camera, CorrespondenceView, IdentitySensor, Pinhole, Real,
};
use calib_optim::{
    PlanarDataset, PlanarIntrinsicsParams, PlanarIntrinsicsEstimate, PlanarIntrinsicsSolveOptions,
};

use crate::session::types::{ExportOptions, FilterOptions};
use crate::session::ProblemType;

use super::functions::planar_init_seed_from_views;

/// Planar intrinsics calibration problem (Zhang's method with distortion).
///
/// Estimates camera intrinsics (fx, fy, cx, cy, skew) and Brown-Conrady distortion
/// (k1, k2, k3, p1, p2) from observations of a planar calibration pattern.
pub struct PlanarIntrinsicsProblem;

impl ProblemType for PlanarIntrinsicsProblem {
    type Observations = PlanarIntrinsicsObservations;
    type InitialValues = PlanarIntrinsicsInitial;
    type OptimizedResults = PlanarIntrinsicsOptimized;
    type ExportReport = PlanarIntrinsicsReport;
    type InitOptions = PlanarIntrinsicsInitOptions;
    type OptimOptions = PlanarIntrinsicsOptimOptions;

    fn problem_name() -> &'static str {
        "planar_intrinsics"
    }

    fn initialize(
        obs: &Self::Observations,
        _opts: &Self::InitOptions,
    ) -> Result<Self::InitialValues> {
        ensure!(
            obs.views.len() >= 3,
            "need at least 3 views for planar initialization (got {})",
            obs.views.len()
        );

        let params = planar_init_seed_from_views(&obs.views)?;

        Ok(PlanarIntrinsicsInitial {
            intrinsics: params.intrinsics(),
            distortion: params.distortion(),
            poses: params.poses().to_vec(),
        })
    }

    fn optimize(
        obs: &Self::Observations,
        init: &Self::InitialValues,
        opts: &Self::OptimOptions,
    ) -> Result<Self::OptimizedResults> {
        use calib_optim::{optimize_planar_intrinsics, PlanarDataset, PlanarIntrinsicsParams};

        let dataset = PlanarDataset {
            views: obs.views.clone(),
        };

        let params =
            PlanarIntrinsicsParams::new_from_components(init.intrinsics, init.distortion, init.poses.clone())?;

        let result = optimize_planar_intrinsics(
            &dataset,
            params,
            opts.solve_opts.clone(),
            opts.backend_opts.clone(),
        )?;

        // Compute per-view errors for filtering
        let per_view_errors = compute_per_view_errors(
            &obs.views,
            &result.params.intrinsics(),
            &result.params.distortion(),
            result.params.poses(),
        )?;

        Ok(PlanarIntrinsicsOptimized {
            intrinsics: result.params.intrinsics(),
            distortion: result.params.distortion(),
            poses: result.params.poses().to_vec(),
            final_cost: result.report.final_cost,
            per_view_errors,
        })
    }

    fn filter_observations(
        obs: &Self::Observations,
        result: &Self::OptimizedResults,
        opts: &FilterOptions,
    ) -> Result<Self::Observations> {
        let camera = Camera::new(
            Pinhole,
            result.distortion,
            IdentitySensor,
            result.intrinsics,
        );

        let threshold = opts.max_reproj_error.unwrap_or(f64::INFINITY);
        let mut filtered_views = Vec::new();

        for (view, pose) in obs.views.iter().zip(&result.poses) {
            let mut filtered_3d = Vec::new();
            let mut filtered_2d = Vec::new();
            let mut filtered_weights = Vec::new();

            for (i, (p3d, p2d)) in view.points_3d.iter().zip(&view.points_2d).enumerate() {
                let p_cam = pose.transform_point(p3d);
                if let Some(projected) = camera.project_point_c(&p_cam.coords) {
                    let error = (projected - *p2d).norm();

                    if error <= threshold {
                        filtered_3d.push(*p3d);
                        filtered_2d.push(*p2d);
                        if let Some(ref w) = view.weights {
                            if i < w.len() {
                                filtered_weights.push(w[i]);
                            }
                        }
                    }
                }
            }

            // Check minimum points
            if filtered_3d.len() >= opts.min_points_per_view {
                filtered_views.push(CorrespondenceView {
                    points_3d: filtered_3d,
                    points_2d: filtered_2d,
                    weights: if view.weights.is_some() && !filtered_weights.is_empty() {
                        Some(filtered_weights)
                    } else {
                        None
                    },
                });
            } else if !opts.remove_sparse_views {
                // Keep original view if not removing sparse views
                filtered_views.push(view.clone());
            }
            // else: drop the view entirely
        }

        ensure!(!filtered_views.is_empty(), "filtering removed all views");

        Ok(PlanarIntrinsicsObservations {
            views: filtered_views,
        })
    }

    fn export(result: &Self::OptimizedResults, opts: &ExportOptions) -> Result<Self::ExportReport> {
        use calib_core::make_pinhole_camera;

        let camera = make_pinhole_camera(result.intrinsics, result.distortion);
        let camera_params = pinhole_camera_params(&camera);

        // Compute mean reprojection error
        let mean_error = if result.per_view_errors.is_empty() {
            0.0
        } else {
            result.per_view_errors.iter().sum::<Real>() / result.per_view_errors.len() as Real
        };

        Ok(PlanarIntrinsicsReport {
            camera: camera_params,
            final_cost: result.final_cost,
            mean_reproj_error: mean_error,
            poses: if opts.include_poses {
                Some(result.poses.clone())
            } else {
                None
            },
        })
    }
}

/// Compute per-view mean reprojection errors.
fn compute_per_view_errors(
    views: &[CorrespondenceView],
    intrinsics: &calib_core::FxFyCxCySkew<Real>,
    distortion: &calib_core::BrownConrady5<Real>,
    poses: &[calib_core::Iso3],
) -> Result<Vec<Real>> {
    let camera = Camera::new(Pinhole, *distortion, IdentitySensor, *intrinsics);

    let mut per_view_errors = Vec::with_capacity(views.len());

    for (view, pose) in views.iter().zip(poses) {
        let mut view_error = 0.0;
        let mut count = 0usize;

        for (p3d, p2d) in view.points_3d.iter().zip(&view.points_2d) {
            let p_cam = pose.transform_point(p3d);
            if let Some(projected) = camera.project_point_c(&p_cam.coords) {
                view_error += (projected - *p2d).norm();
                count += 1;
            }
        }

        if count > 0 {
            per_view_errors.push(view_error / count as Real);
        } else {
            per_view_errors.push(0.0);
        }
    }

    Ok(per_view_errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::CalibrationSession;
    use calib_core::{synthetic::planar, BrownConrady5, FxFyCxCySkew, Pt3, Vec2};

    fn make_test_camera() -> calib_core::PinholeCamera {
        use calib_core::make_pinhole_camera;

        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };
        make_pinhole_camera(k, dist)
    }

    #[test]
    fn planar_intrinsics_full_workflow() {
        let cam_gt = make_test_camera();

        let board_points = planar::grid_points(5, 4, 0.05);
        let poses = planar::poses_yaw_y_z(3, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new_with_description(
            "Synthetic planar intrinsics test".to_string(),
        );

        let obs_id = session.add_observations(PlanarIntrinsicsObservations { views });

        // Initialize
        let init_id = session
            .run_init(obs_id, PlanarIntrinsicsInitOptions::default())
            .expect("initialization should succeed");

        let init = session.get_initial_values(init_id).unwrap();
        assert!(init.poses.len() == 3);

        // Optimize
        let result_id = session
            .run_optimize(obs_id, init_id, PlanarIntrinsicsOptimOptions::default())
            .expect("optimization should succeed");

        let result = session.get_optimized_results(result_id).unwrap();
        assert!(result.final_cost < 1e-6, "final cost too high: {}", result.final_cost);

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

        assert!(report.poses.is_some());
        assert_eq!(report.poses.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn planar_intrinsics_filter_workflow() {
        let cam_gt = make_test_camera();

        let board_points = planar::grid_points(6, 5, 0.05);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let mut views = planar::project_views_all(&cam_gt, &board_points, &poses).unwrap();

        // Add outlier to first view
        if let Some(first_view) = views.first_mut() {
            if let Some(p) = first_view.points_2d.first_mut() {
                p.x += 50.0; // Large outlier
            }
        }

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        let obs_id = session.add_observations(PlanarIntrinsicsObservations { views });

        let init_id = session
            .run_init(obs_id, PlanarIntrinsicsInitOptions::default())
            .unwrap();
        let result_id = session
            .run_optimize(obs_id, init_id, PlanarIntrinsicsOptimOptions::default())
            .unwrap();

        // Filter with tight threshold
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
        // Outlier should be removed
        assert!(filtered.num_points() < 4 * 6 * 5);
    }

    #[test]
    fn session_json_checkpoint() {
        let views = vec![CorrespondenceView {
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
        }];

        let mut session = CalibrationSession::<PlanarIntrinsicsProblem>::new();
        session.add_observations(PlanarIntrinsicsObservations { views });

        let json = session.to_json().unwrap();
        assert!(json.contains("planar_intrinsics"));

        let restored: CalibrationSession<PlanarIntrinsicsProblem> =
            CalibrationSession::from_json(&json).unwrap();
        assert_eq!(restored.artifact_count(), 1);
    }
}
