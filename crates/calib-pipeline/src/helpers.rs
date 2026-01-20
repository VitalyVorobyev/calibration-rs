//! Helper functions for granular calibration workflows.
//!
//! This module provides individual operations that can be composed into custom
//! calibration pipelines. Unlike the session API which enforces a specific flow,
//! these functions give you full control over the calibration process.
//!
//! # Example: Custom Workflow
//!
//! ```ignore
//! use calib_pipeline::helpers::*;
//! use calib_pipeline::{BackendSolveOptions, PlanarIntrinsicsSolveOptions};
//!
//! // Step 1: Initialize intrinsics
//! let init_result = initialize_planar_intrinsics(&views, &init_opts)?;
//!
//! // Inspect intermediate results
//! println!("Initial fx: {}, fy: {}", init_result.intrinsics.fx, init_result.intrinsics.fy);
//!
//! // Step 2: Optimize after inspecting the seed
//! let optim_opts = PlanarIntrinsicsSolveOptions::default();
//! let backend_opts = BackendSolveOptions::default();
//! let final_result =
//!     optimize_planar_intrinsics_from_init(&dataset, &init_result, &optim_opts, &backend_opts)?;
//! println!("Final reprojection error: {:.2} px", final_result.mean_reproj_error);
//! ```

use anyhow::{ensure, Result};
use calib_core::{
    CorrespondenceView, FxFyCxCySkew, Iso3, Mat3, PinholeCamera, Pt2, Real, View, compute_mean_reproj_error, make_pinhole_camera
};
use calib_linear::iterative_intrinsics::{
    estimate_intrinsics_iterative, IterativeCalibView, IterativeIntrinsicsOptions,
};
use calib_optim::{
    optimize_planar_intrinsics, BackendSolveOptions, PlanarDataset, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, PlanarIntrinsicsEstimate
};

/// Optimize camera intrinsics from initial estimates using non-linear refinement.
pub fn optimize_planar_intrinsics_from_init(
    dataset: &PlanarDataset,
    init: &PlanarIntrinsicsParams,
    solve_opts: &PlanarIntrinsicsSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<PlanarIntrinsicsEstimate> {
    let views: Vec<CorrespondenceView> = dataset.views.iter().map(|v| v.obs.clone()).collect();

    // Optimization packs only [fx, fy, cx, cy]; enforce zero skew.
    let mut intrinsics = init.camera.k;
    intrinsics.skew = 0.0;

    // Recover pose seeds from homographies using provided intrinsics
    let homographies = planar_homographies_from_views(&views)?;
    let kmtx = k_matrix_from_intrinsics(&intrinsics);
    let poses0 = poses_from_homographies(&kmtx, &homographies)?;

    let planar_init =
        PlanarIntrinsicsParams::new_from_components(intrinsics, init.camera.dist, poses0)?;

    // Run optimization
    optimize_planar_intrinsics(
        dataset,
        planar_init,
        solve_opts.clone(),
        backend_opts.clone(),
    )
}

fn k_matrix_from_intrinsics(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

fn planar_homographies_from_views(views: &[CorrespondenceView]) -> Result<Vec<Mat3>> {
    use calib_linear::homography::dlt_homography;

    let mut homographies = Vec::with_capacity(views.len());
    for view in views {
        let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
        let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
        let h = dlt_homography(&board_2d, &pixel_2d)?;
        homographies.push(h);
    }
    Ok(homographies)
}

fn poses_from_homographies(kmtx: &Mat3, homographies: &[Mat3]) -> Result<Vec<Iso3>> {
    use calib_linear::planar_pose::estimate_planar_pose_from_h;

    homographies
        .iter()
        .map(|h| estimate_planar_pose_from_h(kmtx, h))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{make_pinhole_camera, synthetic::planar, BrownConrady5};
    use calib_linear::distortion_fit::DistortionFitOptions;

    fn generate_synthetic_views() -> PlanarDataset {
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
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).expect("projection");
        PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
    }

    #[test]
    fn initialize_planar_intrinsics_smoke_test() {
        let dataset = generate_synthetic_views();

        let opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: true,
                iters: 8,
            },
            zero_skew: true,
        };

        let result = estimate_intrinsics_iterative(&dataset, &opts).expect("init should succeed");

        assert!(result.k.fx > 0.0);
        assert!(result.k.fy > 0.0);
        assert!(result.dist.k1.abs() < 1.0);
    }

    #[test]
    fn optimize_from_init_improves_error() {
        let dataset = generate_synthetic_views();
        let views: Vec<CorrespondenceView> = dataset.views.iter().map(|v| v.obs.clone()).collect();

        let init_opts = IterativeIntrinsicsOptions {
            iterations: 1,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: true,
                iters: 8,
            },
            zero_skew: true,
        };

        let init_result =
            estimate_intrinsics_iterative(&views, &init_opts).expect("init should succeed");

        let solve_opts = PlanarIntrinsicsSolveOptions {
            fix_poses: vec![0],
            ..Default::default()
        };

        let backend_opts = BackendSolveOptions {
            max_iters: 20,
            ..Default::default()
        };

        let optim_result = optimize_planar_intrinsics_from_init(
            &dataset,
            &init_result,
            &solve_opts,
            &backend_opts,
        )
        .expect("optimization should succeed");

        assert!(
            optim_result.mean_reproj_error < 10.0,
            "final error too high: {}",
            optim_result.mean_reproj_error
        );

        assert!(
            optim_result.final_cost < 10.0,
            "final cost too high: {}",
            optim_result.final_cost
        );
    }
}
