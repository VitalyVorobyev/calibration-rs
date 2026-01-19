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
//! let final_result = optimize_planar_intrinsics_from_init(
//!     &views,
//!     &init_result,
//!     &optim_opts,
//!     &backend_opts
//! )?;
//! println!("Final reprojection error: {:.2} px", final_result.mean_reproj_error);
//! ```

use anyhow::Result;
use calib_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, IdentitySensor, Iso3, Mat3, Pinhole,
    Pt2, Real,
};
use calib_linear::iterative_intrinsics::{
    estimate_intrinsics_iterative, IterativeCalibView, IterativeIntrinsicsOptions,
};
use calib_optim::{
    optimize_planar_intrinsics, BackendSolveOptions, PlanarDataset, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions,
};
use serde::{Deserialize, Serialize};

/// Result from linear intrinsics initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsInitResult {
    /// Estimated camera intrinsics.
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Estimated Brown-Conrady distortion.
    pub distortion: BrownConrady5<Real>,
}

/// Result from non-linear intrinsics optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanarIntrinsicsOptimResult {
    /// Optimized camera intrinsics.
    pub intrinsics: FxFyCxCySkew<Real>,
    /// Optimized Brown-Conrady distortion.
    pub distortion: BrownConrady5<Real>,
    /// Optimized poses (board-to-camera transforms).
    pub poses: Vec<Iso3>,
    /// Final optimization cost.
    pub final_cost: Real,
    /// Mean reprojection error after optimization (pixels).
    pub mean_reproj_error: Real,
}

/// Initialize camera intrinsics using iterative Zhang's method.
pub fn initialize_planar_intrinsics(
    views: &[CorrespondenceView],
    opts: &IterativeIntrinsicsOptions,
) -> Result<PlanarIntrinsicsInitResult> {
    let calib_views: Vec<IterativeCalibView> = views
        .iter()
        .map(|v| {
            let board_2d: Vec<Pt2> = v.points_3d.iter().map(|p3d| Pt2::new(p3d.x, p3d.y)).collect();
            let pixel_2d: Vec<Pt2> = v.points_2d.iter().map(|v2d| Pt2::new(v2d.x, v2d.y)).collect();
            IterativeCalibView {
                board_points: board_2d,
                pixel_points: pixel_2d,
            }
        })
        .collect();

    let result = estimate_intrinsics_iterative(&calib_views, *opts)?;

    Ok(PlanarIntrinsicsInitResult {
        intrinsics: result.intrinsics,
        distortion: result.distortion,
    })
}

/// Optimize camera intrinsics from initial estimates using non-linear refinement.
pub fn optimize_planar_intrinsics_from_init(
    views: &[CorrespondenceView],
    init: &PlanarIntrinsicsInitResult,
    solve_opts: &PlanarIntrinsicsSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<PlanarIntrinsicsOptimResult> {
    let dataset = PlanarDataset {
        views: views.to_vec(),
    };

    // Optimization packs only [fx, fy, cx, cy]; enforce zero skew.
    let mut intrinsics = init.intrinsics;
    intrinsics.skew = 0.0;

    // Recover pose seeds from homographies using provided intrinsics
    let homographies = planar_homographies_from_views(views)?;
    let kmtx = k_matrix_from_intrinsics(&intrinsics);
    let poses0 = poses_from_homographies(&kmtx, &homographies)?;

    let planar_init =
        PlanarIntrinsicsParams::new_from_components(intrinsics, init.distortion, poses0)?;

    // Run optimization
    let optim_result = optimize_planar_intrinsics(
        &dataset,
        planar_init,
        solve_opts.clone(),
        backend_opts.clone(),
    )?;

    // Extract results
    let result_intrinsics = optim_result.params.intrinsics();
    let result_distortion = optim_result.params.distortion();
    let result_poses = optim_result.params.poses().to_vec();

    // Compute mean reprojection error
    let mean_reproj_error =
        compute_mean_reproj_error(views, &result_intrinsics, &result_distortion, &result_poses)?;

    Ok(PlanarIntrinsicsOptimResult {
        intrinsics: result_intrinsics,
        distortion: result_distortion,
        poses: result_poses,
        final_cost: optim_result.report.final_cost,
        mean_reproj_error,
    })
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
        .map(|h| estimate_planar_pose_from_h(kmtx, h).map_err(|e| anyhow::anyhow!("{}", e)))
        .collect()
}

fn compute_mean_reproj_error(
    views: &[CorrespondenceView],
    intrinsics: &FxFyCxCySkew<Real>,
    distortion: &BrownConrady5<Real>,
    poses: &[Iso3],
) -> Result<Real> {
    let camera = Camera::new(Pinhole, *distortion, IdentitySensor, *intrinsics);

    let mut total_error = 0.0;
    let mut total_points = 0;

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

    if total_points == 0 {
        anyhow::bail!("No valid projections for error computation");
    }

    Ok(total_error / total_points as Real)
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{make_pinhole_camera, synthetic::planar};
    use calib_linear::distortion_fit::DistortionFitOptions;

    fn generate_synthetic_views() -> Vec<CorrespondenceView> {
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
        planar::project_views_all(&cam_gt, &board_points, &poses).expect("projection")
    }

    #[test]
    fn initialize_planar_intrinsics_smoke_test() {
        let views = generate_synthetic_views();

        let opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: true,
                iters: 8,
            },
            zero_skew: true,
        };

        let result = initialize_planar_intrinsics(&views, &opts).expect("init should succeed");

        assert!(result.intrinsics.fx > 0.0);
        assert!(result.intrinsics.fy > 0.0);
        assert!(result.distortion.k1.abs() < 1.0);
    }

    #[test]
    fn optimize_from_init_improves_error() {
        let views = generate_synthetic_views();

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
            initialize_planar_intrinsics(&views, &init_opts).expect("init should succeed");

        let solve_opts = PlanarIntrinsicsSolveOptions {
            fix_poses: vec![0],
            ..Default::default()
        };

        let backend_opts = BackendSolveOptions {
            max_iters: 20,
            ..Default::default()
        };

        let optim_result =
            optimize_planar_intrinsics_from_init(&views, &init_result, &solve_opts, &backend_opts)
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
