//! Shared helpers for planar intrinsics workflow family.
//!
//! This module is internal to the pipeline crate and consolidates common
//! initialization logic used by multiple planar problem variants.

use anyhow::{Context, Result};
use vision_calibration_core::{
    CorrespondenceView, FxFyCxCySkew, Iso3, Mat3, PinholeCamera, PlanarDataset, Pt2, Real,
};
use vision_calibration_linear::{
    IterativeIntrinsicsOptions, dlt_homography, estimate_intrinsics_iterative,
    estimate_planar_pose_from_h,
};

/// Result of shared planar bootstrap (homographies + initial camera + poses).
#[derive(Debug, Clone)]
pub(crate) struct PlanarBootstrap {
    pub homographies: Vec<Mat3>,
    pub camera: PinholeCamera,
    pub poses: Vec<Iso3>,
}

/// Estimate homographies, intrinsics/distortion, and per-view poses.
pub(crate) fn bootstrap_planar_intrinsics(
    dataset: &PlanarDataset,
    init_opts: IterativeIntrinsicsOptions,
) -> Result<PlanarBootstrap> {
    let homographies = estimate_view_homographies(dataset)?;
    let camera = estimate_intrinsics_iterative(dataset, init_opts)
        .context("iterative planar intrinsics estimation failed")?;
    let poses = recover_planar_poses_from_homographies(&homographies, &camera.k)?;

    Ok(PlanarBootstrap {
        homographies,
        camera,
        poses,
    })
}

pub(crate) fn estimate_view_homographies(dataset: &PlanarDataset) -> Result<Vec<Mat3>> {
    let mut homographies = Vec::with_capacity(dataset.num_views());
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

pub(crate) fn recover_planar_poses_from_homographies(
    homographies: &[Mat3],
    intrinsics: &FxFyCxCySkew<Real>,
) -> Result<Vec<Iso3>> {
    let k = intrinsics_k_matrix(intrinsics);
    let mut poses = Vec::with_capacity(homographies.len());
    for (idx, h) in homographies.iter().enumerate() {
        let pose = estimate_planar_pose_from_h(&k, h)
            .with_context(|| format!("failed to recover pose for view {}", idx))?;
        poses.push(pose);
    }
    Ok(poses)
}

fn board_and_pixel_points(view: &CorrespondenceView) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
    (board_2d, pixel_2d)
}

fn intrinsics_k_matrix(k: &FxFyCxCySkew<Real>) -> Mat3 {
    Mat3::new(k.fx, k.skew, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{
        BrownConrady5, FxFyCxCySkew, View, make_pinhole_camera, synthetic::planar,
    };
    use vision_calibration_linear::DistortionFitOptions;

    fn make_test_dataset() -> PlanarDataset {
        let cam_gt = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 900.0,
                fy: 880.0,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        let board_points = planar::grid_points(6, 5, 0.04);
        let poses = planar::poses_yaw_y_z(4, 0.0, 0.1, 0.6, 0.1);
        let views = planar::project_views_all(&cam_gt, &board_points, &poses).expect("views");
        PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).expect("dataset")
    }

    #[test]
    fn bootstrap_estimates_homographies_and_poses_per_view() {
        let dataset = make_test_dataset();
        let result = bootstrap_planar_intrinsics(
            &dataset,
            IterativeIntrinsicsOptions {
                iterations: 2,
                distortion_opts: DistortionFitOptions {
                    fix_k3: true,
                    fix_tangential: false,
                    iters: 8,
                },
                zero_skew: true,
            },
        )
        .expect("bootstrap");

        assert_eq!(result.homographies.len(), dataset.num_views());
        assert_eq!(result.poses.len(), dataset.num_views());
        assert!(result.camera.k.fx.is_finite());
        assert!(result.camera.k.fy.is_finite());
    }
}
