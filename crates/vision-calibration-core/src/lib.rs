//! Core math and geometry primitives for `calibration-rs`.
//!
//! This crate provides the foundational building blocks used by all other
//! crates in the workspace:
//!
//! - linear algebra type aliases (`Real`, `Vec2`, `Pt3`, and friends),
//! - composable camera models (projection + distortion + sensor + intrinsics),
//! - a deterministic, model-agnostic RANSAC engine.
//!
//! Camera pipeline (conceptually):
//! `pixel = intrinsics(sensor(distortion(projection(dir))))`
//!
//! The sensor stage supports a Scheimpflug/tilted sensor homography aligned
//! with OpenCV's `computeTiltProjectionMatrix`.
//!
//! # Modules
//!
//! - \[`math`\]: basic type aliases and homogeneous helpers.
//! - \[`models`\]: camera model traits and configuration wrappers.
//! - \[`ransac`\]: generic robust estimation helpers.
//! - \[`synthetic`\]: deterministic synthetic data helpers (tests/examples/benchmarks).
//!
//! # Example
//!
//! ```no_run
//! use vision_calibration_core::{
//!     CameraParams, DistortionParams, FxFyCxCySkew, IntrinsicsParams, ProjectionParams,
//!     SensorParams,
//! };
//!
//! let params = CameraParams {
//!     projection: ProjectionParams::Pinhole,
//!     distortion: DistortionParams::None,
//!     sensor: SensorParams::Identity,
//!     intrinsics: IntrinsicsParams::FxFyCxCySkew {
//!         params: FxFyCxCySkew {
//!             fx: 800.0,
//!             fy: 800.0,
//!             cx: 640.0,
//!             cy: 360.0,
//!             skew: 0.0,
//!         },
//!     },
//! };
//! let cam = params.build();
//! let px = cam.project_point_c(&nalgebra::Vector3::new(0.1, 0.2, 1.0));
//! assert!(px.is_some());
//! ```

/// Linear algebra type aliases and helpers.
mod math;
/// Camera models and distortion utilities.
mod models;
/// Generic RANSAC engine and traits.
mod ransac;
/// Deterministic synthetic data generation helpers.
///
/// This module provides small, reusable building blocks for constructing
/// synthetic calibration problems (planar grids, poses, projections, noise).
/// It is used in workspace tests/examples and can be useful for benchmarking
/// and regression testing.
pub mod synthetic;
/// Test utilities for cross-crate calibration testing.
///
/// This module is public to allow usage in integration tests across
/// the workspace, but is not intended for production use.
pub mod test_utils;
/// Common types for observations, results, and options.
mod types;
mod view;

pub use math::*;
pub use models::*;
pub use ransac::*;
pub use types::*;
pub use view::*;

use anyhow::Result;

pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

pub fn make_pinhole_camera(k: FxFyCxCySkew<Real>, dist: BrownConrady5<Real>) -> PinholeCamera {
    Camera::new(Pinhole, dist, IdentitySensor, k)
}

pub fn pinhole_camera_params(camera: &PinholeCamera) -> CameraParams {
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

pub struct TargetPose {
    pub camera_se3_target: Iso3,
}

pub fn compute_mean_reproj_error(
    camera: &PinholeCamera,
    views: &[View<TargetPose>],
) -> Result<Real> {
    let mut total_error = 0.0;
    let mut total_points = 0;

    for view in views {
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = view.meta.camera_se3_target * p3d;
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

/// Compute reprojection error statistics for a multi-camera rig dataset.
///
/// Uses the transform chain:
/// - `cam_se3_target = cam_se3_rig[cam] * rig_se3_target[view]`
/// - `p_cam = cam_se3_target * p_target`
///
/// where:
/// - `cam_se3_rig[cam]` is `T_C_R` (rig -> camera)
/// - `rig_se3_target[view]` is `T_R_T` (target -> rig)
///
/// Points that fail projection are skipped. Returns an error if no points are
/// successfully projected.
pub fn compute_rig_reprojection_stats<Meta>(
    cameras: &[PinholeCamera],
    dataset: &RigDataset<Meta>,
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> Result<ReprojectionStats> {
    anyhow::ensure!(
        cameras.len() == dataset.num_cameras,
        "camera count {} != num_cameras {}",
        cameras.len(),
        dataset.num_cameras
    );
    anyhow::ensure!(
        cam_se3_rig.len() == dataset.num_cameras,
        "cam_se3_rig count {} != num_cameras {}",
        cam_se3_rig.len(),
        dataset.num_cameras
    );
    anyhow::ensure!(
        rig_se3_target.len() == dataset.num_views(),
        "rig_se3_target count {} != num_views {}",
        rig_se3_target.len(),
        dataset.num_views()
    );

    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max = 0.0_f64;
    let mut count = 0usize;

    for (view_idx, view) in dataset.views.iter().enumerate() {
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else {
                continue;
            };
            let cam = &cameras[cam_idx];
            let cam_se3_target = cam_se3_rig[cam_idx] * rig_se3_target[view_idx];

            for (pw, uv) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_cam = cam_se3_target * pw;
                let Some(proj) = cam.project_point_c(&p_cam.coords) else {
                    continue;
                };
                let err = (proj - *uv).norm();
                sum += err;
                sum_sq += err * err;
                max = max.max(err);
                count += 1;
            }
        }
    }

    if count == 0 {
        anyhow::bail!("No valid projections for error computation");
    }

    let count_f = count as f64;
    Ok(ReprojectionStats {
        mean: sum / count_f,
        rms: (sum_sq / count_f).sqrt(),
        max,
        count,
    })
}

/// Compute per-camera reprojection error statistics for a multi-camera rig dataset.
///
/// Uses the same transform chain as [`compute_rig_reprojection_stats`], but splits
/// the aggregation by camera index.
pub fn compute_rig_reprojection_stats_per_camera<Meta>(
    cameras: &[PinholeCamera],
    dataset: &RigDataset<Meta>,
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> Result<Vec<ReprojectionStats>> {
    anyhow::ensure!(
        cameras.len() == dataset.num_cameras,
        "camera count {} != num_cameras {}",
        cameras.len(),
        dataset.num_cameras
    );
    anyhow::ensure!(
        cam_se3_rig.len() == dataset.num_cameras,
        "cam_se3_rig count {} != num_cameras {}",
        cam_se3_rig.len(),
        dataset.num_cameras
    );
    anyhow::ensure!(
        rig_se3_target.len() == dataset.num_views(),
        "rig_se3_target count {} != num_views {}",
        rig_se3_target.len(),
        dataset.num_views()
    );

    let mut sum = vec![0.0_f64; dataset.num_cameras];
    let mut sum_sq = vec![0.0_f64; dataset.num_cameras];
    let mut max = vec![0.0_f64; dataset.num_cameras];
    let mut count = vec![0usize; dataset.num_cameras];

    for (view_idx, view) in dataset.views.iter().enumerate() {
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else {
                continue;
            };
            let cam = &cameras[cam_idx];
            let cam_se3_target = cam_se3_rig[cam_idx] * rig_se3_target[view_idx];

            for (pw, uv) in obs.points_3d.iter().zip(obs.points_2d.iter()) {
                let p_cam = cam_se3_target * pw;
                let Some(proj) = cam.project_point_c(&p_cam.coords) else {
                    continue;
                };
                let err = (proj - *uv).norm();
                sum[cam_idx] += err;
                sum_sq[cam_idx] += err * err;
                max[cam_idx] = max[cam_idx].max(err);
                count[cam_idx] += 1;
            }
        }
    }

    let mut stats = Vec::with_capacity(dataset.num_cameras);
    for cam_idx in 0..dataset.num_cameras {
        if count[cam_idx] == 0 {
            anyhow::bail!(
                "camera {} has no valid projections for error computation",
                cam_idx
            );
        }
        let n = count[cam_idx] as f64;
        stats.push(ReprojectionStats {
            mean: sum[cam_idx] / n,
            rms: (sum_sq[cam_idx] / n).sqrt(),
            max: max[cam_idx],
            count: count[cam_idx],
        });
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3};

    fn make_iso(angles: (Real, Real, Real), t: (Real, Real, Real)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
        let tr = Translation3::new(t.0, t.1, t.2);
        Iso3::from_parts(tr, rot.into())
    }

    #[test]
    fn rig_reprojection_stats_zero_for_perfect_data() -> Result<()> {
        let cam0 = make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        );
        let cam1 = cam0.clone();

        let cam_se3_rig = vec![Iso3::identity(), make_iso((0.0, 0.0, 0.0), (0.2, 0.0, 0.0))];
        let rig_se3_target = vec![make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))];

        let points_3d = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.1, 0.0, 0.0),
            Pt3::new(0.0, 0.1, 0.0),
            Pt3::new(0.1, 0.1, 0.0),
        ];

        let make_obs = |cam: &PinholeCamera, cam_se3_target: &Iso3| -> Result<CorrespondenceView> {
            let points_2d = points_3d
                .iter()
                .map(|p| {
                    let p_cam = cam_se3_target * p;
                    cam.project_point_c(&p_cam.coords)
                        .ok_or_else(|| anyhow::anyhow!("projection failed"))
                })
                .collect::<Result<Vec<_>>>()?;
            CorrespondenceView::new(points_3d.clone(), points_2d)
        };

        let view = RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![
                    Some(make_obs(&cam0, &(cam_se3_rig[0] * rig_se3_target[0]))?),
                    Some(make_obs(&cam1, &(cam_se3_rig[1] * rig_se3_target[0]))?),
                ],
            },
        };
        let dataset = RigDataset::new(vec![view], 2)?;

        let cameras = [cam0, cam1];
        let stats =
            compute_rig_reprojection_stats(&cameras, &dataset, &cam_se3_rig, &rig_se3_target)?;
        assert_eq!(stats.count, 8);
        assert!(stats.mean < 1e-12, "mean {}", stats.mean);
        assert!(stats.rms < 1e-12, "rms {}", stats.rms);
        assert!(stats.max < 1e-12, "max {}", stats.max);

        let per_cam = compute_rig_reprojection_stats_per_camera(
            &cameras,
            &dataset,
            &cam_se3_rig,
            &rig_se3_target,
        )?;
        assert_eq!(per_cam.len(), 2);
        assert_eq!(per_cam[0].count, 4);
        assert_eq!(per_cam[1].count, 4);
        assert!(per_cam[0].mean < 1e-12, "cam0 mean {}", per_cam[0].mean);
        assert!(per_cam[1].mean < 1e-12, "cam1 mean {}", per_cam[1].mean);

        Ok(())
    }
}
