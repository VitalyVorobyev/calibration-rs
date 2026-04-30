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

/// Typed error enum for this crate.
pub mod error;
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

pub use error::Error;
pub use math::*;
pub use models::*;
pub use ransac::*;
pub use types::*;
pub use view::*;

/// Concrete pinhole camera alias used across single-camera workflows.
///
/// Composition:
/// - projection: [`Pinhole`]
/// - distortion: [`BrownConrady5`]
/// - sensor: [`IdentitySensor`]
/// - intrinsics: [`FxFyCxCySkew`]
pub type PinholeCamera =
    Camera<Real, Pinhole, BrownConrady5<Real>, IdentitySensor, FxFyCxCySkew<Real>>;

/// Build a [`PinholeCamera`] from intrinsics and Brown-Conrady distortion.
pub fn make_pinhole_camera(k: FxFyCxCySkew<Real>, dist: BrownConrady5<Real>) -> PinholeCamera {
    Camera::new(Pinhole, dist, IdentitySensor, k)
}

/// Convert a concrete [`PinholeCamera`] into serializable [`CameraParams`].
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

/// Metadata carrying the per-view pose `camera_se3_target`.
pub struct TargetPose {
    /// Pose of target frame in camera frame (`T_C_T`).
    pub camera_se3_target: Iso3,
}

/// Compute mean per-point reprojection error for a calibrated camera and posed views.
///
/// The transform chain is `p_cam = T_C_T * p_target`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if no points could be projected.
pub fn compute_mean_reproj_error(
    camera: &PinholeCamera,
    views: &[View<TargetPose>],
) -> Result<Real, Error> {
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
        return Err(Error::invalid_input(
            "no valid projections for error computation",
        ));
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
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if array lengths don't match expected counts
/// or if no points could be projected.
pub fn compute_rig_reprojection_stats<Meta>(
    cameras: &[PinholeCamera],
    dataset: &RigDataset<Meta>,
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> Result<ReprojectionStats, Error> {
    if cameras.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "camera count {} != num_cameras {}",
            cameras.len(),
            dataset.num_cameras
        )));
    }
    if cam_se3_rig.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "cam_se3_rig count {} != num_cameras {}",
            cam_se3_rig.len(),
            dataset.num_cameras
        )));
    }
    if rig_se3_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "rig_se3_target count {} != num_views {}",
            rig_se3_target.len(),
            dataset.num_views()
        )));
    }

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
        return Err(Error::invalid_input(
            "no valid projections for error computation",
        ));
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
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if array lengths don't match expected counts
/// or if any camera has no valid projections.
pub fn compute_rig_reprojection_stats_per_camera<Meta>(
    cameras: &[PinholeCamera],
    dataset: &RigDataset<Meta>,
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> Result<Vec<ReprojectionStats>, Error> {
    if cameras.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "camera count {} != num_cameras {}",
            cameras.len(),
            dataset.num_cameras
        )));
    }
    if cam_se3_rig.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "cam_se3_rig count {} != num_cameras {}",
            cam_se3_rig.len(),
            dataset.num_cameras
        )));
    }
    if rig_se3_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "rig_se3_target count {} != num_views {}",
            rig_se3_target.len(),
            dataset.num_views()
        )));
    }

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
            return Err(Error::invalid_input(format!(
                "camera {cam_idx} has no valid projections for error computation"
            )));
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

/// Compute per-feature target reprojection residuals for a single calibrated
/// camera and posed planar views.
///
/// Produces one [`TargetFeatureResidual`] per (view, feature) pair, regardless
/// of whether projection succeeds. Failed projections are recorded with
/// `projected_px = None` and `error_px = None` so the diagnose UI can
/// distinguish "feature absent" from "projection diverged".
///
/// All records carry `camera = 0`. Records are pose-major: outer = view
/// index, inner = feature index in that view's `points_3d`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if `camera_se3_target.len() != dataset.num_views()`.
pub fn compute_planar_target_residuals(
    camera: &PinholeCamera,
    dataset: &PlanarDataset,
    camera_se3_target: &[Iso3],
) -> Result<Vec<TargetFeatureResidual>, Error> {
    if camera_se3_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "camera_se3_target count {} != num_views {}",
            camera_se3_target.len(),
            dataset.num_views()
        )));
    }

    let mut out = Vec::new();
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let pose = camera_se3_target[view_idx];
        for (feature_idx, (p3d, p2d)) in view
            .obs
            .points_3d
            .iter()
            .zip(view.obs.points_2d.iter())
            .enumerate()
        {
            let p_cam = pose * p3d;
            let (projected_px, error_px) = match camera.project_point_c(&p_cam.coords) {
                Some(proj) => {
                    let err = (proj - *p2d).norm();
                    (Some([proj.x, proj.y]), Some(err))
                }
                None => (None, None),
            };
            out.push(TargetFeatureResidual {
                pose: view_idx,
                camera: 0,
                feature: feature_idx,
                target_xyz_m: [p3d.x, p3d.y, p3d.z],
                observed_px: [p2d.x, p2d.y],
                projected_px,
                error_px,
            });
        }
    }
    Ok(out)
}

/// Compute per-feature target reprojection residuals for a multi-camera rig
/// dataset, mirroring [`compute_rig_reprojection_stats_per_camera`].
///
/// Transform chain: `cam_se3_target = cam_se3_rig[cam] * rig_se3_target[view]`.
/// Records are emitted pose-major: outer = view, inner = camera (camera-`None`
/// slots skipped), innermost = feature index in that view+camera's
/// `points_3d`. Failed projections become records with `projected_px` and
/// `error_px` set to `None`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] if any of:
/// - `cameras.len() != dataset.num_cameras`
/// - `cam_se3_rig.len() != dataset.num_cameras`
/// - `rig_se3_target.len() != dataset.num_views()`.
pub fn compute_rig_target_residuals<Meta>(
    cameras: &[PinholeCamera],
    dataset: &RigDataset<Meta>,
    cam_se3_rig: &[Iso3],
    rig_se3_target: &[Iso3],
) -> Result<Vec<TargetFeatureResidual>, Error> {
    if cameras.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "camera count {} != num_cameras {}",
            cameras.len(),
            dataset.num_cameras
        )));
    }
    if cam_se3_rig.len() != dataset.num_cameras {
        return Err(Error::invalid_input(format!(
            "cam_se3_rig count {} != num_cameras {}",
            cam_se3_rig.len(),
            dataset.num_cameras
        )));
    }
    if rig_se3_target.len() != dataset.num_views() {
        return Err(Error::invalid_input(format!(
            "rig_se3_target count {} != num_views {}",
            rig_se3_target.len(),
            dataset.num_views()
        )));
    }

    let mut out = Vec::new();
    for (view_idx, view) in dataset.views.iter().enumerate() {
        for (cam_idx, cam_obs) in view.obs.cameras.iter().enumerate() {
            let Some(obs) = cam_obs else {
                continue;
            };
            let cam = &cameras[cam_idx];
            let cam_se3_target = cam_se3_rig[cam_idx] * rig_se3_target[view_idx];

            for (feature_idx, (p3d, p2d)) in
                obs.points_3d.iter().zip(obs.points_2d.iter()).enumerate()
            {
                let p_cam = cam_se3_target * p3d;
                let (projected_px, error_px) = match cam.project_point_c(&p_cam.coords) {
                    Some(proj) => {
                        let err = (proj - *p2d).norm();
                        (Some([proj.x, proj.y]), Some(err))
                    }
                    None => (None, None),
                };
                out.push(TargetFeatureResidual {
                    pose: view_idx,
                    camera: cam_idx,
                    feature: feature_idx,
                    target_xyz_m: [p3d.x, p3d.y, p3d.z],
                    observed_px: [p2d.x, p2d.y],
                    projected_px,
                    error_px,
                });
            }
        }
    }
    Ok(out)
}

/// Aggregate a sequence of per-feature pixel errors into a
/// [`FeatureResidualHistogram`].
///
/// Bucket edges are fixed at [`REPROJECTION_HISTOGRAM_EDGES_PX`]
/// (`[1, 2, 5, 10]` px) producing five buckets `[<=1, <=2, <=5, <=10, >10]`.
///
/// `NaN` errors are skipped silently. Negative errors are bucketed into the
/// `<=1` bucket (their absolute value is not used; the helpers in this crate
/// only emit non-negative `error_px` so this case does not arise from
/// well-formed inputs).
///
/// Empty input yields the [`FeatureResidualHistogram::default()`] value
/// (all-zero counts; mean and max both zero).
pub fn build_feature_histogram(
    errors_px: impl IntoIterator<Item = f64>,
) -> FeatureResidualHistogram {
    let edges = REPROJECTION_HISTOGRAM_EDGES_PX;
    let mut counts = [0usize; 5];
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut max = 0.0f64;

    for err in errors_px {
        if err.is_nan() {
            continue;
        }
        let bucket = if err <= edges[0] {
            0
        } else if err <= edges[1] {
            1
        } else if err <= edges[2] {
            2
        } else if err <= edges[3] {
            3
        } else {
            4
        };
        counts[bucket] += 1;
        count += 1;
        sum += err;
        if err > max {
            max = err;
        }
    }

    let mean = if count == 0 { 0.0 } else { sum / count as f64 };
    FeatureResidualHistogram {
        bucket_edges_px: edges,
        counts,
        count,
        mean,
        max,
    }
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
    fn rig_reprojection_stats_zero_for_perfect_data() -> anyhow::Result<()> {
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

        let make_obs =
            |cam: &PinholeCamera, cam_se3_target: &Iso3| -> anyhow::Result<CorrespondenceView> {
                let points_2d = points_3d
                    .iter()
                    .map(|p| {
                        let p_cam = cam_se3_target * p;
                        cam.project_point_c(&p_cam.coords)
                            .ok_or_else(|| anyhow::anyhow!("projection failed"))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                CorrespondenceView::new(points_3d.clone(), points_2d)
                    .map_err(|e| anyhow::anyhow!("{e}"))
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

    fn make_test_pinhole() -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 800.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        )
    }

    #[test]
    fn planar_target_residuals_are_zero_for_perfect_data() -> anyhow::Result<()> {
        let camera = make_test_pinhole();
        let pose = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0));

        let points_3d = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let points_2d: Vec<_> = points_3d
            .iter()
            .map(|p| {
                let p_cam = pose * p;
                camera.project_point_c(&p_cam.coords).unwrap()
            })
            .collect();
        let view =
            View::without_meta(CorrespondenceView::new(points_3d.clone(), points_2d).unwrap());
        let dataset = PlanarDataset::new(vec![view])?;

        let residuals = compute_planar_target_residuals(&camera, &dataset, &[pose])?;
        assert_eq!(residuals.len(), 4);
        for (i, r) in residuals.iter().enumerate() {
            assert_eq!(r.pose, 0);
            assert_eq!(r.camera, 0);
            assert_eq!(r.feature, i);
            assert_eq!(
                r.target_xyz_m,
                [points_3d[i].x, points_3d[i].y, points_3d[i].z]
            );
            assert!(r.projected_px.is_some(), "feature {i} projection diverged");
            let err = r.error_px.unwrap();
            assert!(err < 1e-9, "feature {i} error {err}");
        }
        Ok(())
    }

    #[test]
    fn planar_target_residuals_rejects_pose_count_mismatch() -> anyhow::Result<()> {
        let camera = make_test_pinhole();
        let pose = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0));
        let points_3d = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let points_2d: Vec<_> = points_3d
            .iter()
            .map(|p| {
                let p_cam = pose * p;
                camera.project_point_c(&p_cam.coords).unwrap()
            })
            .collect();
        let view =
            View::without_meta(CorrespondenceView::new(points_3d.clone(), points_2d).unwrap());
        let dataset = PlanarDataset::new(vec![view])?;

        // 0 poses for 1 view → mismatch.
        let result = compute_planar_target_residuals(&camera, &dataset, &[]);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn rig_target_residuals_are_zero_for_perfect_data() -> anyhow::Result<()> {
        let cam0 = make_test_pinhole();
        let cam1 = cam0.clone();
        let cam_se3_rig = vec![Iso3::identity(), make_iso((0.0, 0.0, 0.0), (0.2, 0.0, 0.0))];
        let rig_se3_target = vec![make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))];
        let points_3d = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];

        let make_obs =
            |cam: &PinholeCamera, cam_se3_target: &Iso3| -> anyhow::Result<CorrespondenceView> {
                let points_2d = points_3d
                    .iter()
                    .map(|p| {
                        let p_cam = cam_se3_target * p;
                        cam.project_point_c(&p_cam.coords)
                            .ok_or_else(|| anyhow::anyhow!("projection failed"))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                CorrespondenceView::new(points_3d.clone(), points_2d)
                    .map_err(|e| anyhow::anyhow!("{e}"))
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
        let residuals =
            compute_rig_target_residuals(&cameras, &dataset, &cam_se3_rig, &rig_se3_target)?;
        assert_eq!(residuals.len(), 8);
        for r in &residuals {
            assert!(r.projected_px.is_some());
            assert!(r.error_px.unwrap() < 1e-9);
        }
        // Iteration order: pose-major, then camera (0 before 1), then feature.
        assert_eq!(residuals[0].camera, 0);
        assert_eq!(residuals[3].camera, 0);
        assert_eq!(residuals[4].camera, 1);
        assert_eq!(residuals[7].camera, 1);
        // Feature index resets per (pose, camera) slot.
        assert_eq!(residuals[0].feature, 0);
        assert_eq!(residuals[3].feature, 3);
        assert_eq!(residuals[4].feature, 0);
        Ok(())
    }

    #[test]
    fn rig_target_residuals_skips_camera_none_slots() -> anyhow::Result<()> {
        // View 0: camera 0 only. View 1: both cameras. Expect 4 + 8 = 12 records.
        let cam0 = make_test_pinhole();
        let cam1 = cam0.clone();
        let cam_se3_rig = vec![Iso3::identity(), make_iso((0.0, 0.0, 0.0), (0.2, 0.0, 0.0))];
        let rig_se3_target = vec![
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            make_iso((0.0, 0.05, 0.0), (0.0, 0.0, 1.2)),
        ];
        let points_3d = vec![
            Pt3::new(0.0, 0.0, 0.0),
            Pt3::new(0.05, 0.0, 0.0),
            Pt3::new(0.0, 0.05, 0.0),
            Pt3::new(0.05, 0.05, 0.0),
        ];
        let make_obs =
            |cam: &PinholeCamera, cam_se3_target: &Iso3| -> anyhow::Result<CorrespondenceView> {
                let points_2d = points_3d
                    .iter()
                    .map(|p| {
                        let p_cam = cam_se3_target * p;
                        cam.project_point_c(&p_cam.coords)
                            .ok_or_else(|| anyhow::anyhow!("projection failed"))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                CorrespondenceView::new(points_3d.clone(), points_2d)
                    .map_err(|e| anyhow::anyhow!("{e}"))
            };

        let view0 = RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![
                    Some(make_obs(&cam0, &(cam_se3_rig[0] * rig_se3_target[0]))?),
                    None,
                ],
            },
        };
        let view1 = RigView {
            meta: NoMeta,
            obs: RigViewObs {
                cameras: vec![
                    Some(make_obs(&cam0, &(cam_se3_rig[0] * rig_se3_target[1]))?),
                    Some(make_obs(&cam1, &(cam_se3_rig[1] * rig_se3_target[1]))?),
                ],
            },
        };
        let dataset = RigDataset::new(vec![view0, view1], 2)?;

        let residuals =
            compute_rig_target_residuals(&[cam0, cam1], &dataset, &cam_se3_rig, &rig_se3_target)?;
        assert_eq!(residuals.len(), 12);
        // First 4 records are pose=0 cam=0; next 4 are pose=1 cam=0; last 4 are pose=1 cam=1.
        assert_eq!((residuals[0].pose, residuals[0].camera), (0, 0));
        assert_eq!((residuals[3].pose, residuals[3].camera), (0, 0));
        assert_eq!((residuals[4].pose, residuals[4].camera), (1, 0));
        assert_eq!((residuals[7].pose, residuals[7].camera), (1, 0));
        assert_eq!((residuals[8].pose, residuals[8].camera), (1, 1));
        assert_eq!((residuals[11].pose, residuals[11].camera), (1, 1));
        Ok(())
    }

    #[test]
    fn build_feature_histogram_buckets_correctly() {
        // Edges: [1, 2, 5, 10] → buckets [<=1, <=2, <=5, <=10, >10].
        // Test exact boundary values land in the lower bucket.
        let errors = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 11.0, 100.0];
        let h = build_feature_histogram(errors);
        assert_eq!(h.count, 10);
        assert_eq!(h.counts, [2, 2, 2, 2, 2]);
        assert_eq!(h.bucket_edges_px, REPROJECTION_HISTOGRAM_EDGES_PX);
        let expected_mean = errors.iter().sum::<f64>() / errors.len() as f64;
        assert!((h.mean - expected_mean).abs() < 1e-12);
        assert_eq!(h.max, 100.0);
    }

    #[test]
    fn build_feature_histogram_empty_input_returns_default() {
        let h = build_feature_histogram(std::iter::empty());
        assert_eq!(h, FeatureResidualHistogram::default());
    }

    #[test]
    fn build_feature_histogram_skips_nan() {
        let h = build_feature_histogram([0.5, f64::NAN, 1.5, f64::NAN, 3.0]);
        assert_eq!(h.count, 3);
        assert_eq!(h.counts, [1, 1, 1, 0, 0]);
        assert!((h.mean - (0.5 + 1.5 + 3.0) / 3.0).abs() < 1e-12);
        assert_eq!(h.max, 3.0);
    }
}
