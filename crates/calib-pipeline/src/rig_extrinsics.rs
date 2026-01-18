//! High-level multi-camera rig extrinsics pipeline.

use crate::session::{problem_types::RigExtrinsicsProblem, CalibrationSession};
use anyhow::{Context, Result};
use calib_core::{CameraParams, Iso3, Pt3, Real};
use serde::{Deserialize, Serialize};

pub use crate::session::problem_types::{
    RigExtrinsicsInitOptions, RigExtrinsicsObservations as RigExtrinsicsInput,
    RigExtrinsicsOptimOptions, RigExtrinsicsOptimized as RigExtrinsicsReport, RigViewData,
};

/// End-to-end rig extrinsics configuration (init + non-linear refinement).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigExtrinsicsConfig {
    #[serde(default)]
    pub init: RigExtrinsicsInitOptions,
    #[serde(default)]
    pub optim: RigExtrinsicsOptimOptions,
}

/// Run the full rig extrinsics pipeline (init + optimize) and return a report.
pub fn run_rig_extrinsics(
    input: &RigExtrinsicsInput,
    config: &RigExtrinsicsConfig,
) -> Result<RigExtrinsicsReport> {
    let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
    session.set_observations(input.clone());
    session.initialize(config.init.clone())?;
    session.optimize(config.optim.clone())?;
    session.export()
}

/// Reprojection error summary for a rig calibration result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RigReprojectionErrors {
    /// Mean L2 reprojection error per point (pixels).
    pub mean_px: Option<Real>,
    /// RMS L2 reprojection error per point (pixels).
    pub rmse_px: Option<Real>,
    /// Mean reprojection error per camera (pixels).
    pub per_camera_mean_px: Vec<Option<Real>>,
    /// RMS reprojection error per camera (pixels).
    pub per_camera_rmse_px: Vec<Option<Real>>,
    /// Mean reprojection error per view (pixels).
    pub per_view_mean_px: Vec<Option<Real>>,
    /// RMS reprojection error per view (pixels).
    pub per_view_rmse_px: Vec<Option<Real>>,
    /// Number of points used in the metric.
    pub num_points_used: usize,
    /// Number of points skipped (not projectable).
    pub num_points_skipped: usize,
}

/// Compute reprojection errors for arbitrary rig parameters.
pub fn rig_reprojection_errors(
    input: &RigExtrinsicsInput,
    cameras: &[CameraParams],
    cam_to_rig: &[Iso3],
    rig_from_target: &[Iso3],
) -> Result<RigReprojectionErrors> {
    anyhow::ensure!(
        input.num_cameras == cameras.len(),
        "camera params count {} != num_cameras {}",
        cameras.len(),
        input.num_cameras
    );
    anyhow::ensure!(
        input.num_cameras == cam_to_rig.len(),
        "cam_to_rig count {} != num_cameras {}",
        cam_to_rig.len(),
        input.num_cameras
    );
    anyhow::ensure!(
        input.views.len() == rig_from_target.len(),
        "rig_from_target count {} != num_views {}",
        rig_from_target.len(),
        input.views.len()
    );

    let mut per_camera_sum = vec![0.0; input.num_cameras];
    let mut per_camera_sq_sum = vec![0.0; input.num_cameras];
    let mut per_camera_n = vec![0usize; input.num_cameras];

    let mut per_view_sum = vec![0.0; input.views.len()];
    let mut per_view_sq_sum = vec![0.0; input.views.len()];
    let mut per_view_n = vec![0usize; input.views.len()];

    let mut total_sum = 0.0;
    let mut total_sq_sum = 0.0;
    let mut total_n = 0usize;
    let mut skipped = 0usize;

    for (view_idx, view) in input.views.iter().enumerate() {
        anyhow::ensure!(
            view.cameras.len() == input.num_cameras,
            "view {} has {} cameras, expected {}",
            view_idx,
            view.cameras.len(),
            input.num_cameras
        );

        for (cam_idx, cam_view_opt) in view.cameras.iter().enumerate() {
            let Some(cam_view) = cam_view_opt else {
                continue;
            };
            anyhow::ensure!(
                cam_view.points_3d.len() == cam_view.points_2d.len(),
                "view {} camera {} has mismatched 3D/2D point count",
                view_idx,
                cam_idx
            );

            let k = match cameras[cam_idx].intrinsics {
                calib_core::IntrinsicsParams::FxFyCxCySkew { params } => params,
            };
            let dist: calib_core::BrownConrady5<f64> = match cameras[cam_idx].distortion {
                calib_core::DistortionParams::BrownConrady5 { params } => params,
                calib_core::DistortionParams::None => calib_core::BrownConrady5 {
                    k1: 0.0,
                    k2: 0.0,
                    k3: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                    iters: 8,
                },
            };
            let cam_model = crate::make_pinhole_camera(k, dist);

            for (pw, uv) in cam_view.points_3d.iter().zip(cam_view.points_2d.iter()) {
                let p_rig: Pt3 = rig_from_target[view_idx].transform_point(pw);
                let p_cam: Pt3 = cam_to_rig[cam_idx].inverse().transform_point(&p_rig);
                let Some(proj) = cam_model.project_point(&p_cam) else {
                    skipped += 1;
                    continue;
                };

                let du = proj.x - uv.x;
                let dv = proj.y - uv.y;
                let e = (du * du + dv * dv).sqrt();

                per_camera_sum[cam_idx] += e;
                per_camera_sq_sum[cam_idx] += e * e;
                per_camera_n[cam_idx] += 1;

                per_view_sum[view_idx] += e;
                per_view_sq_sum[view_idx] += e * e;
                per_view_n[view_idx] += 1;

                total_sum += e;
                total_sq_sum += e * e;
                total_n += 1;
            }
        }
    }

    let mean = if total_n > 0 {
        Some(total_sum / total_n as Real)
    } else {
        None
    };
    let rmse = if total_n > 0 {
        Some((total_sq_sum / total_n as Real).sqrt())
    } else {
        None
    };

    let per_camera_mean_px = (0..input.num_cameras)
        .map(|i| {
            if per_camera_n[i] > 0 {
                Some(per_camera_sum[i] / per_camera_n[i] as Real)
            } else {
                None
            }
        })
        .collect();
    let per_camera_rmse_px = (0..input.num_cameras)
        .map(|i| {
            if per_camera_n[i] > 0 {
                Some((per_camera_sq_sum[i] / per_camera_n[i] as Real).sqrt())
            } else {
                None
            }
        })
        .collect();

    let per_view_mean_px = (0..input.views.len())
        .map(|i| {
            if per_view_n[i] > 0 {
                Some(per_view_sum[i] / per_view_n[i] as Real)
            } else {
                None
            }
        })
        .collect();
    let per_view_rmse_px = (0..input.views.len())
        .map(|i| {
            if per_view_n[i] > 0 {
                Some((per_view_sq_sum[i] / per_view_n[i] as Real).sqrt())
            } else {
                None
            }
        })
        .collect();

    Ok(RigReprojectionErrors {
        mean_px: mean,
        rmse_px: rmse,
        per_camera_mean_px,
        per_camera_rmse_px,
        per_view_mean_px,
        per_view_rmse_px,
        num_points_used: total_n,
        num_points_skipped: skipped,
    })
}

/// Compute reprojection errors for a rig calibration report.
pub fn rig_reprojection_errors_from_report(
    input: &RigExtrinsicsInput,
    report: &RigExtrinsicsReport,
) -> Result<RigReprojectionErrors> {
    rig_reprojection_errors(
        input,
        &report.cameras,
        &report.cam_to_rig,
        &report.rig_from_target,
    )
    .context("rig reprojection error evaluation failed")
}
