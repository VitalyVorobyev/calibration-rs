//! Step functions for single laserline device calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, CorrespondenceView, FxFyCxCySkew, Iso3, NoMeta, Pinhole, PlanarDataset,
    Pt2, Real, SensorModel, View,
};
use vision_calibration_linear::laserline::{
    LaserlinePlaneSolver, LaserlineView as LinearLaserlineView,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    LaserPlane, LaserlineParams, LaserlineStats, compute_laserline_stats, optimize_laserline,
};

use crate::session::CalibrationSession;

use super::problem::{LaserlineDeviceConfig, LaserlineDeviceOutput, LaserlineDeviceProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

/// Options for the initialization step.
#[derive(Debug, Clone, Default)]
pub struct DeviceInitOptions {
    /// Override the number of iterations for iterative intrinsics estimation.
    pub iterations: Option<usize>,
}

/// Manual initialization seeds for laserline device calibration.
///
/// All fields are `Option<T>`:
/// - `None` means *auto-initialize this group* (same path as plain `step_init`).
/// - `Some(value)` means *use this value*; do not auto-initialize.
///
/// **Sensor is intentionally not in this struct** — for laserline devices the
/// sensor model is a hardware property taken from `session.config.init.sensor_init`.
/// See ADR 0011.
///
/// Partial-seed semantics:
/// - `intrinsics: Some` skips `estimate_intrinsics_iterative`. Distortion defaults
///   to `BrownConrady5::default()` (zeros) unless also seeded; poses recover from
///   per-view homographies using the manual intrinsics.
/// - `plane: Some` skips the linear plane fit. Note that `initial_plane_rmse` is
///   recorded as `None` in this case (no fit RMSE is meaningful for a manual seed).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LaserlineDeviceManualInit {
    /// Manual intrinsics seed.
    pub intrinsics: Option<FxFyCxCySkew<Real>>,
    /// Manual distortion seed.
    pub distortion: Option<BrownConrady5<Real>>,
    /// Manual per-view poses (`camera_se3_target`).
    pub poses: Option<Vec<Iso3>>,
    /// Manual laser plane seed (camera-frame `(n̂, d)`).
    pub plane: Option<LaserPlane>,
}

/// Options for the optimization step.
#[derive(Debug, Clone, Default)]
pub struct DeviceOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

fn board_and_pixel_points(view: &CorrespondenceView) -> (Vec<Pt2>, Vec<Pt2>) {
    let board_2d: Vec<Pt2> = view.points_3d.iter().map(|p| Pt2::new(p.x, p.y)).collect();
    let pixel_2d: Vec<Pt2> = view.points_2d.iter().map(|v| Pt2::new(v.x, v.y)).collect();
    (board_2d, pixel_2d)
}

fn planar_dataset_from_input(
    input: &[vision_calibration_optim::LaserlineView],
) -> Result<PlanarDataset, Error> {
    let views: Vec<View<NoMeta>> = input
        .iter()
        .map(|view| View::without_meta(view.obs.clone()))
        .collect();
    PlanarDataset::new(views).map_err(Error::Core)
}

fn estimate_poses(
    input: &[vision_calibration_optim::LaserlineView],
    intrinsics: &FxFyCxCySkew<vision_calibration_core::Real>,
) -> Result<Vec<Iso3>, Error> {
    let k_matrix = intrinsics.k_matrix();
    let mut poses = Vec::with_capacity(input.len());
    for (idx, view) in input.iter().enumerate() {
        let (board_2d, pixel_2d) = board_and_pixel_points(&view.obs);
        let h = dlt_homography(&board_2d, &pixel_2d).map_err(|e| {
            Error::numerical(format!("failed to compute homography for view {idx}: {e}"))
        })?;
        let pose = estimate_planar_pose_from_h(&k_matrix, &h)
            .map_err(|e| Error::numerical(format!("failed to recover pose for view {idx}: {e}")))?;
        poses.push(pose);
    }
    Ok(poses)
}

fn linear_plane_init<Sm>(
    input: &[vision_calibration_optim::LaserlineView],
    camera: &Camera<
        vision_calibration_core::Real,
        Pinhole,
        vision_calibration_core::BrownConrady5<vision_calibration_core::Real>,
        Sm,
        FxFyCxCySkew<vision_calibration_core::Real>,
    >,
    poses: &[Iso3],
) -> Result<(vision_calibration_optim::LaserPlane, f64), Error>
where
    Sm: SensorModel<vision_calibration_core::Real>,
{
    let views: Vec<LinearLaserlineView> = input
        .iter()
        .zip(poses.iter())
        .map(|(view, pose)| LinearLaserlineView {
            laser_pixels: view.meta.laser_pixels.clone(),
            camera_se3_target: *pose,
        })
        .collect();

    let estimate = LaserlinePlaneSolver::from_views(&views, camera)
        .map_err(|e| Error::numerical(format!("laser plane init failed: {e}")))?;
    let plane =
        vision_calibration_optim::LaserPlane::new(estimate.normal.into_inner(), estimate.distance);
    Ok((plane, estimate.rmse))
}

fn update_state_with_stats(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    stats: &LaserlineStats,
    final_cost: f64,
) {
    session.state.final_cost = Some(final_cost);
    session.state.mean_reproj_error = Some(stats.mean_reproj_error);
    session.state.mean_laser_error = Some(stats.mean_laser_error);
    session.state.per_view_reproj_errors = Some(stats.per_view_reproj_errors.clone());
    session.state.per_view_laser_errors = Some(stats.per_view_laser_errors.clone());
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics, distortion, per-view poses, and laser plane from any
/// combination of manual seeds and auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate that
/// passes `LaserlineDeviceManualInit::default()` (all-`None`, full auto path).
///
/// See [`LaserlineDeviceManualInit`] for partial-seed semantics. The sensor model
/// is always taken from `session.config.init.sensor_init` regardless.
///
/// # Errors
///
/// - Input not set, or fewer than 3 views.
/// - Auto-init computation fails (homography / Zhang's / linear plane).
/// - `manual.poses` is `Some` but its length does not match the view count.
pub fn step_set_init(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    manual: LaserlineDeviceManualInit,
    opts: Option<DeviceInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    let view_count = input.len();

    let opts = opts.unwrap_or_default();
    let mut init_opts = session.config.init_opts();
    if let Some(iters) = opts.iterations {
        init_opts.iterations = iters;
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    let (intrinsics, distortion) = if let Some(k) = manual.intrinsics {
        manual_fields.push("intrinsics");
        let d = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                BrownConrady5::default()
            }
        };
        (k, d)
    } else {
        auto_fields.push("intrinsics");
        let planar_dataset = planar_dataset_from_input(input)?;
        let camera_init = estimate_intrinsics_iterative(&planar_dataset, init_opts)
            .map_err(|e| Error::numerical(format!("intrinsics initialization failed: {e}")))?;
        let d = match manual.distortion {
            Some(d) => {
                manual_fields.push("distortion");
                d
            }
            None => {
                auto_fields.push("distortion");
                camera_init.dist
            }
        };
        (camera_init.k, d)
    };

    let poses = match manual.poses {
        Some(p) => {
            manual_fields.push("poses");
            if p.len() != view_count {
                let msg = format!(
                    "manual poses count ({}) does not match view count ({})",
                    p.len(),
                    view_count
                );
                session.log_failure("init", msg.clone());
                return Err(Error::invalid_input(msg));
            }
            p
        }
        None => {
            auto_fields.push("poses");
            estimate_poses(input, &intrinsics)?
        }
    };

    let sensor = session.config.init.sensor_init;
    let camera = Camera::new(Pinhole, distortion, sensor.compile(), intrinsics);

    let (plane, plane_rmse) = match manual.plane {
        Some(p) => {
            manual_fields.push("plane");
            (p, None)
        }
        None => {
            auto_fields.push("plane");
            let (p, rmse) = linear_plane_init(input, &camera, &poses)?;
            (p, Some(rmse))
        }
    };

    let initial_params = LaserlineParams::new(intrinsics, distortion, sensor, poses, plane)?;

    session.state.initial_params = Some(initial_params);
    session.state.initial_plane_rmse = plane_rmse;
    session.state.clear_optimization();

    let source = format_init_source(&manual_fields, &auto_fields);
    let plane_note = match plane_rmse {
        Some(rmse) => format!("plane_rmse={:.4}", rmse),
        None => "plane=manual".to_string(),
    };
    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, {} {}",
            intrinsics.fx, intrinsics.fy, plane_note, source
        ),
    );

    Ok(())
}

fn format_init_source(manual: &[&str], auto: &[&str]) -> String {
    match (manual.is_empty(), auto.is_empty()) {
        (false, false) => format!(
            "(manual: {}; auto: {})",
            manual.join(", "),
            auto.join(", ")
        ),
        (false, true) => format!("(manual: {})", manual.join(", ")),
        (true, false) => format!("(auto: {})", auto.join(", ")),
        (true, true) => "(empty)".to_string(),
    }
}

/// Initialize intrinsics, poses, and laser plane from observations using full
/// auto-init.
///
/// Convenience wrapper around [`step_set_init`] with
/// `LaserlineDeviceManualInit::default()`.
pub fn step_init(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    opts: Option<DeviceInitOptions>,
) -> Result<(), Error> {
    step_set_init(session, LaserlineDeviceManualInit::default(), opts)
}

/// Optimize laserline calibration using non-linear bundle adjustment.
pub fn step_optimize(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    opts: Option<DeviceOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    let initial = session
        .state
        .initial_params
        .clone()
        .ok_or_else(|| Error::not_available("initial params (call step_init first)"))?;

    let opts = opts.unwrap_or_default();
    let solve_opts = session.config.solve_opts();
    let mut backend_opts = session.config.backend_opts();
    if let Some(max_iters) = opts.max_iters {
        backend_opts.max_iters = max_iters;
    }
    if let Some(verbosity) = opts.verbosity {
        backend_opts.verbosity = verbosity;
    }

    let result = optimize_laserline(input, &initial, &solve_opts, &backend_opts)?;

    let stats = compute_laserline_stats(input, &result.params, solve_opts.laser_residual_type)?;

    update_state_with_stats(session, &stats, result.report.final_cost);

    let output = LaserlineDeviceOutput {
        estimate: result.clone(),
        stats: stats.clone(),
    };
    session.set_output(output);

    session.log_success_with_notes(
        "optimize",
        format!(
            "mean reproj={:.3}px, mean laser={:.3}",
            stats.mean_reproj_error, stats.mean_laser_error
        ),
    );

    Ok(())
}

/// Run full calibration pipeline: init → optimize.
pub fn run_calibration(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    config: Option<LaserlineDeviceConfig>,
) -> Result<(), Error> {
    if let Some(cfg) = config {
        session.set_config(cfg)?;
    }
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}
