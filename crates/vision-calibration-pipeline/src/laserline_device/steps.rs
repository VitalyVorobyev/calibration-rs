//! Step functions for single laserline device calibration.

use anyhow::{Context, Result, anyhow};
use vision_calibration_core::{
    Camera, CorrespondenceView, FxFyCxCySkew, Iso3, NoMeta, Pinhole, PlanarDataset, Pt2,
    SensorModel, View,
};
use vision_calibration_linear::laserline::{
    LaserlinePlaneSolver, LaserlineView as LinearLaserlineView,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    LaserlineParams, LaserlineStats, compute_laserline_stats, optimize_laserline,
};

use crate::session::CalibrationSession;

use super::problem::{LaserlineDeviceConfig, LaserlineDeviceOutput, LaserlineDeviceProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

/// Options for the initialization step.
#[derive(Debug, Clone, Default)]
pub struct InitOptions {
    /// Override the number of iterations for iterative intrinsics estimation.
    pub iterations: Option<usize>,
}

/// Options for the optimization step.
#[derive(Debug, Clone, Default)]
pub struct OptimizeOptions {
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
) -> Result<PlanarDataset> {
    let views: Vec<View<NoMeta>> = input
        .iter()
        .map(|view| View::without_meta(view.obs.clone()))
        .collect();
    PlanarDataset::new(views)
}

fn estimate_poses(
    input: &[vision_calibration_optim::LaserlineView],
    intrinsics: &FxFyCxCySkew<vision_calibration_core::Real>,
) -> Result<Vec<Iso3>> {
    let k_matrix = intrinsics.k_matrix();
    let mut poses = Vec::with_capacity(input.len());
    for (idx, view) in input.iter().enumerate() {
        let (board_2d, pixel_2d) = board_and_pixel_points(&view.obs);
        let h = dlt_homography(&board_2d, &pixel_2d).with_context(|| {
            format!(
                "failed to compute homography for view {} (need >=4 well-conditioned points)",
                idx
            )
        })?;
        let pose = estimate_planar_pose_from_h(&k_matrix, &h)
            .with_context(|| format!("failed to recover pose for view {}", idx))?;
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
) -> Result<(vision_calibration_optim::LaserPlane, f64)>
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

    let estimate =
        LaserlinePlaneSolver::from_views(&views, camera).context("laser plane init failed")?;
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

/// Initialize intrinsics, poses, and laser plane from observations.
pub fn step_init(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    opts: Option<InitOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    let opts = opts.unwrap_or_default();
    let mut init_opts = session.config.init_opts();
    if let Some(iters) = opts.iterations {
        init_opts.iterations = iters;
    }

    let planar_dataset = planar_dataset_from_input(input)?;
    let camera_init = estimate_intrinsics_iterative(&planar_dataset, init_opts)
        .context("intrinsics initialization failed")?;

    let poses = estimate_poses(input, &camera_init.k).context("pose initialization failed")?;

    let sensor = session.config.init.sensor_init;
    let camera = Camera::new(Pinhole, camera_init.dist, sensor.compile(), camera_init.k);

    let (plane, plane_rmse) = linear_plane_init(input, &camera, &poses)?;

    let initial_params =
        LaserlineParams::new(camera_init.k, camera_init.dist, sensor, poses, plane)?;

    session.state.initial_params = Some(initial_params);
    session.state.initial_plane_rmse = Some(plane_rmse);
    session.state.clear_optimization();

    session.log_success_with_notes(
        "init",
        format!(
            "fx={:.1}, fy={:.1}, plane_rmse={:.4}",
            camera_init.k.fx, camera_init.k.fy, plane_rmse
        ),
    );

    Ok(())
}

/// Optimize laserline calibration using non-linear bundle adjustment.
pub fn step_optimize(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    opts: Option<OptimizeOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    let initial = session
        .state
        .initial_params
        .clone()
        .ok_or_else(|| anyhow!("initialization required before optimization"))?;

    let opts = opts.unwrap_or_default();
    let solve_opts = session.config.solve_opts();
    let mut backend_opts = session.config.backend_opts();
    if let Some(max_iters) = opts.max_iters {
        backend_opts.max_iters = max_iters;
    }
    if let Some(verbosity) = opts.verbosity {
        backend_opts.verbosity = verbosity;
    }

    let result = optimize_laserline(input, &initial, &solve_opts, &backend_opts)
        .context("laserline optimization failed")?;

    let stats = compute_laserline_stats(input, &result.params, solve_opts.laser_residual_type)
        .context("failed to compute laserline stats")?;

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
) -> Result<()> {
    if let Some(cfg) = config {
        session.set_config(cfg)?;
    }
    step_init(session, None)?;
    step_optimize(session, None)?;
    Ok(())
}
