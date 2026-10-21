//! Step functions for single laserline device calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraModel, CameraModelSpec, CameraParams, FxFyCxCy, Iso3, NoMeta,
    PlanarDataset, Real, ScheimpflugParams, SensorKind, SensorParams, View,
};
use vision_calibration_linear::laserline::{
    LaserlinePlaneSolver, LaserlineView as LinearLaserlineView,
};
use vision_calibration_optim::{
    LaserPlane, LaserlineParams, LaserlineStats, PlanarIntrinsicsParams, compute_laserline_stats,
    optimize_laserline,
};

use crate::planar_family::{CameraSeed, bootstrap_camera, check_sensor_seed, optimize_camera};
use crate::rig_family::format_init_source;
use crate::session::CalibrationSession;

use super::problem::{LaserlineDeviceConfig, LaserlineDeviceOutput, LaserlineDeviceProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

/// Options for the initialization step.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
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
/// The camera model is `session.config.camera.model`; the seeds are values in
/// it. Partial-seed semantics:
/// - `intrinsics: Some` skips Zhang's method (and, for a Scheimpflug sensor,
///   the linear tilt sweep). Distortion starts at zero unless also seeded;
///   poses recover from per-view homographies using the manual intrinsics.
/// - `distortion` seeds Brown-Conrady coefficients, embedded into the
///   camera's distortion model (extra coefficients start at zero).
/// - `sensor` seeds the tilt of a Scheimpflug camera; a seeded tilt replaces
///   the linear tilt sweep. An untilted camera takes only a zero tilt (a
///   non-zero one is a typed error).
/// - `plane: Some` skips the linear plane fit. Note that `initial_plane_rmse` is
///   recorded as `None` in this case (no fit RMSE is meaningful for a manual seed).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct LaserlineDeviceManualInit {
    /// Manual intrinsics seed.
    pub intrinsics: Option<FxFyCxCy<Real>>,
    /// Manual Brown-Conrady distortion seed.
    pub distortion: Option<BrownConrady5<Real>>,
    /// Manual sensor tilt seed (Scheimpflug camera; zero otherwise).
    pub sensor: Option<ScheimpflugParams>,
    /// Manual per-view poses (`camera_se3_target`).
    pub poses: Option<Vec<Iso3>>,
    /// Manual laser plane seed (camera-frame `(n̂, d)`).
    pub plane: Option<LaserPlane>,
}

/// Options for the optimization step.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DeviceOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

/// Typed return value of [`step_init`] / [`step_init_with_seed`].
///
/// Carries the seeded-or-fitted initial estimates. The same values are also
/// written into `session.state`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LaserlineDeviceInitResult {
    /// Initial camera, in the configured model.
    pub camera: CameraParams,
    /// Initial per-view target poses (`camera_se3_target`).
    pub poses: Vec<Iso3>,
    /// Initial laser plane (camera-frame `(n̂, d)`).
    pub plane: LaserPlane,
    /// RMSE of the linear laser plane fit (meters); `None` when the plane was
    /// supplied as a manual seed.
    pub plane_rmse: Option<f64>,
}

/// Typed return value of [`step_optimize`].
///
/// Aggregates the optimization metrics of the laserline-device solve; the same
/// values are also written into `session.state`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LaserlineDeviceOptimizeResult {
    /// Final cost reported by the non-linear solver.
    pub final_cost: f64,
    /// Mean reprojection error in pixels.
    pub mean_reproj_error: f64,
    /// Mean laser residual (units depend on the selected residual type).
    pub mean_laser_error: f64,
    /// Per-view mean reprojection errors in pixels.
    pub per_view_reproj_errors: Vec<f64>,
    /// Per-view mean laser residuals.
    pub per_view_laser_errors: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

fn planar_dataset_from_input(
    input: &[vision_calibration_optim::LaserlineView],
) -> Result<PlanarDataset, Error> {
    let views: Vec<View<NoMeta>> = input
        .iter()
        .map(|view| View::without_meta(view.obs.clone()))
        .collect();
    PlanarDataset::new(views).map_err(Error::Core)
}

fn linear_plane_init(
    input: &[vision_calibration_optim::LaserlineView],
    camera: &CameraModel,
    poses: &[Iso3],
) -> Result<(LaserPlane, f64), Error> {
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
    let plane = LaserPlane::new(estimate.normal.into_inner(), estimate.distance);
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

/// Initialize the camera, per-view poses, and laser plane from any
/// combination of manual seeds and auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate that
/// passes `LaserlineDeviceManualInit::default()` (all-`None`, full auto path).
///
/// The camera's model is `session.config.camera.model`. A held Scheimpflug
/// tilt (the default, see `LaserlineDeviceConfig::holds_tilt`) is the
/// `sensor` seed, or zero, and the camera initializes as untilted. An
/// estimated tilt without a seed comes from the linear tilt sweep, then a
/// camera-only solve on the target views through the staged tilt multi-start
/// (a cold `tilt = 0` start falls into the tilt / principal-point /
/// distortion degeneracy). See [`LaserlineDeviceManualInit`] for
/// partial-seed semantics.
///
/// # Errors
///
/// - Input not set, or fewer than 3 views.
/// - A non-zero `manual.sensor` for an untilted camera.
/// - Auto-init computation fails (homography / Zhang's / linear plane).
/// - `manual.poses` is `Some` but its length does not match the view count.
pub fn step_init_with_seed(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    manual: LaserlineDeviceManualInit,
    opts: Option<DeviceInitOptions>,
) -> Result<LaserlineDeviceInitResult, Error> {
    session.validate()?;
    let input = session.require_input()?;
    let view_count = input.len();
    let model = session.config.camera.model;

    let opts = opts.unwrap_or_default();
    let mut init_opts = session.config.init_opts();
    if let Some(iters) = opts.iterations {
        init_opts.iterations = iters;
    }

    if let Some(sensor) = &manual.sensor
        && let Err(e) = check_sensor_seed(model, sensor, "sensor")
    {
        session.log_failure("init", e.to_string());
        return Err(e);
    }
    if let Some(p) = &manual.poses
        && p.len() != view_count
    {
        let msg = format!(
            "manual poses count ({}) does not match view count ({})",
            p.len(),
            view_count
        );
        session.log_failure("init", msg.clone());
        return Err(Error::invalid_input(msg));
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();
    let mut note = |name: &'static str, seeded: bool| {
        if seeded {
            manual_fields.push(name);
        } else {
            auto_fields.push(name);
        }
    };
    note("intrinsics", manual.intrinsics.is_some());
    note("distortion", manual.distortion.is_some());
    if model.sensor == SensorKind::Scheimpflug {
        note("sensor", manual.sensor.is_some());
    }
    note("poses", manual.poses.is_some());
    note("plane", manual.plane.is_some());

    let planar_dataset = planar_dataset_from_input(input)?;
    let seed = CameraSeed {
        intrinsics: manual.intrinsics,
        distortion: manual.distortion,
        sensor: manual.sensor,
    };
    let holds_tilt = session.config.holds_tilt();
    let bootstrap = if holds_tilt {
        // A held tilt is known: initialize as an untilted camera, then
        // attach the tilt (the seed, or zero).
        let untilted = CameraModelSpec {
            sensor: SensorKind::Identity,
            ..model
        };
        let seed = CameraSeed {
            sensor: None,
            ..seed
        };
        let mut boot = bootstrap_camera(0, untilted, &planar_dataset, &seed, init_opts)?;
        boot.camera.sensor = SensorParams::Scheimpflug {
            params: manual.sensor.unwrap_or_default(),
        };
        boot
    } else {
        bootstrap_camera(0, model, &planar_dataset, &seed, init_opts)?
    };
    let (camera, bootstrap_poses) =
        if model.sensor == SensorKind::Scheimpflug && !holds_tilt && manual.sensor.is_none() {
            // Cold-start tilt: the camera alone, through the staged tilt
            // multi-start, before the plane fit needs its rays.
            let initial = PlanarIntrinsicsParams::new(bootstrap.camera, bootstrap.poses)?;
            let refined = optimize_camera(
                0,
                &planar_dataset,
                &initial,
                &session.config.camera,
                &session.config.solver,
                session.config.solver.backend_options(),
            )?;
            (refined.params.camera, refined.params.camera_se3_target)
        } else {
            (bootstrap.camera, bootstrap.poses)
        };
    let poses = manual.poses.unwrap_or(bootstrap_poses);

    let (plane, plane_rmse) = match manual.plane {
        Some(p) => (p, None),
        None => {
            let model = camera
                .build()
                .map_err(|e| Error::invalid_input(format!("camera model: {e}")))?;
            let (p, rmse) = linear_plane_init(input, &model, &poses)?;
            (p, Some(rmse))
        }
    };

    let initial_params = LaserlineParams::new(camera.clone(), poses.clone(), plane.clone())?;

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
            camera.intrinsics.fx, camera.intrinsics.fy, plane_note, source
        ),
    );

    Ok(LaserlineDeviceInitResult {
        camera,
        poses,
        plane,
        plane_rmse,
    })
}

/// Initialize intrinsics, poses, and laser plane from observations using full
/// auto-init.
///
/// Convenience wrapper around [`step_init_with_seed`] with
/// `LaserlineDeviceManualInit::default()`.
pub fn step_init(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    opts: Option<DeviceInitOptions>,
) -> Result<LaserlineDeviceInitResult, Error> {
    step_init_with_seed(session, LaserlineDeviceManualInit::default(), opts)
}

/// Optimize laserline calibration using non-linear bundle adjustment.
pub fn step_optimize(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    opts: Option<DeviceOptimizeOptions>,
) -> Result<LaserlineDeviceOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?;

    let initial = session
        .state
        .initial_params
        .clone()
        .ok_or_else(|| Error::not_available("initial params (call step_init first)"))?;
    let configured = session.config.camera.model;
    if initial.camera.spec() != Some(configured) {
        return Err(Error::invalid_input(format!(
            "the camera model changed since step_init (now {configured:?}); run it again"
        )));
    }

    let opts = opts.unwrap_or_default();
    let solve_opts = session.config.solve_opts();
    let mut backend_opts = session.config.solver.backend_options();
    if let Some(max_iters) = opts.max_iters {
        backend_opts.max_iters = max_iters;
    }
    if let Some(verbosity) = opts.verbosity {
        backend_opts.verbosity = verbosity;
    }

    let result = optimize_laserline(input, &initial, &solve_opts, &backend_opts)?;

    let stats = compute_laserline_stats(input, &result.params, solve_opts.laser_residual_type)?;

    let final_cost = result.report.final_cost;
    update_state_with_stats(session, &stats, final_cost);

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

    Ok(LaserlineDeviceOptimizeResult {
        final_cost,
        mean_reproj_error: stats.mean_reproj_error,
        mean_laser_error: stats.mean_laser_error,
        per_view_reproj_errors: stats.per_view_reproj_errors,
        per_view_laser_errors: stats.per_view_laser_errors,
    })
}

/// Run full calibration pipeline: init → optimize.
pub fn run_calibration(
    session: &mut CalibrationSession<LaserlineDeviceProblem>,
    config: Option<LaserlineDeviceConfig>,
) -> Result<(), Error> {
    if let Some(cfg) = config {
        session.set_config(cfg)?;
    }
    let _ = step_init(session, None)?;
    let _ = step_optimize(session, None)?;
    Ok(())
}
