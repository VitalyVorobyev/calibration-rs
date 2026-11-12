//! Step functions for rig laserline calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::LaserPlane;
use vision_calibration_linear::laserline::{LaserlinePlaneSolver, PlaneFitView};
use vision_calibration_optim::{
    LaserlineSolveOptions, LaserlineStats, RigLaserlineSolveOptions, RigLaserlineUpstream,
    optimize_rig_laserline,
};

use crate::rig_family::inverse_poses;
use crate::session::CalibrationSession;

use super::problem::RigLaserlineDeviceProblem;

/// Options for the laser-plane initialization step.
///
/// The plane fit has no per-call overrides yet; the struct keeps `step_init`
/// in the same shape as every other step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct PlaneInitOptions {}

/// Options for the laser-plane optimization step.
///
/// These options override session config for a single step invocation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct PlaneOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Manual seeds for rig laserline device calibration.
///
/// The upstream rig calibration (cameras, cam_se3_rig, rig_se3_target) is
/// part of `input.upstream` and is *not* a manual-init concern
/// — it's an input contract. Only the laser planes are seedable here.
///
/// When `laser_planes_cam` is `Some`, the linear plane fit is skipped entirely. Length must equal
/// `input.dataset.num_cameras`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PlaneManualInit {
    /// Per-camera laser planes (camera frame). `None` runs the linear fit.
    pub laser_planes_cam: Option<Vec<LaserPlane>>,
}

/// Typed return value of [`step_init`] / [`step_init_with_seed`].
///
/// Carries the seeded-or-fitted initial planes. The same planes are also
/// written into `session.state`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PlaneInitResult {
    /// Initial per-camera laser planes (camera frame).
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Per camera, the RMSE of the linear plane fit (meters); `None` for a
    /// plane supplied as a manual seed.
    pub plane_rmse: Vec<Option<f64>>,
}

/// Typed return value of [`step_optimize`].
///
/// The refined planes and the per-camera statistics; the planes are also the
/// session's output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PlaneOptimizeResult {
    /// Per-camera laser plane in the rig frame.
    pub laser_planes_rig: Vec<LaserPlane>,
    /// Per-camera laser plane in the camera's own frame.
    pub laser_planes_cam: Vec<LaserPlane>,
    /// Mean target reprojection error in pixels, averaged over the cameras.
    pub mean_reproj_error: f64,
    /// Per-camera mean reprojection error and laser residual.
    pub per_cam_stats: Vec<LaserlineStats>,
}

/// Initialize per-camera laser planes from any combination of manual seeds and
/// auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate with
/// `PlaneManualInit::default()`.
///
/// The laser planes are `manual.laser_planes_cam` if `Some`, else a closed-form
/// linear fit per camera.
///
/// # Errors
///
/// - Input not set.
/// - `manual.laser_planes_cam.len() != input.dataset.num_cameras`.
/// - Without a seed, a camera whose linear fit fails (no laser pixels, or
///   pixels that do not determine a plane): the error names it.
pub fn step_init_with_seed(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
    manual: PlaneManualInit,
    opts: Option<PlaneInitOptions>,
) -> Result<PlaneInitResult, Error> {
    let PlaneInitOptions {} = opts.unwrap_or_default();
    session.validate()?;
    let input = session.require_input()?;

    if let Some(p) = &manual.laser_planes_cam
        && p.len() != input.dataset.num_cameras
    {
        let msg = format!(
            "manual laser_planes_cam length ({}) != num_cameras ({})",
            p.len(),
            input.dataset.num_cameras
        );
        session.log_failure("init", msg.clone());
        return Err(Error::invalid_input(msg));
    }

    let (planes, plane_rmse, source): (Vec<LaserPlane>, Vec<Option<f64>>, &'static str) =
        match manual.laser_planes_cam {
            Some(p) => {
                let rmse = vec![None; p.len()];
                (p, rmse, "(manual: laser_planes_cam)")
            }
            None => {
                let fitted: Result<Vec<(LaserPlane, f64)>, Error> = (0..input.dataset.num_cameras)
                    .map(|cam_idx| linear_plane_init(input, cam_idx))
                    .collect();
                match fitted {
                    Ok(fits) => {
                        let (planes, rmse): (Vec<_>, Vec<_>) = fits.into_iter().unzip();
                        (
                            planes,
                            rmse.into_iter().map(Some).collect(),
                            "(auto: linear fit)",
                        )
                    }
                    Err(err) => {
                        session.log_failure("init", err.to_string());
                        return Err(err);
                    }
                }
            }
        };

    session.state.initial_planes_cam = Some(planes.clone());
    session.log_success_with_notes("init", format!("initial planes set {source}"));
    Ok(PlaneInitResult {
        laser_planes_cam: planes,
        plane_rmse,
    })
}

/// Initialize per-camera laser planes by the linear fit.
///
/// Convenience wrapper around [`step_init_with_seed`] with default seeds.
pub fn step_init(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
    opts: Option<PlaneInitOptions>,
) -> Result<PlaneInitResult, Error> {
    step_init_with_seed(session, PlaneManualInit::default(), opts)
}

/// The closed-form plane of camera `cam_idx` from its laser pixels on the
/// upstream's target poses, with the RMSE of the fit (meters).
///
/// # Errors
///
/// [`Error::InvalidInput`] naming the camera when its model does not build,
/// it saw no laser pixels, or its pixels do not determine a plane (a seed in
/// `PlaneManualInit::laser_planes_cam` then starts the solve).
fn linear_plane_init(
    input: &super::problem::RigLaserlineDeviceInput,
    cam_idx: usize,
) -> Result<(LaserPlane, f64), Error> {
    let rig_to_cam = input.upstream.cam_se3_rig[cam_idx];
    let camera = input.upstream.cameras[cam_idx]
        .build()
        .map_err(|e| Error::invalid_input(format!("camera {cam_idx}: camera model: {e}")))?;
    let mut views = Vec::new();
    for (view_idx, view) in input.dataset.views.iter().enumerate() {
        let Some(laser_pixels) = view
            .laser_pixels
            .get(cam_idx)
            .and_then(|slot| slot.clone())
            .filter(|pixels| !pixels.is_empty())
        else {
            continue;
        };
        let rig_se3_target = input.upstream.rig_se3_target[view_idx];
        views.push(PlaneFitView {
            laser_pixels,
            cam_se3_target: rig_to_cam * rig_se3_target,
        });
    }
    if views.is_empty() {
        return Err(Error::invalid_input(format!(
            "camera {cam_idx}: no view has laser pixels, so its laser plane cannot be \
             initialized; seed it in PlaneManualInit::laser_planes_cam"
        )));
    }
    let estimate = LaserlinePlaneSolver::from_views(&views, &camera).map_err(|e| {
        Error::invalid_input(format!(
            "camera {cam_idx}: the linear laser-plane fit failed ({e}); seed the plane \
             in PlaneManualInit::laser_planes_cam"
        ))
    })?;
    let plane = LaserPlane::new(estimate.normal.into_inner(), estimate.distance)
        .map_err(|e| Error::invalid_input(format!("camera {cam_idx}: laser plane: {e}")))?;
    Ok((plane, estimate.rmse))
}

/// Optimize per-camera laser planes and express them in rig frame.
///
/// # Errors
///
/// Returns [`Error`] if init has not been run or the optim solver fails.
pub fn step_optimize(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
    opts: Option<PlaneOptimizeOptions>,
) -> Result<PlaneOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?.clone();

    if !session.state.has_initial_planes() {
        return Err(Error::not_available(
            "initial planes (call step_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let cfg = &session.config;

    let initial_planes = session.state.initial_planes_cam.clone().unwrap();

    let upstream = RigLaserlineUpstream {
        cameras: input.upstream.cameras,
        rig_se3_cam: inverse_poses(&input.upstream.cam_se3_rig),
        rig_se3_target: input.upstream.rig_se3_target,
        uncertainty: input.upstream.uncertainty,
    };

    let solve_opts = RigLaserlineSolveOptions {
        laserline: LaserlineSolveOptions {
            laser_loss: cfg.optimize.laser_loss,
            ..LaserlineSolveOptions::default()
        },
        laser_residual_type: cfg.optimize.laser_residual_type,
    };
    let backend_opts =
        cfg.solver
            .step_backend_options(session.control(), opts.max_iters, opts.verbosity);

    let estimate = match optimize_rig_laserline(
        &input.dataset,
        &upstream,
        &initial_planes,
        &solve_opts,
        &backend_opts,
    ) {
        Ok(e) => e,
        Err(e) => {
            session.log_failure("optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    let mean_reproj: f64 = estimate
        .per_cam_stats
        .iter()
        .map(|s| s.mean_reproj_error)
        .sum::<f64>()
        / estimate.per_cam_stats.len().max(1) as f64;
    session.set_output(estimate.clone());
    session.log_success_with_notes(
        "optimize",
        format!(
            "{} planes calibrated, mean_reproj={mean_reproj:.3}px",
            estimate.laser_planes_rig.len()
        ),
    );
    Ok(PlaneOptimizeResult {
        laser_planes_rig: estimate.laser_planes_rig,
        laser_planes_cam: estimate.laser_planes_cam,
        mean_reproj_error: mean_reproj,
        per_cam_stats: estimate.per_cam_stats,
    })
}

/// Run full rig laserline calibration (init + optimize).
///
/// # Errors
///
/// Returns [`Error`] from any constituent step.
pub fn run_calibration(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
) -> Result<(), Error> {
    session.begin_step("init")?;
    step_init(session, None)?;
    session.begin_step("optimize")?;
    step_optimize(session, None)?;
    Ok(())
}
