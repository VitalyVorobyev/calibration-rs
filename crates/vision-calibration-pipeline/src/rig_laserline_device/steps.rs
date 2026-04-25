//! Step functions for rig laserline calibration.

use crate::Error;
use nalgebra::Vector3;
use vision_calibration_core::{Camera, Pinhole};
use vision_calibration_linear::laserline::{
    LaserlinePlaneSolver, LaserlineView as LinearLaserlineView,
};
use vision_calibration_optim::{
    BackendSolveOptions, LaserPlane, RigLaserlineSolveOptions, RigLaserlineUpstream,
    optimize_rig_laserline,
};

use crate::session::CalibrationSession;

use super::problem::RigLaserlineDeviceProblem;

/// Options controlling both step functions.
#[derive(Debug, Clone, Default)]
pub struct StepOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Initialize per-camera laser planes in camera frame.
///
/// If the input provides `initial_planes_cam`, they are stored as-is.
/// Otherwise a closed-form plane fit is computed from the frozen upstream
/// poses, with a generic default plane as a fallback for degenerate inputs.
///
/// # Errors
///
/// Returns [`Error`] if the session has no input.
pub fn step_init(session: &mut CalibrationSession<RigLaserlineDeviceProblem>) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    let planes: Vec<LaserPlane> = match &input.initial_planes_cam {
        Some(p) => p.clone(),
        None => {
            let mut planes = Vec::with_capacity(input.dataset.num_cameras);
            for cam_idx in 0..input.dataset.num_cameras {
                let default_plane = LaserPlane::new(Vector3::new(0.0, 0.0, 1.0), -0.2);
                let plane = linear_plane_init(input, cam_idx).unwrap_or(default_plane);
                planes.push(plane);
            }
            planes
        }
    };

    session.state.initial_planes_cam = Some(planes);
    session.log_success_with_notes(
        "init",
        "initial planes set (linear fit with default fallback)".to_string(),
    );
    Ok(())
}

fn linear_plane_init(
    input: &super::problem::RigLaserlineDeviceInput,
    cam_idx: usize,
) -> Option<LaserPlane> {
    let rig_to_cam = input.upstream.cam_se3_rig[cam_idx];
    let camera = Camera::new(
        Pinhole,
        input.upstream.distortion[cam_idx],
        input.upstream.sensors[cam_idx].compile(),
        input.upstream.intrinsics[cam_idx],
    );
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
        views.push(LinearLaserlineView {
            laser_pixels,
            camera_se3_target: rig_to_cam * rig_se3_target,
        });
    }
    let estimate = LaserlinePlaneSolver::from_views(&views, &camera).ok()?;
    Some(LaserPlane::new(
        estimate.normal.into_inner(),
        estimate.distance,
    ))
}

/// Optimize per-camera laser planes and express them in rig frame.
///
/// # Errors
///
/// Returns [`Error`] if init has not been run or the optim solver fails.
pub fn step_optimize(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
    opts: Option<StepOptions>,
) -> Result<(), Error> {
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
        intrinsics: input.upstream.intrinsics,
        distortion: input.upstream.distortion,
        sensors: input.upstream.sensors,
        cam_to_rig: input
            .upstream
            .cam_se3_rig
            .iter()
            .map(|t| t.inverse())
            .collect(),
        rig_se3_target: input.upstream.rig_se3_target,
    };

    let solve_opts = RigLaserlineSolveOptions {
        laser_residual_type: cfg.laser_residual_type,
        ..RigLaserlineSolveOptions::default()
    };
    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.or(cfg.max_iters).unwrap_or(100),
        verbosity: opts.verbosity.or(cfg.verbosity).unwrap_or(0),
        ..Default::default()
    };

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
        .per_camera_stats
        .iter()
        .map(|s| s.mean_reproj_error)
        .sum::<f64>()
        / estimate.per_camera_stats.len().max(1) as f64;
    session.set_output(estimate.clone());
    session.log_success_with_notes(
        "optimize",
        format!(
            "{} planes calibrated, mean_reproj={mean_reproj:.3}px",
            estimate.laser_planes_rig.len()
        ),
    );
    Ok(())
}

/// Run full rig laserline calibration (init + optimize).
///
/// # Errors
///
/// Returns [`Error`] from any constituent step.
pub fn run_calibration(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
) -> Result<(), Error> {
    step_init(session)?;
    step_optimize(session, None)?;
    Ok(())
}
