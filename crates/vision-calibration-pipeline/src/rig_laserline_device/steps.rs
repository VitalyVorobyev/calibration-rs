//! Step functions for rig laserline calibration.

use crate::Error;
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
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

/// Manual seeds for rig laserline device calibration.
///
/// The upstream rig calibration (intrinsics, distortion, sensors, cam_se3_rig,
/// rig_se3_target) is part of `input.upstream` and is *not* a manual-init concern
/// — it's an input contract. Only the laser planes are seedable here.
///
/// When `planes_cam` is `Some`, it overrides any `input.initial_planes_cam` and
/// the linear plane fit is skipped entirely. Length must equal
/// `input.dataset.num_cameras`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigLaserlineDeviceManualInit {
    /// Per-camera laser planes (camera frame). Overrides input-supplied seeds.
    pub planes_cam: Option<Vec<LaserPlane>>,
}

/// Initialize per-camera laser planes from any combination of manual seeds and
/// auto-estimation.
///
/// This is the load-bearing init function. [`step_init`] is a thin delegate with
/// `RigLaserlineDeviceManualInit::default()`.
///
/// Resolution order for the laser planes:
/// 1. `manual.planes_cam` if `Some` (highest priority).
/// 2. `input.initial_planes_cam` if `Some`.
/// 3. Closed-form linear fit per camera, with a generic fallback plane for
///    degenerate cases.
///
/// # Errors
///
/// - Input not set.
/// - `manual.planes_cam.len() != input.dataset.num_cameras`.
pub fn step_set_init(
    session: &mut CalibrationSession<RigLaserlineDeviceProblem>,
    manual: RigLaserlineDeviceManualInit,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if let Some(p) = &manual.planes_cam
        && p.len() != input.dataset.num_cameras
    {
        let msg = format!(
            "manual planes_cam length ({}) != num_cameras ({})",
            p.len(),
            input.dataset.num_cameras
        );
        session.log_failure("init", msg.clone());
        return Err(Error::invalid_input(msg));
    }

    let (planes, source): (Vec<LaserPlane>, &'static str) =
        match (manual.planes_cam, &input.initial_planes_cam) {
            (Some(p), _) => (p, "(manual: planes_cam)"),
            (None, Some(p)) => (p.clone(), "(input: initial_planes_cam)"),
            (None, None) => {
                let mut planes = Vec::with_capacity(input.dataset.num_cameras);
                for cam_idx in 0..input.dataset.num_cameras {
                    let default_plane = LaserPlane::new(Vector3::new(0.0, 0.0, 1.0), -0.2);
                    let plane = linear_plane_init(input, cam_idx).unwrap_or(default_plane);
                    planes.push(plane);
                }
                (planes, "(auto: linear fit with default fallback)")
            }
        };

    session.state.initial_planes_cam = Some(planes);
    session.log_success_with_notes("init", format!("initial planes set {source}"));
    Ok(())
}

/// Initialize per-camera laser planes using the input-or-auto path.
///
/// Convenience wrapper around [`step_set_init`] with default seeds.
pub fn step_init(session: &mut CalibrationSession<RigLaserlineDeviceProblem>) -> Result<(), Error> {
    step_set_init(session, RigLaserlineDeviceManualInit::default())
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
