//! Step functions for Scheimpflug rig extrinsics calibration.

use crate::Error;
use vision_calibration_core::{
    CameraFixMask, DistortionFixMask, IntrinsicsFixMask, Iso3, NoMeta, PlanarDataset,
    ScheimpflugParams, View, make_pinhole_camera,
};
use vision_calibration_linear::estimate_extrinsics_from_cam_target_poses;
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, RigExtrinsicsScheimpflugParams, RigExtrinsicsScheimpflugSolveOptions,
    ScheimpflugFixMask, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsSolveOptions,
    optimize_rig_extrinsics_scheimpflug, optimize_scheimpflug_intrinsics,
};

use crate::session::CalibrationSession;

use super::problem::{RigScheimpflugExtrinsicsInput, RigScheimpflugExtrinsicsProblem};

/// Options for per-camera intrinsics initialization.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsInitOptions {
    /// Override the number of iterations.
    pub iterations: Option<usize>,
}

/// Options for per-camera intrinsics optimization.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Options for rig BA optimization.
#[derive(Debug, Clone, Default)]
pub struct RigOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

fn extract_camera_views(
    input: &RigScheimpflugExtrinsicsInput,
    cam_idx: usize,
) -> Vec<Option<View<NoMeta>>> {
    input
        .views
        .iter()
        .map(|view| {
            view.obs
                .cameras
                .get(cam_idx)
                .and_then(|opt_obs| opt_obs.as_ref())
                .map(|obs| View::without_meta(obs.clone()))
        })
        .collect()
}

fn views_to_planar_dataset(
    views: &[Option<View<NoMeta>>],
) -> Result<(PlanarDataset, Vec<usize>), Error> {
    let (valid_views, indices): (Vec<_>, Vec<_>) = views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|view| (view.clone(), i)))
        .unzip();
    if valid_views.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: valid_views.len(),
        });
    }
    let dataset = PlanarDataset::new(valid_views)?;
    Ok((dataset, indices))
}

fn estimate_target_pose(
    k_matrix: &vision_calibration_core::Mat3,
    obs: &vision_calibration_core::CorrespondenceView,
) -> Result<Iso3, Error> {
    let board_2d: Vec<vision_calibration_core::Pt2> = obs
        .points_3d
        .iter()
        .map(|p| vision_calibration_core::Pt2::new(p.x, p.y))
        .collect();
    let pixel_2d: Vec<vision_calibration_core::Pt2> = obs
        .points_2d
        .iter()
        .map(|v| vision_calibration_core::Pt2::new(v.x, v.y))
        .collect();
    let h = dlt_homography(&board_2d, &pixel_2d)
        .map_err(|e| Error::numerical(format!("failed to compute homography: {e}")))?;
    estimate_planar_pose_from_h(k_matrix, &h)
        .map_err(|e| Error::numerical(format!("failed to recover pose from homography: {e}")))
}

/// Initialize intrinsics for all cameras (Zhang + zero Scheimpflug tilts).
///
/// # Errors
///
/// Returns [`Error`] if the input is missing or any camera has insufficient views.
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigScheimpflugExtrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(config.intrinsics_init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: config.fix_k3,
            fix_tangential: config.fix_tangential,
            iters: 8,
        },
        zero_skew: config.zero_skew,
    };

    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    let mut per_cam_intrinsics = Vec::with_capacity(num_cameras);
    let mut per_cam_sensors = Vec::with_capacity(num_cameras);
    let mut per_cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];

    #[allow(clippy::needless_range_loop)]
    for cam_idx in 0..num_cameras {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let camera = estimate_intrinsics_iterative(&planar_dataset, init_opts).map_err(|e| {
            Error::numerical(format!(
                "intrinsics estimation failed for camera {cam_idx}: {e}"
            ))
        })?;

        let k_matrix = vision_calibration_core::Mat3::new(
            camera.k.fx,
            camera.k.skew,
            camera.k.cx,
            0.0,
            camera.k.fy,
            camera.k.cy,
            0.0,
            0.0,
            1.0,
        );
        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            let view = &planar_dataset.views[local_idx];
            let pose = estimate_target_pose(&k_matrix, &view.obs).map_err(|e| {
                Error::numerical(format!(
                    "pose estimation failed for cam {cam_idx} view {global_idx}: {e}"
                ))
            })?;
            per_cam_target_poses[global_idx][cam_idx] = Some(pose);
        }

        per_cam_intrinsics.push(make_pinhole_camera(camera.k, camera.dist));
        per_cam_sensors.push(ScheimpflugParams {
            tilt_x: config.init_tilt_x,
            tilt_y: config.init_tilt_y,
        });
    }

    session.state.per_cam_intrinsics = Some(per_cam_intrinsics);
    session.state.per_cam_sensors = Some(per_cam_sensors);
    session.state.per_cam_target_poses = Some(per_cam_target_poses);

    session.log_success_with_notes(
        "intrinsics_init_all",
        format!("initialized {num_cameras} cameras"),
    );
    Ok(())
}

/// Optimize Scheimpflug intrinsics per camera.
///
/// # Errors
///
/// Returns [`Error`] if intrinsics initialization has not run or the per-camera
/// solver fails.
pub fn step_intrinsics_optimize_all(
    session: &mut CalibrationSession<RigScheimpflugExtrinsicsProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_init_all first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let per_cam_intrinsics = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    let per_cam_sensors = session
        .state
        .per_cam_sensors
        .clone()
        .ok_or_else(|| Error::not_available("per-camera sensors"))?;
    let mut per_cam_target_poses = session
        .state
        .per_cam_target_poses
        .clone()
        .ok_or_else(|| Error::not_available("per-camera target poses"))?;

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut optimized_sensors = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);

    let fix_intrinsics = IntrinsicsFixMask::default();
    // Radial-only distortion (k1, k2 free; k3, p1, p2 fixed). Tangential
    // distortion can absorb tilt-like geometric signal and interfere with
    // Scheimpflug tilt optimization.
    let fix_distortion = DistortionFixMask::radial_only();

    for cam_idx in 0..input.num_cameras {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let initial_poses: Vec<Iso3> = valid_indices
            .iter()
            .map(|&global_idx| {
                per_cam_target_poses[global_idx][cam_idx]
                    .ok_or_else(|| Error::not_available("initial pose"))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let cam = &per_cam_intrinsics[cam_idx];
        let initial_params = ScheimpflugIntrinsicsParams::new(
            cam.k,
            cam.dist,
            per_cam_sensors[cam_idx],
            initial_poses,
        )?;

        let solve_opts = ScheimpflugIntrinsicsSolveOptions {
            robust_loss: config.robust_loss,
            fix_intrinsics,
            fix_distortion,
            fix_scheimpflug: config.fix_scheimpflug_in_intrinsics,
            fix_poses: vec![0],
        };

        let backend_opts = BackendSolveOptions {
            max_iters: opts.max_iters.unwrap_or(config.max_iters),
            verbosity: opts.verbosity.unwrap_or(config.verbosity),
            ..Default::default()
        };

        let result = optimize_scheimpflug_intrinsics(
            &planar_dataset,
            &initial_params,
            solve_opts,
            backend_opts,
        )
        .map_err(|e| {
            Error::numerical(format!(
                "scheimpflug intrinsics optimization failed for camera {cam_idx}: {e}"
            ))
        })?;

        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            per_cam_target_poses[global_idx][cam_idx] =
                Some(result.params.camera_se3_target[local_idx]);
        }

        optimized_cameras.push(make_pinhole_camera(
            result.params.intrinsics,
            result.params.distortion,
        ));
        optimized_sensors.push(result.params.sensor);
        per_cam_reproj_errors.push(result.mean_reproj_error);
    }

    session.state.per_cam_intrinsics = Some(optimized_cameras);
    session.state.per_cam_sensors = Some(optimized_sensors);
    session.state.per_cam_target_poses = Some(per_cam_target_poses);
    session.state.per_cam_reproj_errors = Some(per_cam_reproj_errors.clone());

    let avg_error: f64 =
        per_cam_reproj_errors.iter().sum::<f64>() / per_cam_reproj_errors.len() as f64;
    session.log_success_with_notes(
        "intrinsics_optimize_all",
        format!("avg_reproj_err={avg_error:.3}px"),
    );
    Ok(())
}

/// Initialize rig extrinsics from per-camera target poses (linear).
///
/// # Errors
///
/// Returns [`Error`] if intrinsics have not been computed, or if rig
/// initialization fails (e.g., insufficient overlap).
pub fn step_rig_init(
    session: &mut CalibrationSession<RigScheimpflugExtrinsicsProblem>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_optimize_all first)",
        ));
    }

    let num_views = input.num_views();
    let reference_camera_idx = session.config.reference_camera_idx;

    let per_cam_target_poses = session
        .state
        .per_cam_target_poses
        .clone()
        .ok_or_else(|| Error::not_available("per-camera target poses"))?;

    let extrinsic_result =
        estimate_extrinsics_from_cam_target_poses(&per_cam_target_poses, reference_camera_idx)
            .map_err(|e| Error::numerical(format!("rig extrinsics initialization failed: {e}")))?;

    let cam_se3_rig: Vec<Iso3> = extrinsic_result
        .cam_to_rig
        .iter()
        .map(|t| t.inverse())
        .collect();
    session.state.initial_cam_se3_rig = Some(cam_se3_rig);
    session.state.initial_rig_se3_target = Some(extrinsic_result.rig_from_target);

    session.log_success_with_notes(
        "rig_init",
        format!("ref_cam={reference_camera_idx}, {num_views} views"),
    );
    Ok(())
}

/// Optimize Scheimpflug rig extrinsics via bundle adjustment.
///
/// # Errors
///
/// Returns [`Error`] if init steps have not run or the solver fails.
pub fn step_rig_optimize(
    session: &mut CalibrationSession<RigScheimpflugExtrinsicsProblem>,
    opts: Option<RigOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?.clone();

    if !session.state.has_rig_init() {
        return Err(Error::not_available("rig init (call step_rig_init first)"));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let cameras = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    let sensors = session
        .state
        .per_cam_sensors
        .clone()
        .ok_or_else(|| Error::not_available("per-camera sensors"))?;
    let cam_se3_rig = session
        .state
        .initial_cam_se3_rig
        .clone()
        .ok_or_else(|| Error::not_available("initial cam_se3_rig"))?;
    let cam_to_rig: Vec<Iso3> = cam_se3_rig.iter().map(|t| t.inverse()).collect();
    let rig_from_target = session
        .state
        .initial_rig_se3_target
        .clone()
        .ok_or_else(|| Error::not_available("initial rig_se3_target"))?;

    let initial = RigExtrinsicsScheimpflugParams {
        cameras,
        sensors,
        cam_to_rig,
        rig_from_target,
    };

    let default_fix = if config.refine_intrinsics_in_rig_ba {
        CameraFixMask::default()
    } else {
        CameraFixMask::all_fixed()
    };
    let default_scheimpflug_fix = if config.refine_scheimpflug_in_rig_ba {
        ScheimpflugFixMask::default()
    } else {
        ScheimpflugFixMask {
            tilt_x: true,
            tilt_y: true,
        }
    };

    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == config.reference_camera_idx)
        .collect();
    let fix_rig_poses = if config.fix_first_rig_pose {
        vec![0]
    } else {
        Vec::new()
    };

    let solve_opts = RigExtrinsicsScheimpflugSolveOptions {
        robust_loss: config.robust_loss,
        default_fix,
        camera_overrides: Vec::new(),
        default_scheimpflug_fix,
        scheimpflug_overrides: Vec::new(),
        fix_extrinsics,
        fix_rig_poses,
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.verbosity),
        ..Default::default()
    };

    let result = match optimize_rig_extrinsics_scheimpflug(input, initial, solve_opts, backend_opts)
    {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("rig_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    session.state.rig_ba_final_cost = Some(result.report.final_cost);
    session.state.rig_ba_reproj_error = Some(result.mean_reproj_error);
    session.state.rig_ba_per_cam_reproj_errors = Some(result.per_cam_reproj_errors.clone());

    let mean = result.mean_reproj_error;
    session.set_output(result.clone());

    session.log_success_with_notes(
        "rig_optimize",
        format!(
            "final_cost={:.2e}, mean_reproj_err={mean:.3}px",
            result.report.final_cost
        ),
    );
    Ok(())
}

/// Run the full Scheimpflug rig extrinsics calibration pipeline.
///
/// # Errors
///
/// Returns [`Error`] from any of the constituent steps.
pub fn run_calibration(
    session: &mut CalibrationSession<RigScheimpflugExtrinsicsProblem>,
) -> Result<(), Error> {
    step_intrinsics_init_all(session, None)?;
    step_intrinsics_optimize_all(session, None)?;
    step_rig_init(session)?;
    step_rig_optimize(session, None)?;
    Ok(())
}
