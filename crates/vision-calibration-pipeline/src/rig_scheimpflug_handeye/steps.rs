//! Step functions for Scheimpflug rig hand-eye calibration.
//!
//! Supports both `EyeInHand` and `EyeToHand` modes, selected via
//! `RigScheimpflugHandeyeInitConfig::handeye_mode`.

use crate::Error;
use vision_calibration_core::{
    CameraFixMask, DistortionFixMask, IntrinsicsFixMask, Iso3, NoMeta, PlanarDataset,
    ScheimpflugParams, View, make_pinhole_camera,
};
use vision_calibration_linear::{
    estimate_extrinsics_from_cam_target_poses, estimate_gripper_se3_target_dlt,
    estimate_handeye_dlt, prelude::*,
};
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeMode, HandEyeScheimpflugDataset, HandEyeScheimpflugParams,
    HandEyeScheimpflugSolveOptions, RigExtrinsicsScheimpflugParams,
    RigExtrinsicsScheimpflugSolveOptions, ScheimpflugFixMask, ScheimpflugIntrinsicsParams,
    ScheimpflugIntrinsicsSolveOptions, optimize_handeye_scheimpflug,
    optimize_rig_extrinsics_scheimpflug, optimize_scheimpflug_intrinsics,
};

use crate::session::CalibrationSession;

use super::problem::{RigScheimpflugHandeyeInput, RigScheimpflugHandeyeProblem};

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

/// Options for hand-eye init.
#[derive(Debug, Clone, Default)]
pub struct HandeyeInitOptions {
    /// Minimum motion angle override (degrees).
    pub min_motion_angle_deg: Option<f64>,
}

/// Options for hand-eye optimization.
#[derive(Debug, Clone, Default)]
pub struct HandeyeOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

fn extract_camera_views(
    input: &RigScheimpflugHandeyeInput,
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
    let dataset = PlanarDataset::new(valid_views).map_err(Error::Core)?;
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

/// Initialize Scheimpflug intrinsics for all cameras.
///
/// Source of each camera's initial intrinsics:
/// 1. `config.intrinsics.initial_cameras[i]` if set (skips Zhang).
/// 2. Zhang's method with iterative distortion refinement.
/// 3. If Zhang fails and `config.intrinsics.fallback_to_shared_init` is `true`,
///    reuse the most recent successful camera's intrinsics (intended for
///    homogeneous rigs where all cameras share the same optical design).
///
/// Per-view target poses are always computed from the resulting `K` via a DLT
/// homography + metric recovery; failures on individual views are reported
/// but do not abort the step.
///
/// # Errors
///
/// Returns [`Error`] if:
/// - the input is missing,
/// - `initial_cameras` length does not match `num_cameras`,
/// - *all* cameras fail and no fallback is available.
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    let opts = opts.unwrap_or_default();
    let cfg = &session.config;

    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(cfg.intrinsics.init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: cfg.intrinsics.fix_k3,
            fix_tangential: cfg.intrinsics.fix_tangential,
            iters: 8,
        },
        zero_skew: cfg.intrinsics.zero_skew,
    };

    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    if let Some(override_cams) = &cfg.intrinsics.initial_cameras
        && override_cams.len() != num_cameras
    {
        return Err(Error::InvalidInput {
            reason: format!(
                "initial_cameras has {} entries, expected {num_cameras}",
                override_cams.len()
            ),
        });
    }
    if let Some(override_sensors) = &cfg.intrinsics.initial_sensors
        && override_sensors.len() != num_cameras
    {
        return Err(Error::InvalidInput {
            reason: format!(
                "initial_sensors has {} entries, expected {num_cameras}",
                override_sensors.len()
            ),
        });
    }

    let default_sensor = ScheimpflugParams {
        tilt_x: cfg.intrinsics.init_tilt_x,
        tilt_y: cfg.intrinsics.init_tilt_y,
    };

    // Per-camera Zhang + poses; None when init fails and we may backfill later.
    let mut per_cam_intrinsics: Vec<Option<vision_calibration_core::PinholeCamera>> =
        vec![None; num_cameras];
    let mut per_cam_sensors: Vec<Option<ScheimpflugParams>> = vec![None; num_cameras];
    let mut per_cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];
    let mut per_cam_zhang_failure: Vec<Option<String>> = vec![None; num_cameras];

    for cam_idx in 0..num_cameras {
        // If an explicit override is provided for this camera, use it and skip
        // Zhang but still solve per-view target poses via DLT.
        if let Some(overrides) = cfg.intrinsics.initial_cameras.as_ref() {
            let camera = overrides[cam_idx].clone();
            let sensor = cfg
                .intrinsics
                .initial_sensors
                .as_ref()
                .map(|s| s[cam_idx])
                .unwrap_or(default_sensor);
            solve_per_view_poses(input, cam_idx, &camera, &mut per_cam_target_poses);
            per_cam_intrinsics[cam_idx] = Some(camera);
            per_cam_sensors[cam_idx] = Some(sensor);
            continue;
        }

        let cam_views = extract_camera_views(input, cam_idx);
        let planar_result = views_to_planar_dataset(&cam_views);
        let (planar_dataset, valid_indices) = match planar_result {
            Ok(v) => v,
            Err(e) => {
                per_cam_zhang_failure[cam_idx] = Some(format!("too few views: {e}"));
                continue;
            }
        };

        let camera = match estimate_intrinsics_iterative(&planar_dataset, init_opts) {
            Ok(c) => c,
            Err(e) => {
                per_cam_zhang_failure[cam_idx] = Some(format!("Zhang failed: {e}"));
                continue;
            }
        };
        // Sanity-check Zhang output: a degenerate solve can return
        // non-positive or tiny focal lengths that still pass the inner
        // linear-algebra checks. Treat this as a failure so the fallback
        // kicks in.
        if !(camera.k.fx > 1.0
            && camera.k.fy > 1.0
            && camera.k.fx.is_finite()
            && camera.k.fy.is_finite())
        {
            per_cam_zhang_failure[cam_idx] = Some(format!(
                "Zhang produced degenerate intrinsics fx={}, fy={}",
                camera.k.fx, camera.k.fy
            ));
            continue;
        }
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
            if let Ok(pose) = estimate_target_pose(&k_matrix, &view.obs) {
                per_cam_target_poses[global_idx][cam_idx] = Some(pose);
            }
        }
        per_cam_intrinsics[cam_idx] = Some(make_pinhole_camera(camera.k, camera.dist));
        per_cam_sensors[cam_idx] = Some(
            cfg.intrinsics
                .initial_sensors
                .as_ref()
                .map(|s| s[cam_idx])
                .unwrap_or(default_sensor),
        );
    }

    // Backfill failed cameras from the first successful camera, if enabled.
    let first_ok = per_cam_intrinsics
        .iter()
        .position(Option::is_some)
        .map(|i| {
            (
                per_cam_intrinsics[i].clone().unwrap(),
                per_cam_sensors[i].unwrap_or(default_sensor),
                i,
            )
        });

    let mut fallbacks = Vec::new();
    let mut per_cam_used_fallback = vec![false; num_cameras];
    if let Some((fallback_cam, fallback_sensor, donor_idx)) = &first_ok {
        for cam_idx in 0..num_cameras {
            if per_cam_intrinsics[cam_idx].is_some() {
                continue;
            }
            if !cfg.intrinsics.fallback_to_shared_init {
                break;
            }
            // Solve per-view target poses with the fallback intrinsics so the
            // downstream rig_init has something to work with for this camera.
            solve_per_view_poses(input, cam_idx, fallback_cam, &mut per_cam_target_poses);
            per_cam_intrinsics[cam_idx] = Some(fallback_cam.clone());
            per_cam_sensors[cam_idx] = Some(*fallback_sensor);
            per_cam_used_fallback[cam_idx] = true;
            fallbacks.push((cam_idx, *donor_idx));
        }
    }

    // Finalize: if any camera is still None, we cannot proceed.
    let mut final_cameras = Vec::with_capacity(num_cameras);
    let mut final_sensors = Vec::with_capacity(num_cameras);
    for cam_idx in 0..num_cameras {
        let cam = per_cam_intrinsics[cam_idx].clone().ok_or_else(|| {
            let reason = per_cam_zhang_failure[cam_idx]
                .clone()
                .unwrap_or_else(|| "unknown".to_string());
            Error::Numerical(format!(
                "camera {cam_idx} has no initial intrinsics ({reason}); provide \
                `intrinsics.initial_cameras[{cam_idx}]` or enable `fallback_to_shared_init`"
            ))
        })?;
        let sensor = per_cam_sensors[cam_idx].unwrap_or(default_sensor);
        final_cameras.push(cam);
        final_sensors.push(sensor);
    }

    session.state.per_cam_intrinsics = Some(final_cameras);
    session.state.per_cam_sensors = Some(final_sensors);
    session.state.per_cam_target_poses = Some(per_cam_target_poses);
    session.state.per_cam_used_fallback = Some(per_cam_used_fallback);
    let failures: Vec<_> = per_cam_zhang_failure
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.as_ref().map(|m| (i, m.clone())))
        .collect();
    let notes = if failures.is_empty() && fallbacks.is_empty() {
        format!("initialized {num_cameras} cameras")
    } else {
        format!(
            "initialized {num_cameras} cameras ({} fallback: {:?}; failures: {:?})",
            fallbacks.len(),
            fallbacks,
            failures.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        )
    };
    session.log_success_with_notes("intrinsics_init_all", notes);
    Ok(())
}

fn solve_per_view_poses(
    input: &RigScheimpflugHandeyeInput,
    cam_idx: usize,
    camera: &vision_calibration_core::PinholeCamera,
    per_cam_target_poses: &mut [Vec<Option<Iso3>>],
) {
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
    for (view_idx, view) in input.views.iter().enumerate() {
        let Some(obs_opt) = view.obs.cameras.get(cam_idx) else {
            continue;
        };
        let Some(obs) = obs_opt else { continue };
        if let Ok(pose) = estimate_target_pose(&k_matrix, obs) {
            per_cam_target_poses[view_idx][cam_idx] = Some(pose);
        }
    }
}

/// Optimize Scheimpflug intrinsics for all cameras.
///
/// # Errors
///
/// Returns [`Error`] if intrinsics init has not run or per-camera refinement fails.
pub fn step_intrinsics_optimize_all(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
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
    let cfg = &session.config;

    let per_cam_intrinsics = session.state.per_cam_intrinsics.clone().unwrap();
    let per_cam_sensors = session.state.per_cam_sensors.clone().unwrap();
    let mut per_cam_target_poses = session.state.per_cam_target_poses.clone().unwrap();

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut optimized_sensors = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);

    // Use configured per-camera masks when initial cameras are supplied but
    // not fixed; when Zhang's method provided the seed, the same mask is used
    // (acts as an all-free default unless the caller overrides).
    let fix_intrinsics_default = cfg.intrinsics.fix_intrinsics_in_percam_ba;
    let fix_distortion_default = cfg.intrinsics.fix_distortion_in_percam_ba;
    let fix_intrinsics_override = IntrinsicsFixMask::all_fixed();
    let fix_distortion_override = DistortionFixMask::all_fixed();

    let mut per_cam_failures: Vec<(usize, String)> = Vec::new();
    let per_cam_used_fallback = session
        .state
        .per_cam_used_fallback
        .clone()
        .unwrap_or_else(|| vec![false; input.num_cameras]);

    for cam_idx in 0..input.num_cameras {
        // Cameras that used fallback intrinsics have target poses derived from
        // a shared-K DLT; refining them per-camera can drag the intrinsics and
        // extrinsics into a bad local minimum. Skip the per-camera BA and let
        // the downstream rig BA handle it with a consistent gauge.
        if per_cam_used_fallback[cam_idx] {
            optimized_cameras.push(per_cam_intrinsics[cam_idx].clone());
            optimized_sensors.push(per_cam_sensors[cam_idx]);
            per_cam_reproj_errors.push(f64::NAN);
            continue;
        }
        let cam_views = extract_camera_views(input, cam_idx);
        let planar_result = views_to_planar_dataset(&cam_views);
        let (planar_dataset, valid_indices) = match planar_result {
            Ok(v) => v,
            Err(e) => {
                per_cam_failures.push((cam_idx, format!("insufficient views: {e}")));
                optimized_cameras.push(per_cam_intrinsics[cam_idx].clone());
                optimized_sensors.push(per_cam_sensors[cam_idx]);
                per_cam_reproj_errors.push(f64::NAN);
                continue;
            }
        };

        // Filter to views where we have both a correspondence and a valid
        // initial pose. Without this, a single DLT failure on a fallback-only
        // camera would abort the whole step.
        let mut kept_local: Vec<usize> = Vec::new();
        let mut kept_global: Vec<usize> = Vec::new();
        let mut initial_poses: Vec<Iso3> = Vec::new();
        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            if let Some(pose) = per_cam_target_poses[global_idx][cam_idx] {
                kept_local.push(local_idx);
                kept_global.push(global_idx);
                initial_poses.push(pose);
            }
        }
        if initial_poses.len() < 3 {
            per_cam_failures.push((
                cam_idx,
                format!("only {} views with initial poses", initial_poses.len()),
            ));
            optimized_cameras.push(per_cam_intrinsics[cam_idx].clone());
            optimized_sensors.push(per_cam_sensors[cam_idx]);
            per_cam_reproj_errors.push(f64::NAN);
            continue;
        }
        // Rebuild a filtered planar dataset matching kept_local ordering.
        let filtered_views: Vec<_> = kept_local
            .iter()
            .map(|&i| planar_dataset.views[i].clone())
            .collect();
        let planar_dataset = match PlanarDataset::new(filtered_views) {
            Ok(d) => d,
            Err(e) => {
                per_cam_failures.push((cam_idx, format!("planar dataset build: {e}")));
                optimized_cameras.push(per_cam_intrinsics[cam_idx].clone());
                optimized_sensors.push(per_cam_sensors[cam_idx]);
                per_cam_reproj_errors.push(f64::NAN);
                continue;
            }
        };
        let valid_indices = kept_global;

        let cam = &per_cam_intrinsics[cam_idx];
        let initial_params = ScheimpflugIntrinsicsParams::new(
            cam.k,
            cam.dist,
            per_cam_sensors[cam_idx],
            initial_poses,
        )?;

        let is_overridden = cfg.intrinsics.initial_cameras.is_some()
            && cfg.intrinsics.fix_intrinsics_when_overridden;
        let solve_opts = ScheimpflugIntrinsicsSolveOptions {
            robust_loss: cfg.solver.robust_loss,
            fix_intrinsics: if is_overridden {
                fix_intrinsics_override
            } else {
                fix_intrinsics_default
            },
            fix_distortion: if is_overridden {
                fix_distortion_override
            } else {
                fix_distortion_default
            },
            fix_scheimpflug: cfg.intrinsics.fix_scheimpflug,
            fix_poses: vec![0],
        };
        let backend_opts = BackendSolveOptions {
            max_iters: opts.max_iters.unwrap_or(cfg.solver.max_iters),
            verbosity: opts.verbosity.unwrap_or(cfg.solver.verbosity),
            ..Default::default()
        };

        let result = match optimize_scheimpflug_intrinsics(
            &planar_dataset,
            &initial_params,
            solve_opts,
            backend_opts,
        ) {
            Ok(r) => r,
            Err(e) => {
                per_cam_failures.push((cam_idx, format!("solver: {e}")));
                // Keep the initial intrinsics + sensors; let rig BA refine.
                optimized_cameras.push(per_cam_intrinsics[cam_idx].clone());
                optimized_sensors.push(per_cam_sensors[cam_idx]);
                per_cam_reproj_errors.push(f64::NAN);
                continue;
            }
        };
        // Post-optim sanity check: if the solver ran to completion but
        // produced degenerate intrinsics (fx/fy near zero or non-finite) or a
        // blow-up in reprojection error, fall back to the initial values.
        // This happens on cameras where the per-view DLT poses come from
        // a wrong fallback K and BA follows the bad local geometry.
        let fx = result.params.intrinsics.fx;
        let fy = result.params.intrinsics.fy;
        let reproj = result.mean_reproj_error;
        let degenerate =
            !(fx > 1.0 && fy > 1.0 && fx.is_finite() && fy.is_finite() && reproj.is_finite())
                || reproj > 50.0;
        if degenerate {
            per_cam_failures.push((
                cam_idx,
                format!("optim degenerate (fx={fx:.3}, fy={fy:.3}, reproj={reproj:.3})"),
            ));
            optimized_cameras.push(per_cam_intrinsics[cam_idx].clone());
            optimized_sensors.push(per_cam_sensors[cam_idx]);
            per_cam_reproj_errors.push(f64::NAN);
            continue;
        }
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
    let finite: Vec<f64> = per_cam_reproj_errors
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let avg = if finite.is_empty() {
        f64::NAN
    } else {
        finite.iter().sum::<f64>() / finite.len() as f64
    };
    let notes = if per_cam_failures.is_empty() {
        format!("avg_reproj={avg:.3}px")
    } else {
        format!(
            "avg_reproj={avg:.3}px (failures: {:?})",
            per_cam_failures.iter().map(|(i, _)| *i).collect::<Vec<_>>()
        )
    };
    session.log_success_with_notes("intrinsics_optimize_all", notes);
    Ok(())
}

/// Initialize rig extrinsics from per-camera target poses.
///
/// # Errors
///
/// Returns [`Error`] if intrinsics have not been computed, or if linear init fails.
pub fn step_rig_init(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    if !session.state.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_optimize_all first)",
        ));
    }
    let num_views = input.num_views();
    let reference_camera_idx = session.config.rig.reference_camera_idx;
    let per_cam_target_poses = session.state.per_cam_target_poses.clone().unwrap();

    let result =
        estimate_extrinsics_from_cam_target_poses(&per_cam_target_poses, reference_camera_idx)
            .map_err(|e| Error::numerical(format!("rig init failed: {e}")))?;

    let cam_se3_rig: Vec<Iso3> = result.cam_to_rig.iter().map(|t| t.inverse()).collect();
    session.state.initial_cam_se3_rig = Some(cam_se3_rig);
    session.state.initial_rig_se3_target = Some(result.rig_from_target);
    session.log_success_with_notes(
        "rig_init",
        format!("ref_cam={reference_camera_idx}, {num_views} views"),
    );
    Ok(())
}

/// Optimize rig extrinsics with Scheimpflug bundle adjustment.
///
/// # Errors
///
/// Returns [`Error`] if rig init has not been run or the solver fails.
pub fn step_rig_optimize(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
    opts: Option<RigOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    if !session.state.has_rig_init() {
        return Err(Error::not_available("rig init (call step_rig_init first)"));
    }
    let opts = opts.unwrap_or_default();
    let cfg = &session.config;

    let cameras = session.state.per_cam_intrinsics.clone().unwrap();
    let sensors = session.state.per_cam_sensors.clone().unwrap();
    let cam_se3_rig = session.state.initial_cam_se3_rig.clone().unwrap();
    let cam_to_rig: Vec<Iso3> = cam_se3_rig.iter().map(|t| t.inverse()).collect();
    let rig_from_target = session.state.initial_rig_se3_target.clone().unwrap();

    let initial = RigExtrinsicsScheimpflugParams {
        cameras,
        sensors,
        cam_to_rig,
        rig_from_target,
    };

    let default_fix = if cfg.rig.refine_intrinsics_in_rig_ba {
        CameraFixMask::default()
    } else {
        CameraFixMask::all_fixed()
    };
    let default_scheimpflug_fix = if cfg.rig.refine_scheimpflug_in_rig_ba {
        ScheimpflugFixMask::default()
    } else {
        ScheimpflugFixMask {
            tilt_x: true,
            tilt_y: true,
        }
    };
    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == cfg.rig.reference_camera_idx)
        .collect();
    let fix_rig_poses = if cfg.rig.fix_first_rig_pose {
        vec![0]
    } else {
        Vec::new()
    };

    let solve_opts = RigExtrinsicsScheimpflugSolveOptions {
        robust_loss: cfg.solver.robust_loss,
        default_fix,
        camera_overrides: Vec::new(),
        default_scheimpflug_fix,
        scheimpflug_overrides: Vec::new(),
        fix_extrinsics,
        fix_rig_poses,
    };
    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(cfg.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(cfg.solver.verbosity),
        ..Default::default()
    };

    // Convert RigDataset<RobotPoseMeta> → RigDataset<NoMeta> for the rig extr solver.
    let rig_no_meta: vision_calibration_core::RigDataset<NoMeta> =
        vision_calibration_core::RigDataset::new(
            input
                .views
                .iter()
                .map(|v| vision_calibration_core::RigView {
                    meta: NoMeta,
                    obs: v.obs.clone(),
                })
                .collect(),
            input.num_cameras,
        )?;

    let result =
        match optimize_rig_extrinsics_scheimpflug(rig_no_meta, initial, solve_opts, backend_opts) {
            Ok(r) => r,
            Err(e) => {
                session.log_failure("rig_optimize", e.to_string());
                return Err(Error::from(e));
            }
        };

    let cam_se3_rig: Vec<Iso3> = result
        .params
        .cam_to_rig
        .iter()
        .map(|t| t.inverse())
        .collect();
    session.state.per_cam_intrinsics = Some(result.params.cameras.clone());
    session.state.per_cam_sensors = Some(result.params.sensors.clone());
    session.state.rig_ba_cam_se3_rig = Some(cam_se3_rig);
    session.state.rig_ba_rig_se3_target = Some(result.params.rig_from_target);
    session.state.rig_ba_reproj_error = Some(result.mean_reproj_error);
    session.state.rig_ba_per_cam_reproj_errors = Some(result.per_cam_reproj_errors.clone());
    session.log_success_with_notes(
        "rig_optimize",
        format!(
            "final_cost={:.2e}, mean_reproj={:.3}px",
            result.report.final_cost, result.mean_reproj_error
        ),
    );
    Ok(())
}

/// Initialize hand-eye transform via Tsai-Lenz.
///
/// Mode-dependent semantics:
///
/// - `EyeInHand`: solves for `gripper_se3_rig` (T_G_R) with a fixed target in
///   the base; recovers `base_se3_target` (T_B_T) from the first view.
/// - `EyeToHand`: solves for `gripper_se3_target` (T_G_T) with a target on the
///   gripper and rig fixed in base; recovers `rig_se3_base` (T_R_B) from the
///   first view.
///
/// # Errors
///
/// Returns [`Error`] if rig BA has not run or linear hand-eye estimation fails.
pub fn step_handeye_init(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    if !session.state.has_rig_optimized() {
        return Err(Error::not_available(
            "rig optimization (call step_rig_optimize first)",
        ));
    }
    let opts = opts.unwrap_or_default();
    let handeye_mode = session.config.handeye_init.handeye_mode;
    let min_angle = opts
        .min_motion_angle_deg
        .unwrap_or(session.config.handeye_init.min_motion_angle_deg);

    let robot_poses: Vec<Iso3> = input
        .views
        .iter()
        .map(|v| v.meta.base_se3_gripper)
        .collect();
    let rig_se3_target = session.state.rig_ba_rig_se3_target.clone().unwrap();

    let result = match handeye_mode {
        HandEyeMode::EyeInHand => {
            // estimate_handeye_dlt expects target_se3_rig = rig_se3_target.inverse().
            let target_se3_rig: Vec<Iso3> = rig_se3_target.iter().map(|t| t.inverse()).collect();
            estimate_handeye_dlt(&robot_poses, &target_se3_rig, min_angle).map(|gripper_se3_rig| {
                // base_se3_target = base_se3_gripper * gripper_se3_rig * rig_se3_target
                let base_se3_target = robot_poses[0] * gripper_se3_rig * rig_se3_target[0];
                (gripper_se3_rig, base_se3_target)
            })
        }
        HandEyeMode::EyeToHand => {
            // Target is on the gripper; solve for the fixed gripper_se3_target
            // (T_G_T), then recover rig_se3_base (T_R_B) from the first view.
            estimate_gripper_se3_target_dlt(&robot_poses, &rig_se3_target, min_angle).map(
                |gripper_se3_target| {
                    // T_R_T = T_R_B * T_B_G * T_G_T  =>  T_R_B = T_R_T * (T_B_G * T_G_T)^-1
                    let rig_se3_base =
                        rig_se3_target[0] * (robot_poses[0] * gripper_se3_target).inverse();
                    (rig_se3_base, gripper_se3_target)
                },
            )
        }
    };
    let (handeye, mode_target_pose) = match result {
        Ok(v) => v,
        Err(e) => {
            session.log_failure("handeye_init", e.to_string());
            return Err(Error::numerical(format!(
                "linear hand-eye estimation failed: {e}"
            )));
        }
    };

    session.state.initial_handeye = Some(handeye);
    session.state.initial_mode_target_pose = Some(mode_target_pose);
    session.log_success_with_notes(
        "handeye_init",
        format!(
            "mode={:?} |t|={:.4}m",
            handeye_mode,
            handeye.translation.vector.norm()
        ),
    );
    Ok(())
}

/// Optimize Scheimpflug hand-eye calibration.
///
/// # Errors
///
/// Returns [`Error`] if hand-eye init has not run or the solver fails.
pub fn step_handeye_optimize(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
    opts: Option<HandeyeOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?.clone();
    if !session.state.has_handeye_init() {
        return Err(Error::not_available(
            "hand-eye init (call step_handeye_init first)",
        ));
    }
    let opts = opts.unwrap_or_default();
    let cfg = &session.config;

    let cameras = session.state.per_cam_intrinsics.clone().unwrap();
    let sensors = session.state.per_cam_sensors.clone().unwrap();
    let cam_se3_rig = session.state.rig_ba_cam_se3_rig.clone().unwrap();
    let cam_to_rig: Vec<Iso3> = cam_se3_rig.iter().map(|t| t.inverse()).collect();
    let handeye = session.state.initial_handeye.unwrap();
    let mode_target_pose = session.state.initial_mode_target_pose.unwrap();

    let initial = HandEyeScheimpflugParams {
        cameras,
        sensors,
        cam_to_rig,
        handeye,
        target_poses: vec![mode_target_pose],
    };

    let fix_extrinsics: Vec<bool> = if cfg.handeye_ba.refine_cam_se3_rig_in_handeye_ba {
        (0..input.num_cameras)
            .map(|i| i == cfg.rig.reference_camera_idx)
            .collect()
    } else {
        vec![true; input.num_cameras]
    };
    let default_scheimpflug_fix = if cfg.handeye_ba.refine_scheimpflug_in_handeye_ba {
        ScheimpflugFixMask::default()
    } else {
        ScheimpflugFixMask {
            tilt_x: true,
            tilt_y: true,
        }
    };

    let solve_opts = HandEyeScheimpflugSolveOptions {
        robust_loss: cfg.solver.robust_loss,
        default_fix: CameraFixMask::all_fixed(),
        camera_overrides: Vec::new(),
        default_scheimpflug_fix,
        scheimpflug_overrides: Vec::new(),
        fix_extrinsics,
        fix_handeye: false,
        fix_target_poses: Vec::new(),
        relax_target_poses: false,
        refine_robot_poses: cfg.handeye_ba.refine_robot_poses,
        robot_rot_sigma: cfg.handeye_ba.robot_rot_sigma,
        robot_trans_sigma: cfg.handeye_ba.robot_trans_sigma,
    };
    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(cfg.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(cfg.solver.verbosity),
        ..Default::default()
    };

    let dataset = HandEyeScheimpflugDataset::new(
        input.views.clone(),
        input.num_cameras,
        cfg.handeye_init.handeye_mode,
    )?;

    let result = match optimize_handeye_scheimpflug(dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    session.state.final_cost = Some(result.report.final_cost);
    session.state.final_reproj_error = Some(result.mean_reproj_error);
    session.set_output(result.clone());
    session.log_success_with_notes(
        "handeye_optimize",
        format!(
            "final_cost={:.2e}, mean_reproj={:.3}px",
            result.report.final_cost, result.mean_reproj_error
        ),
    );
    Ok(())
}

/// Run the full Scheimpflug rig hand-eye calibration pipeline.
///
/// # Errors
///
/// Returns [`Error`] from any constituent step.
pub fn run_calibration(
    session: &mut CalibrationSession<RigScheimpflugHandeyeProblem>,
) -> Result<(), Error> {
    step_intrinsics_init_all(session, None)?;
    step_intrinsics_optimize_all(session, None)?;
    step_rig_init(session)?;
    step_rig_optimize(session, None)?;
    step_handeye_init(session, None)?;
    step_handeye_optimize(session, None)?;
    Ok(())
}
