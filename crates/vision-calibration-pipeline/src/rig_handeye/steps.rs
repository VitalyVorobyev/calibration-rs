//! Step functions for multi-camera rig hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigHandeyeProblem>` to perform calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraFixMask, FxFyCxCySkew, Iso3, NoMeta, PlanarDataset, Real, View,
    compute_rig_reprojection_stats_per_camera, make_pinhole_camera,
};
use vision_calibration_linear::estimate_extrinsics_from_cam_target_poses;
use vision_calibration_linear::prelude::*;
use vision_calibration_linear::{estimate_gripper_se3_target_dlt, estimate_handeye_dlt};
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeDataset, HandEyeMode, HandEyeParams, HandEyeSolveOptions,
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigExtrinsicsParams,
    RigExtrinsicsSolveOptions, optimize_handeye, optimize_planar_intrinsics,
    optimize_rig_extrinsics,
};

use crate::session::CalibrationSession;

use super::problem::{RigHandeyeInput, RigHandeyeProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

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

/// Options for hand-eye initialization.
#[derive(Debug, Clone, Default)]
pub struct HandeyeInitOptions {
    /// Override minimum motion angle (degrees).
    pub min_motion_angle_deg: Option<f64>,
}

/// Manual seeds for the **per-camera intrinsics stage** of rig hand-eye
/// calibration. See `rig_extrinsics::RigIntrinsicsManualInit` for semantics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigHandeyeIntrinsicsManualInit {
    pub per_cam_intrinsics: Option<Vec<FxFyCxCySkew<Real>>>,
    pub per_cam_distortion: Option<Vec<BrownConrady5<Real>>>,
}

/// Manual seeds for the **rig extrinsics stage** of rig hand-eye calibration.
///
/// `cam_se3_rig` and `rig_se3_target` are coupled per ADR 0011 — both must be
/// `Some` or both `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigHandeyeRigManualInit {
    pub cam_se3_rig: Option<Vec<Iso3>>,
    pub rig_se3_target: Option<Vec<Iso3>>,
}

/// Manual seeds for the **hand-eye stage** of rig hand-eye calibration.
///
/// Mode-aware: the meaning of `handeye` and `mode_target_pose` depends on
/// `session.config.handeye_init.handeye_mode`:
/// - `EyeInHand`: `handeye = gripper_se3_rig` (T_G_R), `mode_target_pose =
///   base_se3_target` (T_B_T).
/// - `EyeToHand`: `handeye = rig_se3_base` (T_R_B), `mode_target_pose =
///   gripper_se3_target` (T_G_T).
///
/// The state stores both as mode-agnostic `initial_handeye` and
/// `initial_mode_target_pose`, so this struct mirrors that shape directly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigHandeyeHandeyeManualInit {
    /// Mode-dependent hand-eye transform.
    pub handeye: Option<Iso3>,
    /// Mode-dependent target pose.
    pub mode_target_pose: Option<Iso3>,
}

/// Options for hand-eye optimization.
#[derive(Debug, Clone, Default)]
pub struct HandeyeOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Extract views for a single camera from the rig dataset.
fn extract_camera_views(input: &RigHandeyeInput, cam_idx: usize) -> Vec<Option<View<NoMeta>>> {
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

/// Create a PlanarDataset from non-None views.
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

/// Estimate initial target pose from camera intrinsics and view observations.
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

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics for all cameras.
///
/// Runs Zhang's method with iterative distortion estimation on each camera independently.
///
/// # Errors
///
/// - Input not set
/// - Any camera has fewer than 3 views with observations
pub fn step_set_intrinsics_init_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: RigHandeyeIntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;
    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    if let Some(s) = &manual.per_cam_intrinsics
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_intrinsics length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }
    if let Some(s) = &manual.per_cam_distortion
        && s.len() != num_cameras
    {
        return Err(Error::invalid_input(format!(
            "per_cam_distortion length ({}) != num_cameras ({})",
            s.len(),
            num_cameras
        )));
    }

    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(config.intrinsics.init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: config.intrinsics.fix_k3,
            fix_tangential: config.intrinsics.fix_tangential,
            iters: 8,
        },
        zero_skew: config.intrinsics.zero_skew,
    };

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();
    if manual.per_cam_intrinsics.is_some() {
        manual_fields.push("per_cam_intrinsics");
    } else {
        auto_fields.push("per_cam_intrinsics");
    }
    if manual.per_cam_distortion.is_some() {
        manual_fields.push("per_cam_distortion");
    } else {
        auto_fields.push("per_cam_distortion");
    }

    let mut per_cam_intrinsics = Vec::with_capacity(num_cameras);
    let mut per_cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];

    #[allow(clippy::needless_range_loop)]
    for cam_idx in 0..num_cameras {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let camera = if let Some(seeds) = manual.per_cam_intrinsics.as_ref() {
            let k = seeds[cam_idx];
            let dist = manual
                .per_cam_distortion
                .as_ref()
                .map(|d| d[cam_idx])
                .unwrap_or_default();
            make_pinhole_camera(k, dist)
        } else {
            let bootstrap =
                estimate_intrinsics_iterative(&planar_dataset, init_opts).map_err(|e| {
                    Error::numerical(format!(
                        "intrinsics estimation failed for camera {cam_idx}: {e}"
                    ))
                })?;
            let dist = manual
                .per_cam_distortion
                .as_ref()
                .map(|d| d[cam_idx])
                .unwrap_or(bootstrap.dist);
            make_pinhole_camera(bootstrap.k, dist)
        };

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

        per_cam_intrinsics.push(camera);
    }

    session.state.per_cam_intrinsics = Some(per_cam_intrinsics);
    session.state.per_cam_target_poses = Some(per_cam_target_poses);

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "intrinsics_init_all",
        format!("initialized {} cameras {}", num_cameras, source),
    );

    Ok(())
}

pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    step_set_intrinsics_init_all(session, RigHandeyeIntrinsicsManualInit::default(), opts)
}

fn format_init_source(manual: &[&str], auto: &[&str]) -> String {
    match (manual.is_empty(), auto.is_empty()) {
        (false, false) => format!("(manual: {}; auto: {})", manual.join(", "), auto.join(", ")),
        (false, true) => format!("(manual: {})", manual.join(", ")),
        (true, false) => format!("(auto: {})", auto.join(", ")),
        (true, true) => "(empty)".to_string(),
    }
}

/// Optimize intrinsics for all cameras.
///
/// Refines each camera's intrinsics independently using non-linear optimization.
///
/// Requires [`step_intrinsics_init_all`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Initialization not run
/// - Optimization fails for any camera
pub fn step_intrinsics_optimize_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
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
    let mut per_cam_target_poses = session
        .state
        .per_cam_target_poses
        .clone()
        .ok_or_else(|| Error::not_available("per-camera target poses"))?;

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);

    for cam_idx in 0..input.num_cameras {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        // Get initial poses for this camera
        let initial_poses: Vec<Iso3> = valid_indices
            .iter()
            .map(|&global_idx| {
                per_cam_target_poses[global_idx][cam_idx]
                    .ok_or_else(|| Error::not_available("initial pose for camera view"))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        // Build params
        let initial_params =
            PlanarIntrinsicsParams::new(per_cam_intrinsics[cam_idx].clone(), initial_poses)
                .map_err(|e| {
                    Error::numerical(format!("failed to build params for camera {cam_idx}: {e}"))
                })?;

        // Optimize
        let solve_opts = PlanarIntrinsicsSolveOptions {
            robust_loss: config.solver.robust_loss,
            fix_intrinsics: Default::default(),
            fix_distortion: Default::default(),
            fix_poses: Vec::new(),
        };

        let backend_opts = BackendSolveOptions {
            max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
            verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
            ..Default::default()
        };

        let result =
            optimize_planar_intrinsics(&planar_dataset, &initial_params, solve_opts, backend_opts)
                .map_err(|e| {
                    Error::numerical(format!("optimization failed for camera {cam_idx}: {e}"))
                })?;

        // Update target poses for this camera
        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            per_cam_target_poses[global_idx][cam_idx] = Some(result.params.poses()[local_idx]);
        }

        optimized_cameras.push(result.params.camera.clone());
        per_cam_reproj_errors.push(result.mean_reproj_error);
    }

    // Update state
    session.state.per_cam_intrinsics = Some(optimized_cameras);
    session.state.per_cam_target_poses = Some(per_cam_target_poses);
    session.state.per_cam_reproj_errors = Some(per_cam_reproj_errors.clone());

    let avg_error: f64 =
        per_cam_reproj_errors.iter().sum::<f64>() / per_cam_reproj_errors.len() as f64;
    session.log_success_with_notes(
        "intrinsics_optimize_all",
        format!("avg_reproj_err={:.3}px", avg_error),
    );

    Ok(())
}

/// Initialize rig extrinsics from per-camera target poses.
///
/// Uses linear initialization to estimate camera-to-rig transforms.
///
/// Requires [`step_intrinsics_optimize_all`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Per-camera intrinsics not computed
/// - Insufficient overlapping views between cameras
pub fn step_set_rig_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: RigHandeyeRigManualInit,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_optimize_all first)",
        ));
    }

    let num_views = input.num_views();
    let num_cameras = input.num_cameras;
    let reference_camera_idx = session.config.rig.reference_camera_idx;

    match (&manual.cam_se3_rig, &manual.rig_se3_target) {
        (Some(_), None) | (None, Some(_)) => {
            let msg = "RigHandeyeRigManualInit: cam_se3_rig and rig_se3_target must both be Some \
                       or both None (geometrically coupled per ADR 0011)";
            session.log_failure("rig_init", msg);
            return Err(Error::invalid_input(msg));
        }
        _ => {}
    }

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    let (cam_se3_rig, rig_se3_target) = match (manual.cam_se3_rig, manual.rig_se3_target) {
        (Some(cam_se3_rig), Some(rig_se3_target)) => {
            if cam_se3_rig.len() != num_cameras {
                return Err(Error::invalid_input(format!(
                    "cam_se3_rig length ({}) != num_cameras ({})",
                    cam_se3_rig.len(),
                    num_cameras
                )));
            }
            if rig_se3_target.len() != num_views {
                return Err(Error::invalid_input(format!(
                    "rig_se3_target length ({}) != num_views ({})",
                    rig_se3_target.len(),
                    num_views
                )));
            }
            manual_fields.push("cam_se3_rig");
            manual_fields.push("rig_se3_target");
            (cam_se3_rig, rig_se3_target)
        }
        _ => {
            auto_fields.push("cam_se3_rig");
            auto_fields.push("rig_se3_target");
            let per_cam_target_poses = session
                .state
                .per_cam_target_poses
                .clone()
                .ok_or_else(|| Error::not_available("per-camera target poses"))?;
            let extrinsic_result = estimate_extrinsics_from_cam_target_poses(
                &per_cam_target_poses,
                reference_camera_idx,
            )
            .map_err(|e| Error::numerical(format!("rig extrinsics initialization failed: {e}")))?;
            let cam_se3_rig: Vec<Iso3> = extrinsic_result
                .cam_to_rig
                .iter()
                .map(|t| t.inverse())
                .collect();
            (cam_se3_rig, extrinsic_result.rig_from_target)
        }
    };

    session.state.initial_cam_se3_rig = Some(cam_se3_rig);
    session.state.initial_rig_se3_target = Some(rig_se3_target);

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "rig_init",
        format!(
            "ref_cam={}, {} views {}",
            reference_camera_idx, num_views, source
        ),
    );

    Ok(())
}

pub fn step_rig_init(session: &mut CalibrationSession<RigHandeyeProblem>) -> Result<(), Error> {
    step_set_rig_init(session, RigHandeyeRigManualInit::default())
}

/// Optimize rig extrinsics using bundle adjustment.
///
/// Jointly optimizes:
/// - Per-camera extrinsics (camera-to-rig)
/// - Per-view rig poses (rig-to-target)
/// - Optionally: per-camera intrinsics (if `refine_intrinsics_in_rig_ba` is true)
///
/// Requires [`step_rig_init`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Rig initialization not run
/// - Optimization fails
pub fn step_rig_optimize(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<RigOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_rig_init() {
        return Err(Error::not_available(
            "rig initialization (call step_rig_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let cameras = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
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

    // Build initial params
    let initial = RigExtrinsicsParams {
        cameras,
        cam_to_rig,
        rig_from_target,
    };

    // Configure solve options
    let fix_intrinsics = if config.rig.refine_intrinsics_in_rig_ba {
        CameraFixMask::default()
    } else {
        CameraFixMask::all_fixed()
    };

    // Reference camera has fixed extrinsics (identity)
    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == config.rig.reference_camera_idx)
        .collect();

    let fix_rig_poses = if config.rig.fix_first_rig_pose {
        vec![0]
    } else {
        Vec::new()
    };

    let solve_opts = RigExtrinsicsSolveOptions {
        robust_loss: config.solver.robust_loss,
        default_fix: fix_intrinsics,
        camera_overrides: Vec::new(),
        fix_extrinsics,
        fix_rig_poses,
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..Default::default()
    };

    // Convert input to NoMeta format for rig extrinsics optimization
    let rig_input_no_meta: vision_calibration_core::RigDataset<NoMeta> =
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

    // Run optimization
    let result = match optimize_rig_extrinsics(rig_input_no_meta, initial, solve_opts, backend_opts)
    {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("rig_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    // Update state with refined rig extrinsics
    let cam_se3_rig: Vec<Iso3> = result
        .params
        .cam_to_rig
        .iter()
        .map(|t| t.inverse())
        .collect();
    let per_cam_stats = compute_rig_reprojection_stats_per_camera(
        &result.params.cameras,
        input,
        &cam_se3_rig,
        &result.params.rig_from_target,
    )
    .map_err(|e| {
        Error::numerical(format!(
            "failed to compute per-camera rig BA reprojection error: {e}"
        ))
    })?;
    let total_count: usize = per_cam_stats.iter().map(|s| s.count).sum();
    let total_error: f64 = per_cam_stats
        .iter()
        .map(|s| s.mean * (s.count as f64))
        .sum();
    let mean_reproj_error = total_error / (total_count as f64);
    session.state.rig_ba_cam_se3_rig = Some(cam_se3_rig);
    session.state.rig_ba_rig_se3_target = Some(result.params.rig_from_target.clone());
    session.state.rig_ba_reproj_error = Some(mean_reproj_error);
    session.state.rig_ba_per_cam_reproj_errors =
        Some(per_cam_stats.iter().map(|s| s.mean).collect());
    // Also update cameras in case intrinsics were refined
    session.state.per_cam_intrinsics = Some(result.params.cameras);

    session.log_success_with_notes(
        "rig_optimize",
        format!(
            "final_cost={:.2e}, mean_reproj_err={:.3}px",
            result.report.final_cost, mean_reproj_error
        ),
    );

    Ok(())
}

/// Initialize hand-eye transform from robot poses and rig poses.
///
/// Uses Tsai-Lenz linear initialization.
///
/// Requires [`step_rig_optimize`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Rig optimization not run
/// - Linear hand-eye estimation fails
pub fn step_set_handeye_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: RigHandeyeHandeyeManualInit,
    opts: Option<HandeyeInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_rig_optimized() {
        return Err(Error::not_available(
            "rig optimization results (call step_rig_optimize first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let min_angle = opts
        .min_motion_angle_deg
        .unwrap_or(session.config.handeye_init.min_motion_angle_deg);
    let handeye_mode = session.config.handeye_init.handeye_mode;

    let robot_poses: Vec<Iso3> = input
        .views
        .iter()
        .map(|v| v.meta.base_se3_gripper)
        .collect();
    let rig_se3_target = session
        .state
        .rig_ba_rig_se3_target
        .clone()
        .ok_or_else(|| Error::not_available("rig_se3_target from rig BA"))?;

    let mut manual_fields: Vec<&'static str> = Vec::new();
    let mut auto_fields: Vec<&'static str> = Vec::new();

    let handeye = match manual.handeye {
        Some(t) => {
            manual_fields.push("handeye");
            t
        }
        None => {
            auto_fields.push("handeye");
            match handeye_mode {
                HandEyeMode::EyeInHand => {
                    let target_se3_rig: Vec<Iso3> =
                        rig_se3_target.iter().map(|t| t.inverse()).collect();
                    estimate_handeye_dlt(&robot_poses, &target_se3_rig, min_angle)
                        .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))
                        .map_err(|e| {
                            Error::numerical(format!("linear hand-eye estimation failed: {e}"))
                        })?
                }
                HandEyeMode::EyeToHand => {
                    let gripper_se3_target =
                        estimate_gripper_se3_target_dlt(&robot_poses, &rig_se3_target, min_angle)
                            .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))
                            .map_err(|e| {
                                Error::numerical(format!(
                                    "linear gripper->target estimation failed: {e}"
                                ))
                            })?;
                    rig_se3_target[0] * (robot_poses[0] * gripper_se3_target).inverse()
                }
            }
        }
    };

    let mode_target_pose = match manual.mode_target_pose {
        Some(t) => {
            manual_fields.push("mode_target_pose");
            t
        }
        None => {
            auto_fields.push("mode_target_pose");
            match handeye_mode {
                HandEyeMode::EyeInHand => robot_poses[0] * handeye * rig_se3_target[0],
                HandEyeMode::EyeToHand => {
                    // handeye = T_R_B. Chain: T_R_T = T_R_B * T_B_G * T_G_T.
                    // T_G_T = (T_B_G)^-1 * (T_R_B)^-1 * T_R_T = robot_poses[0]^-1 * handeye^-1 * T_R_T.
                    robot_poses[0].inverse() * handeye.inverse() * rig_se3_target[0]
                }
            }
        }
    };

    session.state.initial_handeye = Some(handeye);
    session.state.initial_mode_target_pose = Some(mode_target_pose);

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "handeye_init",
        format!(
            "translation_norm={:.4}m {}",
            handeye.translation.vector.norm(),
            source
        ),
    );

    Ok(())
}

pub fn step_handeye_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<(), Error> {
    step_set_handeye_init(session, RigHandeyeHandeyeManualInit::default(), opts)
}

/// Optimize hand-eye calibration using bundle adjustment.
///
/// This step jointly optimizes:
/// - Per-camera intrinsics
/// - Per-camera extrinsics (cam_se3_rig) - optionally
/// - Hand-eye transform (mode-dependent)
/// - Fixed target pose (mode-dependent)
/// - Optionally: per-view robot pose corrections
///
/// Requires [`step_handeye_init`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Hand-eye initialization not run
/// - Optimization fails
pub fn step_handeye_optimize(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<HandeyeOptimizeOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?.clone();

    if !session.state.has_handeye_init() {
        return Err(Error::not_available(
            "hand-eye initialization (call step_handeye_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get all initial estimates
    let cameras = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    let cam_se3_rig = session
        .state
        .rig_ba_cam_se3_rig
        .clone()
        .ok_or_else(|| Error::not_available("cam_se3_rig from rig BA"))?;
    let cam_to_rig: Vec<Iso3> = cam_se3_rig.iter().map(|t| t.inverse()).collect();
    let handeye = session
        .state
        .initial_handeye
        .ok_or_else(|| Error::not_available("initial handeye"))?;
    let mode_target_pose = session
        .state
        .initial_mode_target_pose
        .ok_or_else(|| Error::not_available("initial mode target pose"))?;

    // Build initial params for hand-eye optimization
    let initial = HandEyeParams {
        cameras,
        cam_to_rig,
        handeye,
        target_poses: vec![mode_target_pose], // Single fixed target (mode-dependent semantics)
    };

    // Configure solve options
    let fix_intrinsics = CameraFixMask::all_fixed(); // Don't refine intrinsics in final BA

    // Fix cam_se3_rig unless explicitly enabled
    let fix_extrinsics: Vec<bool> = if config.handeye_ba.refine_cam_se3_rig_in_handeye_ba {
        // Only fix reference camera
        (0..input.num_cameras)
            .map(|i| i == config.rig.reference_camera_idx)
            .collect()
    } else {
        // Fix all extrinsics
        vec![true; input.num_cameras]
    };

    let solve_opts = HandEyeSolveOptions {
        robust_loss: config.solver.robust_loss,
        default_fix: fix_intrinsics,
        camera_overrides: Vec::new(),
        fix_extrinsics,
        fix_handeye: false,
        fix_target_poses: Vec::new(),
        relax_target_poses: false, // Single fixed target
        refine_robot_poses: config.handeye_ba.refine_robot_poses,
        robot_rot_sigma: config.handeye_ba.robot_rot_sigma,
        robot_trans_sigma: config.handeye_ba.robot_trans_sigma,
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..Default::default()
    };

    // Convert input to HandEyeDataset
    let handeye_dataset = HandEyeDataset::new(
        input.views.clone(),
        input.num_cameras,
        config.handeye_init.handeye_mode,
    )?;

    // Run optimization
    let result = match optimize_handeye(handeye_dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    // Update state metrics
    session.state.final_cost = Some(result.report.final_cost);
    session.state.final_reproj_error = Some(result.report.final_cost.sqrt());

    // Set output
    session.set_output(result.clone());

    session.log_success_with_notes(
        "handeye_optimize",
        format!("final_cost={:.2e}", result.report.final_cost),
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Function
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full rig hand-eye calibration pipeline.
///
/// Runs: intrinsics_init_all → intrinsics_optimize_all → rig_init → rig_optimize
///       → handeye_init → handeye_optimize.
///
/// # Errors
///
/// Any error from the constituent steps.
pub fn run_calibration(session: &mut CalibrationSession<RigHandeyeProblem>) -> Result<(), Error> {
    step_intrinsics_init_all(session, None)?;
    step_intrinsics_optimize_all(session, None)?;
    step_rig_init(session)?;
    step_rig_optimize(session, None)?;
    step_handeye_init(session, None)?;
    step_handeye_optimize(session, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCySkew, PinholeCamera, Pt2, Pt3, RigDataset,
        RigView, RigViewObs, make_pinhole_camera,
    };
    use vision_calibration_optim::RobotPoseMeta;

    fn make_test_camera(offset: f64) -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0 + offset,
                fy: 780.0 + offset,
                cx: 640.0,
                cy: 360.0,
                skew: 0.0,
            },
            BrownConrady5::default(),
        )
    }

    fn make_iso(angles: (f64, f64, f64), t: (f64, f64, f64)) -> Iso3 {
        let rot = Rotation3::from_euler_angles(angles.0, angles.1, angles.2);
        let tr = Translation3::new(t.0, t.1, t.2);
        Iso3::from_parts(tr, rot.into())
    }

    fn make_test_input() -> RigHandeyeInput {
        // Ground truth cameras and rig
        let cam0 = make_test_camera(0.0);
        let cam1 = make_test_camera(10.0);
        let cam0_se3_rig = Iso3::identity();
        let cam1_se3_rig = make_iso((0.0, 0.0, 0.1), (0.2, 0.0, 0.0));

        // Ground truth hand-eye and target
        let handeye_gt = make_iso((0.05, -0.03, 0.02), (0.03, -0.02, 0.08));
        let target_in_base_gt = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.2));

        // Board points
        let board_pts: Vec<Pt3> = (0..6)
            .flat_map(|i| (0..5).map(move |j| Pt3::new(i as f64 * 0.05, j as f64 * 0.05, 0.0)))
            .collect();

        // Robot poses
        let robot_poses = [
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            make_iso((0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
            make_iso((0.0, 0.1, 0.0), (0.0, 0.1, 0.0)),
            make_iso((0.05, 0.05, 0.0), (-0.1, 0.0, 0.0)),
        ];

        let views: Vec<RigView<RobotPoseMeta>> = robot_poses
            .iter()
            .map(|robot_pose| {
                // rig_se3_target = (robot_pose * handeye)^-1 * target_in_base
                let rig_se3_target = (robot_pose * handeye_gt).inverse() * target_in_base_gt;

                // cam_se3_target = cam_se3_rig * rig_se3_target
                let cam0_se3_target = cam0_se3_rig * rig_se3_target;
                let cam1_se3_target = cam1_se3_rig * rig_se3_target;

                let project_points = |cam: &PinholeCamera, pose: &Iso3| -> Vec<Pt2> {
                    board_pts
                        .iter()
                        .map(|p| {
                            let p_cam = pose.transform_point(p);
                            cam.project_point_c(&p_cam.coords).unwrap()
                        })
                        .collect()
                };

                let pts0 = project_points(&cam0, &cam0_se3_target);
                let pts1 = project_points(&cam1, &cam1_se3_target);

                RigView {
                    meta: RobotPoseMeta {
                        base_se3_gripper: *robot_pose,
                    },
                    obs: RigViewObs {
                        cameras: vec![
                            Some(CorrespondenceView::new(board_pts.clone(), pts0).unwrap()),
                            Some(CorrespondenceView::new(board_pts.clone(), pts1).unwrap()),
                        ],
                    },
                }
            })
            .collect();

        RigDataset::new(views, 2).unwrap()
    }

    #[test]
    fn step_intrinsics_init_all_computes_estimate() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();

        step_intrinsics_init_all(&mut session, None).unwrap();

        assert!(session.state.has_per_cam_intrinsics());
        let cameras = session.state.per_cam_intrinsics.as_ref().unwrap();
        assert_eq!(cameras.len(), 2);
    }

    #[test]
    fn step_rig_init_requires_intrinsics() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();

        let result = step_rig_init(&mut session);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("intrinsics"));
    }

    #[test]
    fn step_rig_optimize_requires_init() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();
        step_intrinsics_optimize_all(&mut session, None).unwrap();

        let result = step_rig_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rig"));
    }

    #[test]
    fn step_handeye_init_requires_rig_optimize() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();
        step_intrinsics_optimize_all(&mut session, None).unwrap();
        step_rig_init(&mut session).unwrap();

        let result = step_handeye_init(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rig optimization"));
    }

    #[test]
    fn step_handeye_optimize_requires_init() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();
        step_intrinsics_optimize_all(&mut session, None).unwrap();
        step_rig_init(&mut session).unwrap();
        step_rig_optimize(&mut session, None).unwrap();

        let result = step_handeye_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hand-eye"));
    }

    #[test]
    fn set_input_clears_state() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();

        assert!(session.state.has_per_cam_intrinsics());

        session.set_input(make_test_input()).unwrap();
        assert!(!session.state.has_per_cam_intrinsics());
    }

    #[test]
    fn set_config_keeps_output() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();

        assert!(session.has_output());

        session
            .set_config(super::super::problem::RigHandeyeConfig {
                solver: super::super::problem::RigHandeyeSolverConfig {
                    max_iters: 100,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();

        assert!(session.has_output());
    }

    #[test]
    fn step_set_handeye_init_eye_to_hand_recovers_target_on_gripper() {
        // Regression test for Codex P1 on PR #32: in EyeToHand mode the
        // auto-derive of `mode_target_pose` from a manual `handeye` seed must
        // use `handeye.inverse()`, since `handeye = T_R_B` and the chain is
        // T_R_T = T_R_B * T_B_G * T_G_T, so T_G_T = T_B_G^-1 * T_R_B^-1 * T_R_T.
        let input = make_test_input();
        let robot_poses: Vec<Iso3> = input
            .views
            .iter()
            .map(|v| v.meta.base_se3_gripper)
            .collect();

        // Ground-truth EyeToHand: rig fixed in robot base, target on gripper.
        let t_r_b = make_iso((0.4, -0.2, 0.1), (0.5, -0.3, 0.2));
        let t_g_t = make_iso((0.15, 0.0, -0.1), (0.05, 0.04, 0.03));
        let rig_se3_target: Vec<Iso3> = robot_poses.iter().map(|tbg| t_r_b * tbg * t_g_t).collect();

        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(input).unwrap();
        session
            .set_config(super::super::problem::RigHandeyeConfig {
                handeye_init: super::super::problem::RigHandeyeInitConfig {
                    handeye_mode: HandEyeMode::EyeToHand,
                    min_motion_angle_deg: 5.0,
                },
                ..Default::default()
            })
            .unwrap();

        // Simulate post-rig-BA state.
        session.state.rig_ba_cam_se3_rig = Some(vec![
            Iso3::identity();
            session.require_input().unwrap().num_cameras
        ]);
        session.state.rig_ba_rig_se3_target = Some(rig_se3_target);

        // Pin handeye to ground truth so the test isolates the
        // mode_target_pose auto-derive arithmetic from DLT precision.
        let manual = RigHandeyeHandeyeManualInit {
            handeye: Some(t_r_b),
            mode_target_pose: None,
        };
        step_set_handeye_init(&mut session, manual, None).unwrap();

        let recovered = session.state.initial_mode_target_pose.unwrap();
        let dt = (recovered.translation.vector - t_g_t.translation.vector).norm();
        let dq = recovered
            .rotation
            .rotation_to(&t_g_t.rotation)
            .angle()
            .abs();
        assert!(dt < 1e-9, "T_G_T translation mismatch: |Δt|={dt}");
        assert!(dq < 1e-9, "T_G_T rotation mismatch: angle={dq}");
    }

    #[test]
    fn json_roundtrip() {
        let mut session =
            CalibrationSession::<RigHandeyeProblem>::with_description("Test rig hand-eye");
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<RigHandeyeProblem>::from_json(&json).unwrap();

        assert_eq!(
            restored.metadata.description,
            Some("Test rig hand-eye".to_string())
        );
        assert!(restored.has_input());
        assert!(restored.has_output());
        assert_eq!(restored.exports.len(), 1);
    }
}
