//! Step functions for multi-camera rig hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigHandeyeProblem>` to perform calibration.

use crate::Error;
use nalgebra::{Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, Camera, CameraFixMask, CorrespondenceView, DistortionFixMask, DistortionModel,
    FxFyCxCySkew, IntrinsicsFixMask, IntrinsicsModel, Iso3, NoMeta, Pinhole, PinholeCamera,
    PlanarDataset, Pt2, RigDataset, RigView, RigViewObs, ScheimpflugParams, SensorModel, View,
    compute_rig_reprojection_stats_per_camera, make_pinhole_camera,
};
use vision_calibration_linear::extrinsics::estimate_extrinsics_from_cam_target_poses;
use vision_calibration_linear::handeye::{estimate_gripper_se3_target_dlt, estimate_handeye_dlt};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeDataset, HandEyeMode, HandEyeParams, HandEyeScheimpflugDataset,
    HandEyeScheimpflugParams, HandEyeScheimpflugSolveOptions, HandEyeSolveOptions,
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigExtrinsicsParams,
    RigExtrinsicsScheimpflugParams, RigExtrinsicsScheimpflugSolveOptions,
    RigExtrinsicsSolveOptions, ScheimpflugBounds, ScheimpflugFixMask,
    ScheimpflugIntrinsicsEstimate, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsSolveOptions,
    ScheimpflugStagedInitOptions, SolveReport, optimize_handeye, optimize_handeye_scheimpflug,
    optimize_planar_intrinsics, optimize_rig_extrinsics, optimize_rig_extrinsics_scheimpflug,
    optimize_scheimpflug_intrinsics, optimize_scheimpflug_intrinsics_staged,
};

use crate::rig_family::{
    RigIntrinsicsSeeds, SensorFlavour, bootstrap_rig_intrinsics, format_init_source,
    guard_percam_reproj_errors, views_to_planar_dataset,
};
use crate::session::CalibrationSession;

use super::problem::{
    RigHandeyeInput, RigHandeyeIntrinsicsManualInit, RigHandeyeOutput, RigHandeyeProblem,
    SensorMode,
};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::common::{
    HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions, IntrinsicsOptimizeOptions,
};

/// Options for rig BA optimization.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct RigOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Manual seeds for the **rig extrinsics stage** of rig hand-eye calibration.
///
/// `cam_se3_rig` and `rig_se3_target` are coupled per ADR 0011 — both must be
/// `Some` or both `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigHandeyeRigManualInit {
    /// Per-camera `T_C_R` — camera-from-rig. `None` runs the linear extrinsics fit.
    pub cam_se3_rig: Option<Vec<Iso3>>,
    /// Per-view `T_R_T` — rig-from-target. Must be paired with `cam_se3_rig`.
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
#[non_exhaustive]
pub struct RigHandeyeHandeyeManualInit {
    /// Mode-dependent hand-eye transform.
    pub handeye: Option<Iso3>,
    /// Mode-dependent target pose.
    pub mode_target_pose: Option<Iso3>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Extract views for a single camera from the rig dataset.
///
/// Input-type-specific; the rest of the per-camera bootstrap chain is shared
/// via [`crate::rig_family`].
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

#[derive(Debug, Clone)]
struct ScheimpflugCameraWork {
    cam_idx: usize,
    dataset: PlanarDataset,
    valid_indices: Vec<usize>,
    initial: ScheimpflugIntrinsicsParams,
}

#[derive(Debug, Clone, Copy)]
struct NominalScheimpflugSeed {
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
}

#[derive(Debug, Clone)]
struct ScoredScheimpflugSeed {
    score: f64,
    params: ScheimpflugIntrinsicsParams,
}

#[derive(Debug, Clone)]
struct GoodRigContext {
    cam_to_rig: Vec<Iso3>,
    rig_from_target: Vec<Iso3>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

/// Typed return value of [`step_intrinsics_init_all`] /
/// [`step_intrinsics_init_all_with_seed`] for rig hand-eye calibration.
///
/// Structurally similar to [`crate::rig_extrinsics::RigIntrinsicsInitAllResult`]
/// but lives in this module so the two problem types can evolve independently.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RigHandeyeIntrinsicsInitAllResult {
    /// Per-camera initial pinhole intrinsics + distortion.
    pub per_cam_intrinsics: Vec<PinholeCamera>,
    /// Per-camera Scheimpflug sensor parameters; `None` for pinhole rigs.
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
    /// Per-camera target poses: `[view][cam] -> Option<Iso3>` (`camera_se3_target`).
    /// Inner `None` marks views where that camera did not observe the target.
    pub per_cam_target_poses: Vec<Vec<Option<Iso3>>>,
}

/// Typed return value of [`step_intrinsics_optimize_all`] for rig hand-eye
/// calibration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RigHandeyeIntrinsicsOptimizeAllResult {
    /// Per-camera refined pinhole intrinsics + distortion.
    pub per_cam_intrinsics: Vec<PinholeCamera>,
    /// Per-camera refined Scheimpflug sensor parameters; `None` for pinhole rigs.
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
    /// Per-camera mean reprojection error in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
}

/// Typed return value of [`step_rig_init`] / [`step_rig_init_with_seed`] for rig
/// hand-eye calibration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RigHandeyeRigInitResult {
    /// Initial per-camera `T_C_R` — camera-from-rig (reference camera is identity).
    pub initial_cam_se3_rig: Vec<Iso3>,
    /// Initial per-view `T_R_T` — rig-from-target.
    pub initial_rig_se3_target: Vec<Iso3>,
}

/// Typed return value of [`step_rig_optimize`] for rig hand-eye calibration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RigHandeyeRigOptimizeResult {
    /// Mean reprojection error in pixels across the whole rig after rig BA.
    pub mean_reproj_error: f64,
    /// Per-camera mean reprojection error in pixels after rig BA.
    pub per_cam_reproj_errors: Vec<f64>,
    /// Optimized per-camera `T_C_R` — camera-from-rig.
    pub cam_se3_rig: Vec<Iso3>,
    /// Optimized per-view `T_R_T` — rig-from-target.
    pub rig_se3_target: Vec<Iso3>,
}

/// Typed return value of [`step_handeye_init`] / [`step_handeye_init_with_seed`].
///
/// The interpretation of both fields is mode-dependent — see
/// [`RigHandeyeHandeyeManualInit`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RigHandeyeHandeyeInitResult {
    /// Mode-dependent initial hand-eye transform.
    ///
    /// - `EyeInHand`: `gripper_se3_rig` (`T_G_R`).
    /// - `EyeToHand`: `rig_se3_base` (`T_R_B`).
    pub initial_handeye: Iso3,
    /// Mode-dependent fixed target pose. The state field models a genuine
    /// "may or may not be populated in this mode" situation; this result
    /// preserves the same shape.
    ///
    /// - `EyeInHand`: `base_se3_target` (`T_B_T`).
    /// - `EyeToHand`: `gripper_se3_target` (`T_G_T`).
    pub initial_mode_target_pose: Option<Iso3>,
}

/// Typed return value of [`step_handeye_optimize`] for rig hand-eye calibration.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RigHandeyeHandeyeOptimizeResult {
    /// Mean reprojection error in pixels after the final hand-eye BA.
    pub mean_reproj_error: f64,
    /// Final solver cost from the hand-eye BA.
    pub final_cost: f64,
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
pub fn step_intrinsics_init_all_with_seed(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: RigHandeyeIntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<RigHandeyeIntrinsicsInitAllResult, Error> {
    session.validate()?;
    let input = session.require_input()?;
    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(config.intrinsics.init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: config.intrinsics.fix_k3,
            fix_tangential: config.intrinsics.fix_tangential,
            iters: 8,
        },
        zero_skew: config.intrinsics.zero_skew,
    };

    let flavour = match &config.sensor {
        SensorMode::Pinhole => SensorFlavour::Pinhole,
        SensorMode::Scheimpflug {
            init_tilt_x,
            init_tilt_y,
            ..
        } => SensorFlavour::Scheimpflug {
            default_tilt_x: *init_tilt_x,
            default_tilt_y: *init_tilt_y,
        },
    };

    let seeds = RigIntrinsicsSeeds {
        per_cam_intrinsics: manual.per_cam_intrinsics,
        per_cam_distortion: manual.per_cam_distortion,
        per_cam_sensors: manual.per_cam_sensors,
    };

    let bootstrap = bootstrap_rig_intrinsics(
        num_cameras,
        num_views,
        |cam_idx| extract_camera_views(input, cam_idx),
        seeds,
        init_opts,
        flavour,
    )?;

    let per_cam_intrinsics = bootstrap.bundle.cameras;
    let per_cam_sensors = bootstrap.bundle.scheimpflug;
    let per_cam_target_poses = bootstrap.per_cam_target_poses;

    session.state.per_cam_intrinsics = Some(per_cam_intrinsics.clone());
    session.state.per_cam_sensors = per_cam_sensors.clone();
    session.state.per_cam_target_poses = Some(per_cam_target_poses.clone());
    session.state.per_cam_intrinsics_auto = bootstrap.manual_fields.is_empty();

    let source = format_init_source(&bootstrap.manual_fields, &bootstrap.auto_fields);
    session.log_success_with_notes(
        "intrinsics_init_all",
        format!("initialized {num_cameras} cameras {source}"),
    );

    Ok(RigHandeyeIntrinsicsInitAllResult {
        per_cam_intrinsics,
        per_cam_sensors,
        per_cam_target_poses,
    })
}

/// Initialize intrinsics for all cameras using full auto-init (Zhang's per camera).
///
/// Convenience wrapper around [`step_intrinsics_init_all_with_seed`] with default seeds.
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<RigHandeyeIntrinsicsInitAllResult, Error> {
    step_intrinsics_init_all_with_seed(session, RigHandeyeIntrinsicsManualInit::default(), opts)
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
) -> Result<RigHandeyeIntrinsicsOptimizeAllResult, Error> {
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

    let max_iters = opts.max_iters.unwrap_or(config.solver.max_iters);
    let verbosity = opts.verbosity.unwrap_or(config.solver.verbosity);

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);
    let mut optimized_sensors = match &config.sensor {
        SensorMode::Pinhole => None,
        SensorMode::Scheimpflug { .. } => Some(Vec::with_capacity(input.num_cameras)),
    };
    let mut scheimpflug_work_items = Vec::new();

    let per_cam_sensors_in = match &config.sensor {
        SensorMode::Pinhole => None,
        SensorMode::Scheimpflug { .. } => Some(
            session
                .state
                .per_cam_sensors
                .clone()
                .ok_or_else(|| Error::not_available("per-camera Scheimpflug sensors"))?,
        ),
    };

    for cam_idx in 0..input.num_cameras {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views).map_err(|e| {
            Error::numerical(format!("camera {cam_idx} has insufficient views: {e}"))
        })?;

        let initial_poses: Vec<Iso3> = valid_indices
            .iter()
            .map(|&global_idx| {
                per_cam_target_poses[global_idx][cam_idx]
                    .ok_or_else(|| Error::not_available("initial pose for camera view"))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        match &config.sensor {
            SensorMode::Pinhole => {
                let initial_params = PlanarIntrinsicsParams::from_pinhole(
                    per_cam_intrinsics[cam_idx].clone(),
                    initial_poses,
                )
                .map_err(|e| {
                    Error::numerical(format!("failed to build params for camera {cam_idx}: {e}"))
                })?;

                let solve_opts = PlanarIntrinsicsSolveOptions {
                    robust_loss: config.solver.robust_loss,
                    fix_intrinsics: Default::default(),
                    fix_distortion: Default::default(),
                    fix_poses: Vec::new(),
                };

                let backend_opts = BackendSolveOptions {
                    max_iters,
                    verbosity,
                    ..Default::default()
                };

                let result = optimize_planar_intrinsics(
                    &planar_dataset,
                    &initial_params,
                    solve_opts,
                    backend_opts,
                )
                .map_err(|e| {
                    Error::numerical(format!("optimization failed for camera {cam_idx}: {e}"))
                })?;

                for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
                    per_cam_target_poses[global_idx][cam_idx] =
                        Some(result.params.poses()[local_idx]);
                }

                optimized_cameras.push(result.params.pinhole_camera().map_err(|e| {
                    Error::numerical(format!(
                        "camera {cam_idx} result not pinhole-compatible: {e}"
                    ))
                })?);
                per_cam_reproj_errors.push(result.mean_reproj_error);
            }
            SensorMode::Scheimpflug {
                fix_scheimpflug_in_intrinsics,
                distortion_mask_in_percam_ba,
                ..
            } => {
                let cam = &per_cam_intrinsics[cam_idx];
                let sensor = per_cam_sensors_in.as_ref().unwrap()[cam_idx];
                let initial_params =
                    ScheimpflugIntrinsicsParams::new(cam.k, cam.dist, sensor, initial_poses)?;
                if session.state.per_cam_intrinsics_auto {
                    scheimpflug_work_items.push(ScheimpflugCameraWork {
                        cam_idx,
                        dataset: planar_dataset.clone(),
                        valid_indices: valid_indices.clone(),
                        initial: initial_params.clone(),
                    });
                }

                let solve_opts = ScheimpflugIntrinsicsSolveOptions {
                    robust_loss: config.solver.robust_loss,
                    fix_intrinsics: IntrinsicsFixMask::default(),
                    fix_distortion: *distortion_mask_in_percam_ba,
                    fix_scheimpflug: *fix_scheimpflug_in_intrinsics,
                    fix_poses: vec![0],
                    bounds: None,
                };

                let backend_opts = BackendSolveOptions {
                    max_iters,
                    verbosity,
                    ..Default::default()
                };

                let result = if session.state.per_cam_intrinsics_auto {
                    optimize_scheimpflug_intrinsics_auto_multistart(
                        cam_idx,
                        &planar_dataset,
                        &initial_params,
                        solve_opts,
                        backend_opts,
                        config.intrinsics.init_iterations,
                        config.intrinsics.fix_k3,
                        config.intrinsics.zero_skew,
                    )?
                } else {
                    optimize_scheimpflug_intrinsics_staged(
                        &planar_dataset,
                        &initial_params,
                        solve_opts,
                        &ScheimpflugStagedInitOptions::default(),
                        backend_opts,
                    )
                    .map_err(|e| {
                        Error::numerical(format!(
                            "Scheimpflug intrinsics optimization failed for camera {cam_idx}: {e}"
                        ))
                    })?
                };

                for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
                    per_cam_target_poses[global_idx][cam_idx] =
                        Some(result.params.camera_se3_target[local_idx]);
                }

                optimized_cameras.push(make_pinhole_camera(
                    result.params.intrinsics,
                    result.params.distortion,
                ));
                optimized_sensors
                    .as_mut()
                    .unwrap()
                    .push(result.params.sensor);
                per_cam_reproj_errors.push(result.mean_reproj_error);
            }
        }
    }

    if session.state.per_cam_intrinsics_auto
        && let SensorMode::Scheimpflug {
            fix_scheimpflug_in_intrinsics,
            distortion_mask_in_percam_ba,
            ..
        } = &config.sensor
    {
        let solve_opts = ScheimpflugIntrinsicsSolveOptions {
            robust_loss: config.solver.robust_loss,
            fix_intrinsics: IntrinsicsFixMask::default(),
            fix_distortion: *distortion_mask_in_percam_ba,
            fix_scheimpflug: *fix_scheimpflug_in_intrinsics,
            fix_poses: vec![0],
            bounds: None,
        };
        let backend_opts = BackendSolveOptions {
            max_iters,
            verbosity,
            ..Default::default()
        };
        if let Some(sensors) = optimized_sensors.as_mut() {
            recover_bad_scheimpflug_cameras_from_nominal(
                &scheimpflug_work_items,
                &mut optimized_cameras,
                sensors,
                &mut per_cam_reproj_errors,
                &mut per_cam_target_poses,
                solve_opts,
                backend_opts,
                config.rig.reference_camera_idx,
            )?;
        }
    }

    // Fail fast on any diverged camera before its garbage intrinsics poison the
    // linear rig init and the joint rig + hand-eye bundle adjustment.
    guard_percam_reproj_errors(&per_cam_reproj_errors)?;

    session.state.per_cam_intrinsics = Some(optimized_cameras.clone());
    session.state.per_cam_sensors = optimized_sensors.clone();
    session.state.per_cam_target_poses = Some(per_cam_target_poses);
    session.state.per_cam_reproj_errors = Some(per_cam_reproj_errors.clone());

    let avg_error: f64 =
        per_cam_reproj_errors.iter().sum::<f64>() / per_cam_reproj_errors.len() as f64;
    session.log_success_with_notes(
        "intrinsics_optimize_all",
        format!("avg_reproj_err={avg_error:.3}px"),
    );

    Ok(RigHandeyeIntrinsicsOptimizeAllResult {
        per_cam_intrinsics: optimized_cameras,
        per_cam_sensors: optimized_sensors,
        per_cam_reproj_errors,
    })
}

#[allow(clippy::too_many_arguments)]
fn optimize_scheimpflug_intrinsics_auto_multistart(
    cam_idx: usize,
    dataset: &PlanarDataset,
    initial: &ScheimpflugIntrinsicsParams,
    mut solve_opts: ScheimpflugIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
    init_iterations: usize,
    fix_k3_in_init: bool,
    zero_skew: bool,
) -> Result<ScheimpflugIntrinsicsEstimate, Error> {
    solve_opts.fix_poses.clear();
    let mut best: Option<ScheimpflugIntrinsicsEstimate> = None;
    let mut last_err: Option<Error> = None;

    let mut seeds = Vec::with_capacity(12);
    seeds.push((initial.clone(), false));
    for tilt_x in [-0.16, -0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16] {
        let init = match estimate_scheimpflug_intrinsics_iterative(
            dataset,
            ScheimpflugIntrinsicsInitOptions {
                iterations: init_iterations,
                distortion_opts: DistortionFitOptions {
                    fix_tangential: true,
                    fix_k3: fix_k3_in_init,
                    iters: 8,
                },
                zero_skew,
                tilt_x_seeds: vec![tilt_x],
                tilt_y_seeds: vec![0.0],
                refine_rounds: 1,
                ..Default::default()
            },
        ) {
            Ok(init) => init,
            Err(e) => {
                last_err = Some(Error::numerical(format!(
                    "Scheimpflug linear multistart failed for camera {cam_idx}, tilt_x={tilt_x}: {e}"
                )));
                continue;
            }
        };
        let params = ScheimpflugIntrinsicsParams::new(
            init.camera.k,
            BrownConrady5 {
                iters: 8,
                ..BrownConrady5::default()
            },
            init.sensor,
            init.poses,
        )?;
        if !seeds.iter().any(|(s, _)| {
            (s.sensor.tilt_x - params.sensor.tilt_x).abs() < 1e-4
                && (s.sensor.tilt_y - params.sensor.tilt_y).abs() < 1e-4
                && (s.intrinsics.fx - params.intrinsics.fx).abs() < 1.0
                && (s.intrinsics.fy - params.intrinsics.fy).abs() < 1.0
        }) {
            seeds.push((params, false));
        }
    }

    for (params, _) in seeds {
        let staged = ScheimpflugStagedInitOptions {
            tilt_x_seeds: vec![params.sensor.tilt_x],
            tilt_y_seeds: vec![params.sensor.tilt_y],
            sweep_max_iters: backend_opts.max_iters.clamp(1, 20),
            ..Default::default()
        };
        match optimize_scheimpflug_intrinsics_staged(
            dataset,
            &params,
            solve_opts.clone(),
            &staged,
            backend_opts.clone(),
        ) {
            Ok(candidate) if candidate.mean_reproj_error.is_finite() => {
                let better = best
                    .as_ref()
                    .is_none_or(|current| candidate.mean_reproj_error < current.mean_reproj_error);
                if better {
                    best = Some(candidate);
                }
                if best
                    .as_ref()
                    .is_some_and(|current| current.mean_reproj_error < 0.45)
                {
                    break;
                }
            }
            Ok(_) => {}
            Err(e) => {
                last_err = Some(Error::numerical(format!(
                    "Scheimpflug intrinsics optimization failed for camera {cam_idx}: {e}"
                )));
            }
        }
    }

    best.ok_or_else(|| {
        last_err.unwrap_or_else(|| {
            Error::numerical(format!(
                "all Scheimpflug intrinsics multistart candidates failed for camera {cam_idx}"
            ))
        })
    })
}

const SCHEIMPFLUG_NOMINAL_GOOD_REPROJ_PX: f64 = 0.75;
const SCHEIMPFLUG_NOMINAL_RECOVER_REPROJ_PX: f64 = 0.50;
const SCHEIMPFLUG_NOMINAL_TOP_LOCAL_CANDIDATES: usize = 10;
const SCHEIMPFLUG_NOMINAL_RIG_EXPAND_SOURCE: usize = 80;
const SCHEIMPFLUG_NOMINAL_TOP_RIG_CANDIDATES: usize = 6;

#[allow(clippy::too_many_arguments)]
fn recover_bad_scheimpflug_cameras_from_nominal(
    work_items: &[ScheimpflugCameraWork],
    cameras: &mut [PinholeCamera],
    sensors: &mut [ScheimpflugParams],
    per_cam_reproj_errors: &mut [f64],
    per_cam_target_poses: &mut [Vec<Option<Iso3>>],
    solve_opts: ScheimpflugIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
    reference_camera_idx: usize,
) -> Result<(), Error> {
    let good_indices = good_nominal_scheimpflug_indices(cameras, sensors, per_cam_reproj_errors);
    if good_indices.len() < 2 {
        return Ok(());
    };
    let Some(nominal) = build_nominal_scheimpflug_seed(cameras, sensors, &good_indices) else {
        return Ok(());
    };
    let rig_context = build_good_scheimpflug_rig_context(
        per_cam_target_poses,
        &good_indices,
        reference_camera_idx,
    )
    .ok();

    for item in work_items {
        let cam_idx = item.cam_idx;
        if per_cam_reproj_errors
            .get(cam_idx)
            .is_some_and(|e| e.is_finite() && *e <= SCHEIMPFLUG_NOMINAL_RECOVER_REPROJ_PX)
        {
            continue;
        }

        let Some(candidate) = recover_one_scheimpflug_camera_from_nominal(
            item,
            nominal,
            rig_context.as_ref(),
            cameras,
            sensors,
            per_cam_target_poses.len(),
            &solve_opts,
            &backend_opts,
        )?
        else {
            continue;
        };

        let current = per_cam_reproj_errors
            .get(cam_idx)
            .copied()
            .unwrap_or(f64::INFINITY);
        if candidate.mean_reproj_error.is_finite()
            && (!current.is_finite() || candidate.mean_reproj_error < current)
        {
            cameras[cam_idx] =
                make_pinhole_camera(candidate.params.intrinsics, candidate.params.distortion);
            sensors[cam_idx] = candidate.params.sensor;
            per_cam_reproj_errors[cam_idx] = candidate.mean_reproj_error;
            for (local_idx, &global_idx) in item.valid_indices.iter().enumerate() {
                per_cam_target_poses[global_idx][cam_idx] =
                    Some(candidate.params.camera_se3_target[local_idx]);
            }
        }
    }

    Ok(())
}

fn build_nominal_scheimpflug_seed(
    cameras: &[PinholeCamera],
    sensors: &[ScheimpflugParams],
    good_indices: &[usize],
) -> Option<NominalScheimpflugSeed> {
    let intrinsics = FxFyCxCySkew {
        fx: median_of(good_indices.iter().map(|&idx| cameras[idx].k.fx))?,
        fy: median_of(good_indices.iter().map(|&idx| cameras[idx].k.fy))?,
        cx: median_of(good_indices.iter().map(|&idx| cameras[idx].k.cx))?,
        cy: median_of(good_indices.iter().map(|&idx| cameras[idx].k.cy))?,
        skew: 0.0,
    };
    let distortion = BrownConrady5 {
        k1: median_of(good_indices.iter().map(|&idx| cameras[idx].dist.k1))?,
        k2: median_of(good_indices.iter().map(|&idx| cameras[idx].dist.k2))?,
        k3: median_of(good_indices.iter().map(|&idx| cameras[idx].dist.k3))?,
        p1: median_of(good_indices.iter().map(|&idx| cameras[idx].dist.p1))?,
        p2: median_of(good_indices.iter().map(|&idx| cameras[idx].dist.p2))?,
        iters: cameras[good_indices[0]].dist.iters.max(8),
    };
    let sensor = ScheimpflugParams {
        tilt_x: median_of(good_indices.iter().map(|&idx| sensors[idx].tilt_x))?,
        tilt_y: median_of(good_indices.iter().map(|&idx| sensors[idx].tilt_y))?,
    };

    Some(NominalScheimpflugSeed {
        intrinsics,
        distortion,
        sensor,
    })
}

fn good_nominal_scheimpflug_indices(
    cameras: &[PinholeCamera],
    sensors: &[ScheimpflugParams],
    errors: &[f64],
) -> Vec<usize> {
    cameras
        .iter()
        .zip(sensors.iter())
        .zip(errors.iter())
        .enumerate()
        .filter_map(|(idx, ((camera, sensor), error))| {
            is_good_nominal_scheimpflug_source(camera, *sensor, *error).then_some(idx)
        })
        .collect()
}

fn build_good_scheimpflug_rig_context(
    per_cam_target_poses: &[Vec<Option<Iso3>>],
    good_indices: &[usize],
    reference_camera_idx: usize,
) -> Result<GoodRigContext, Error> {
    if per_cam_target_poses.is_empty() || good_indices.len() < 2 {
        return Err(Error::invalid_input(
            "need at least two good cameras with target poses",
        ));
    }
    let ref_pos = good_indices
        .iter()
        .position(|&idx| idx == reference_camera_idx)
        .unwrap_or(0);
    let reduced: Vec<Vec<Option<Iso3>>> = per_cam_target_poses
        .iter()
        .map(|view| {
            good_indices
                .iter()
                .map(|&cam_idx| view.get(cam_idx).cloned().unwrap_or(None))
                .collect()
        })
        .collect();

    let reduced_extrinsics =
        estimate_extrinsics_from_cam_target_poses(&reduced, ref_pos).map_err(|e| {
            Error::numerical(format!(
                "good-camera rig context initialization failed: {e}"
            ))
        })?;
    let num_cameras = per_cam_target_poses[0].len();
    let mut cam_to_rig = vec![Iso3::identity(); num_cameras];
    for (reduced_idx, &cam_idx) in good_indices.iter().enumerate() {
        cam_to_rig[cam_idx] = reduced_extrinsics.cam_to_rig[reduced_idx];
    }

    Ok(GoodRigContext {
        cam_to_rig,
        rig_from_target: reduced_extrinsics.rig_from_target,
    })
}

fn is_good_nominal_scheimpflug_source(
    camera: &PinholeCamera,
    sensor: ScheimpflugParams,
    error: f64,
) -> bool {
    error.is_finite()
        && error <= SCHEIMPFLUG_NOMINAL_GOOD_REPROJ_PX
        && camera.k.fx.is_finite()
        && camera.k.fy.is_finite()
        && camera.k.cx.is_finite()
        && camera.k.cy.is_finite()
        && (100.0..10_000.0).contains(&camera.k.fx)
        && (100.0..10_000.0).contains(&camera.k.fy)
        && camera.dist.k1.is_finite()
        && camera.dist.k2.is_finite()
        && camera.dist.k3.is_finite()
        && camera.dist.p1.is_finite()
        && camera.dist.p2.is_finite()
        && camera.dist.k1.abs() < 2.0
        && camera.dist.k2.abs() < 2.0
        && sensor.tilt_x.is_finite()
        && sensor.tilt_y.is_finite()
        && sensor.tilt_x.abs() < 0.30
        && sensor.tilt_y.abs() < 0.30
}

fn median_of(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    })
}

#[allow(clippy::too_many_arguments)]
fn recover_one_scheimpflug_camera_from_nominal(
    item: &ScheimpflugCameraWork,
    nominal: NominalScheimpflugSeed,
    rig_context: Option<&GoodRigContext>,
    cameras: &[PinholeCamera],
    sensors: &[ScheimpflugParams],
    num_views: usize,
    solve_opts: &ScheimpflugIntrinsicsSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<Option<ScheimpflugIntrinsicsEstimate>, Error> {
    let mut scored = score_nominal_scheimpflug_seeds(&item.dataset, nominal);
    score_scheimpflug_seed(&item.dataset, &item.initial, &mut scored);
    scored.sort_by(|a, b| a.score.total_cmp(&b.score));

    let mut best = scored
        .first()
        .cloned()
        .map(scheimpflug_estimate_from_scored_seed);

    for scored_seed in scored
        .iter()
        .take(SCHEIMPFLUG_NOMINAL_TOP_LOCAL_CANDIDATES)
        .cloned()
    {
        let candidate = optimize_scheimpflug_intrinsics(
            &item.dataset,
            &scored_seed.params,
            nominal_pose_only_solve_opts(solve_opts.clone()),
            BackendSolveOptions {
                max_iters: backend_opts.max_iters.max(60),
                ..backend_opts.clone()
            },
        )
        .unwrap_or_else(|_| scheimpflug_estimate_from_scored_seed(scored_seed));

        let candidate = optimize_scheimpflug_intrinsics(
            &item.dataset,
            &candidate.params,
            nominal_tight_refine_solve_opts(solve_opts.clone(), nominal),
            BackendSolveOptions {
                max_iters: (backend_opts.max_iters * 2).max(100),
                ..backend_opts.clone()
            },
        )
        .unwrap_or(candidate);

        if candidate.mean_reproj_error.is_finite()
            && best
                .as_ref()
                .is_none_or(|current| candidate.mean_reproj_error < current.mean_reproj_error)
        {
            best = Some(candidate);
        }
    }

    if best.as_ref().is_some_and(|candidate| {
        candidate.mean_reproj_error <= SCHEIMPFLUG_NOMINAL_RECOVER_REPROJ_PX
    }) {
        return Ok(best);
    }

    if let Some(context) = rig_context {
        let mut rig_scored = scored
            .iter()
            .filter_map(|seed| {
                score_bad_scheimpflug_seed_against_good_rig(item, &seed.params, context)
                    .filter(|score| score.is_finite())
                    .map(|score| (score, seed.clone()))
            })
            .collect::<Vec<_>>();
        rig_scored.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut rig_candidates = nominal_rig_anchor_seeds(&item.dataset, nominal);
        rig_candidates.extend(
            rig_scored
                .into_iter()
                .take(SCHEIMPFLUG_NOMINAL_TOP_RIG_CANDIDATES)
                .map(|(_, seed)| seed),
        );

        for seed in rig_candidates {
            if let Some(candidate) = optimize_bad_scheimpflug_camera_against_good_rig(
                item,
                &seed.params,
                context,
                cameras,
                sensors,
                num_views,
                solve_opts,
                backend_opts,
            )? && candidate.mean_reproj_error.is_finite()
                && best
                    .as_ref()
                    .is_none_or(|current| candidate.mean_reproj_error < current.mean_reproj_error)
            {
                best = Some(candidate);
            }
        }
    }

    Ok(best)
}

fn score_bad_scheimpflug_seed_against_good_rig(
    item: &ScheimpflugCameraWork,
    params: &ScheimpflugIntrinsicsParams,
    context: &GoodRigContext,
) -> Option<f64> {
    let cam_to_rig =
        estimate_candidate_cam_to_rig(params, &item.valid_indices, &context.rig_from_target)?;
    let cam_from_rig = cam_to_rig.inverse();
    let camera = Camera::new(
        Pinhole,
        params.distortion,
        params.sensor.compile(),
        params.intrinsics,
    );
    let mut sum = 0.0;
    let mut count = 0usize;

    for (local_idx, &global_idx) in item.valid_indices.iter().enumerate() {
        if global_idx >= context.rig_from_target.len() {
            continue;
        }
        let cam_from_target = cam_from_rig * context.rig_from_target[global_idx];
        let view = &item.dataset.views[local_idx];
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = cam_from_target.transform_point(p3d);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2d).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }

    (count > 0).then_some(sum / count as f64)
}

#[allow(clippy::too_many_arguments)]
fn optimize_bad_scheimpflug_camera_against_good_rig(
    item: &ScheimpflugCameraWork,
    candidate_params: &ScheimpflugIntrinsicsParams,
    context: &GoodRigContext,
    cameras: &[PinholeCamera],
    sensors: &[ScheimpflugParams],
    num_views: usize,
    intrinsics_solve_opts: &ScheimpflugIntrinsicsSolveOptions,
    backend_opts: &BackendSolveOptions,
) -> Result<Option<ScheimpflugIntrinsicsEstimate>, Error> {
    let Some(initial_cam_to_rig) = estimate_candidate_cam_to_rig(
        candidate_params,
        &item.valid_indices,
        &context.rig_from_target,
    ) else {
        return Ok(None);
    };
    let dataset = bad_camera_only_scheimpflug_rig_dataset(item, cameras.len(), num_views)?;

    let mut rig_cameras = cameras.to_vec();
    rig_cameras[item.cam_idx] =
        make_pinhole_camera(candidate_params.intrinsics, candidate_params.distortion);
    let mut rig_sensors = sensors.to_vec();
    rig_sensors[item.cam_idx] = candidate_params.sensor;
    let mut cam_to_rig = context.cam_to_rig.clone();
    if cam_to_rig.len() != cameras.len() {
        return Ok(None);
    }
    cam_to_rig[item.cam_idx] = initial_cam_to_rig;

    let mut fix_extrinsics = vec![true; cameras.len()];
    fix_extrinsics[item.cam_idx] = false;
    let initial = RigExtrinsicsScheimpflugParams {
        cameras: rig_cameras,
        sensors: rig_sensors,
        cam_to_rig,
        rig_from_target: context.rig_from_target.clone(),
    };
    let solve_opts = RigExtrinsicsScheimpflugSolveOptions {
        robust_loss: intrinsics_solve_opts.robust_loss,
        default_fix: CameraFixMask::all_fixed(),
        camera_overrides: Vec::new(),
        default_scheimpflug_fix: ScheimpflugFixMask {
            tilt_x: true,
            tilt_y: true,
        },
        scheimpflug_overrides: Vec::new(),
        fix_extrinsics,
        fix_rig_poses: (0..num_views).collect(),
    };
    let Ok(result) = optimize_rig_extrinsics_scheimpflug(
        dataset,
        initial,
        solve_opts,
        BackendSolveOptions {
            max_iters: backend_opts.max_iters.max(80),
            ..backend_opts.clone()
        },
    ) else {
        return Ok(None);
    };
    let mean = result
        .per_cam_reproj_errors
        .get(item.cam_idx)
        .copied()
        .unwrap_or(f64::INFINITY);
    if !mean.is_finite() {
        return Ok(None);
    }

    let camera = &result.params.cameras[item.cam_idx];
    let sensor = result.params.sensors[item.cam_idx];
    let cam_from_rig = result.params.cam_to_rig[item.cam_idx].inverse();
    let mut poses = Vec::with_capacity(item.valid_indices.len());
    for &global_idx in &item.valid_indices {
        if global_idx >= result.params.rig_from_target.len() {
            return Ok(None);
        }
        poses.push(cam_from_rig * result.params.rig_from_target[global_idx]);
    }
    let params = ScheimpflugIntrinsicsParams::new(camera.k, camera.dist, sensor, poses)?;
    Ok(Some(ScheimpflugIntrinsicsEstimate {
        params,
        report: result.report,
        mean_reproj_error: mean,
    }))
}

fn bad_camera_only_scheimpflug_rig_dataset(
    item: &ScheimpflugCameraWork,
    num_cameras: usize,
    num_views: usize,
) -> Result<RigDataset<NoMeta>, Error> {
    let mut obs_by_view: Vec<Option<CorrespondenceView>> = vec![None; num_views];
    for (local_idx, &global_idx) in item.valid_indices.iter().enumerate() {
        if global_idx < num_views {
            obs_by_view[global_idx] = Some(item.dataset.views[local_idx].obs.clone());
        }
    }

    let views = obs_by_view
        .into_iter()
        .map(|obs| {
            let mut cameras = vec![None; num_cameras];
            cameras[item.cam_idx] = obs;
            RigView {
                obs: RigViewObs { cameras },
                meta: NoMeta,
            }
        })
        .collect();
    RigDataset::new(views, num_cameras).map_err(Error::from)
}

fn estimate_candidate_cam_to_rig(
    params: &ScheimpflugIntrinsicsParams,
    valid_indices: &[usize],
    rig_from_target: &[Iso3],
) -> Option<Iso3> {
    let mut candidates = Vec::with_capacity(valid_indices.len());
    for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
        if global_idx >= rig_from_target.len() || local_idx >= params.camera_se3_target.len() {
            continue;
        }
        candidates
            .push(rig_from_target[global_idx] * params.camera_se3_target[local_idx].inverse());
    }
    average_isometries_for_recovery(&candidates)
}

fn average_isometries_for_recovery(poses: &[Iso3]) -> Option<Iso3> {
    if poses.is_empty() {
        return None;
    }

    let mut t_sum = Vector3::<f64>::zeros();
    for pose in poses {
        t_sum += pose.translation.vector;
    }
    let t_avg = Translation3::from(t_sum / poses.len() as f64);

    let q0 = poses[0].rotation;
    let mut acc = nalgebra::Vector4::<f64>::zeros();
    for pose in poses {
        let coords = pose.rotation.coords;
        let sign = if q0.coords.dot(&coords) < 0.0 {
            -1.0
        } else {
            1.0
        };
        acc += coords * sign;
    }
    if acc.norm_squared() == 0.0 {
        return Some(Iso3::from_parts(t_avg, UnitQuaternion::identity()));
    }
    let q = Quaternion::from_vector(acc / poses.len() as f64).normalize();
    Some(Iso3::from_parts(t_avg, UnitQuaternion::from_quaternion(q)))
}

fn score_nominal_scheimpflug_seeds(
    dataset: &PlanarDataset,
    nominal: NominalScheimpflugSeed,
) -> Vec<ScoredScheimpflugSeed> {
    let mut scored = Vec::new();
    let pp_offsets = [-24.0, -12.0, 0.0, 12.0, 24.0];
    let tilt_x_offsets = [-0.020, -0.010, 0.0, 0.010, 0.020];
    let tilt_y_offsets = [-0.010, 0.0, 0.010];

    for dcx in pp_offsets {
        for dcy in pp_offsets {
            for dtx in tilt_x_offsets {
                for dty in tilt_y_offsets {
                    let mut k = nominal.intrinsics;
                    k.cx += dcx;
                    k.cy += dcy;
                    let sensor = ScheimpflugParams {
                        tilt_x: nominal.sensor.tilt_x + dtx,
                        tilt_y: nominal.sensor.tilt_y + dty,
                    };
                    score_scheimpflug_seed_parts(
                        dataset,
                        k,
                        nominal.distortion,
                        sensor,
                        &mut scored,
                    );
                }
            }
        }
    }

    scored.sort_by(|a, b| a.score.total_cmp(&b.score));
    let mut expanded = scored
        .iter()
        .take(SCHEIMPFLUG_NOMINAL_RIG_EXPAND_SOURCE)
        .cloned()
        .collect::<Vec<_>>();
    for base in scored.iter().take(SCHEIMPFLUG_NOMINAL_RIG_EXPAND_SOURCE) {
        for dk1 in [-0.04, 0.0, 0.04] {
            for dk2 in [-0.08, 0.0, 0.08] {
                if dk1 == 0.0 && dk2 == 0.0 {
                    continue;
                }
                let mut dist = base.params.distortion;
                dist.k1 += dk1;
                dist.k2 += dk2;
                score_scheimpflug_seed_parts(
                    dataset,
                    base.params.intrinsics,
                    dist,
                    base.params.sensor,
                    &mut expanded,
                );
            }
        }
    }
    expanded
}

fn nominal_rig_anchor_seeds(
    dataset: &PlanarDataset,
    nominal: NominalScheimpflugSeed,
) -> Vec<ScoredScheimpflugSeed> {
    let mut scored = Vec::new();
    for (dcx, dcy, dk1, dk2, dty) in [
        (12.0, 24.0, -0.04, 0.08, 0.0),
        (12.0, 24.0, -0.04, 0.12, 0.0),
        (18.0, 24.0, -0.04, 0.08, 0.0),
        (12.0, 18.0, -0.04, 0.08, 0.0),
        (24.0, 24.0, -0.04, 0.08, 0.0),
        (0.0, 24.0, -0.04, 0.08, 0.0),
        (12.0, 30.0, -0.04, 0.08, 0.0),
        (12.0, 24.0, -0.06, 0.08, 0.0),
        (12.0, 24.0, -0.02, 0.08, 0.0),
        (12.0, 24.0, -0.04, 0.08, -0.01),
    ] {
        let mut k = nominal.intrinsics;
        k.cx += dcx;
        k.cy += dcy;
        let mut dist = nominal.distortion;
        dist.k1 += dk1;
        dist.k2 += dk2;
        let sensor = ScheimpflugParams {
            tilt_x: nominal.sensor.tilt_x,
            tilt_y: nominal.sensor.tilt_y + dty,
        };
        score_scheimpflug_seed_parts(dataset, k, dist, sensor, &mut scored);
    }
    scored
}

fn score_scheimpflug_seed(
    dataset: &PlanarDataset,
    params: &ScheimpflugIntrinsicsParams,
    scored: &mut Vec<ScoredScheimpflugSeed>,
) {
    if let Ok(poses) = recover_scheimpflug_poses_for_seed(
        dataset,
        &params.intrinsics,
        &params.distortion,
        params.sensor,
    ) && let Ok(params) =
        ScheimpflugIntrinsicsParams::new(params.intrinsics, params.distortion, params.sensor, poses)
    {
        let score = mean_scheimpflug_reproj(dataset, &params);
        if score.is_finite() {
            scored.push(ScoredScheimpflugSeed { score, params });
        }
    }
}

fn score_scheimpflug_seed_parts(
    dataset: &PlanarDataset,
    intrinsics: FxFyCxCySkew<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
    scored: &mut Vec<ScoredScheimpflugSeed>,
) {
    if !intrinsics.fx.is_finite()
        || !intrinsics.fy.is_finite()
        || !intrinsics.cx.is_finite()
        || !intrinsics.cy.is_finite()
        || !distortion.k1.is_finite()
        || !distortion.k2.is_finite()
        || !sensor.tilt_x.is_finite()
        || !sensor.tilt_y.is_finite()
    {
        return;
    }
    let Ok(poses) = recover_scheimpflug_poses_for_seed(dataset, &intrinsics, &distortion, sensor)
    else {
        return;
    };
    let Ok(params) = ScheimpflugIntrinsicsParams::new(intrinsics, distortion, sensor, poses) else {
        return;
    };
    let score = mean_scheimpflug_reproj(dataset, &params);
    if score.is_finite() {
        scored.push(ScoredScheimpflugSeed { score, params });
    }
}

fn nominal_pose_only_solve_opts(
    mut opts: ScheimpflugIntrinsicsSolveOptions,
) -> ScheimpflugIntrinsicsSolveOptions {
    opts.fix_intrinsics = IntrinsicsFixMask::all_fixed();
    opts.fix_distortion = DistortionFixMask::all_fixed();
    opts.fix_scheimpflug = ScheimpflugFixMask {
        tilt_x: true,
        tilt_y: true,
    };
    opts.fix_poses.clear();
    opts.bounds = None;
    opts
}

fn nominal_tight_refine_solve_opts(
    mut opts: ScheimpflugIntrinsicsSolveOptions,
    nominal: NominalScheimpflugSeed,
) -> ScheimpflugIntrinsicsSolveOptions {
    let k = nominal.intrinsics;
    opts.fix_intrinsics = IntrinsicsFixMask::default();
    opts.fix_poses.clear();
    opts.bounds = Some(ScheimpflugBounds {
        fx: Some((0.94 * k.fx, 1.06 * k.fx)),
        fy: Some((0.94 * k.fy, 1.06 * k.fy)),
        cx: Some((k.cx - 60.0, k.cx + 60.0)),
        cy: Some((k.cy - 60.0, k.cy + 60.0)),
        tilt_x: Some((nominal.sensor.tilt_x - 0.06, nominal.sensor.tilt_x + 0.06)),
        tilt_y: Some((nominal.sensor.tilt_y - 0.04, nominal.sensor.tilt_y + 0.04)),
    });
    opts
}

fn scheimpflug_estimate_from_scored_seed(
    seed: ScoredScheimpflugSeed,
) -> ScheimpflugIntrinsicsEstimate {
    ScheimpflugIntrinsicsEstimate {
        params: seed.params,
        report: SolveReport {
            final_cost: seed.score * seed.score,
            num_iters: 0,
        },
        mean_reproj_error: seed.score,
    }
}

fn mean_scheimpflug_reproj(dataset: &PlanarDataset, params: &ScheimpflugIntrinsicsParams) -> f64 {
    let camera = Camera::new(
        Pinhole,
        params.distortion,
        params.sensor.compile(),
        params.intrinsics,
    );
    let mut sum = 0.0;
    let mut count = 0usize;

    for (view, pose) in dataset.views.iter().zip(params.camera_se3_target.iter()) {
        for (p3d, p2d) in view.obs.points_3d.iter().zip(view.obs.points_2d.iter()) {
            let p_cam = pose.transform_point(p3d);
            let Some(projected) = camera.project_point(&p_cam) else {
                continue;
            };
            let err = (projected - *p2d).norm();
            if err.is_finite() {
                sum += err;
                count += 1;
            }
        }
    }

    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

fn recover_scheimpflug_poses_for_seed(
    dataset: &PlanarDataset,
    intrinsics: &FxFyCxCySkew<f64>,
    distortion: &BrownConrady5<f64>,
    sensor: ScheimpflugParams,
) -> Result<Vec<Iso3>, Error> {
    let sensor_model = sensor.compile();
    let k_matrix = intrinsics.k_matrix();
    let mut poses = Vec::with_capacity(dataset.num_views());
    for (view_idx, view) in dataset.views.iter().enumerate() {
        let board = view.obs.planar_points();
        let ideal_pixels: Vec<Pt2> = view
            .obs
            .points_2d
            .iter()
            .map(|p| {
                let sensor_pt = intrinsics.pixel_to_sensor(p);
                let distorted = sensor_model.sensor_to_normalized(&sensor_pt);
                let undistorted = distortion.undistort(&distorted);
                intrinsics.sensor_to_pixel(&undistorted)
            })
            .collect();
        let h = dlt_homography(&board, &ideal_pixels).map_err(|e| {
            Error::numerical(format!(
                "Scheimpflug cross-camera pose homography failed for view {view_idx}: {e}"
            ))
        })?;
        let pose = estimate_planar_pose_from_h(&k_matrix, &h).map_err(|e| {
            Error::numerical(format!(
                "Scheimpflug cross-camera pose recovery failed for view {view_idx}: {e}"
            ))
        })?;
        poses.push(pose);
    }
    Ok(poses)
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
pub fn step_rig_init_with_seed(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: RigHandeyeRigManualInit,
) -> Result<RigHandeyeRigInitResult, Error> {
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

    session.state.initial_cam_se3_rig = Some(cam_se3_rig.clone());
    session.state.initial_rig_se3_target = Some(rig_se3_target.clone());

    let source = format_init_source(&manual_fields, &auto_fields);
    session.log_success_with_notes(
        "rig_init",
        format!(
            "ref_cam={}, {} views {}",
            reference_camera_idx, num_views, source
        ),
    );

    Ok(RigHandeyeRigInitResult {
        initial_cam_se3_rig: cam_se3_rig,
        initial_rig_se3_target: rig_se3_target,
    })
}

/// Initialize rig extrinsics using full auto-init (linear extrinsics fit).
///
/// Convenience wrapper around [`step_rig_init_with_seed`] with default seeds.
pub fn step_rig_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
) -> Result<RigHandeyeRigInitResult, Error> {
    step_rig_init_with_seed(session, RigHandeyeRigManualInit::default())
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
) -> Result<RigHandeyeRigOptimizeResult, Error> {
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

    let fix_intrinsics = if config.rig.refine_intrinsics_in_rig_ba {
        CameraFixMask::default()
    } else {
        CameraFixMask::all_fixed()
    };

    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == config.rig.reference_camera_idx)
        .collect();

    let fix_rig_poses = if config.rig.fix_first_rig_pose {
        vec![0]
    } else {
        Vec::new()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..Default::default()
    };

    // Convert input to NoMeta — both rig BA solvers expect NoMeta.
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

    struct RigOpt {
        cameras: Vec<vision_calibration_core::PinholeCamera>,
        cam_to_rig: Vec<Iso3>,
        rig_from_target: Vec<Iso3>,
        sensors: Option<Vec<ScheimpflugParams>>,
        per_cam_reproj_errors: Vec<f64>,
        mean_reproj_error: f64,
        final_cost: f64,
    }

    let opt: RigOpt = match &config.sensor {
        SensorMode::Pinhole => {
            let initial = RigExtrinsicsParams {
                cameras,
                cam_to_rig,
                rig_from_target,
            };
            let solve_opts = RigExtrinsicsSolveOptions {
                robust_loss: config.solver.robust_loss,
                default_fix: fix_intrinsics,
                camera_overrides: Vec::new(),
                fix_extrinsics,
                fix_rig_poses,
            };
            let result = match optimize_rig_extrinsics(
                rig_input_no_meta.clone(),
                initial,
                solve_opts,
                backend_opts,
            ) {
                Ok(r) => r,
                Err(e) => {
                    session.log_failure("rig_optimize", e.to_string());
                    return Err(Error::from(e));
                }
            };
            // Pinhole: recompute via the shared core helper for consistency
            // with the original pinhole behaviour.
            let cam_se3_rig: Vec<Iso3> = result
                .params
                .cam_to_rig
                .iter()
                .map(|t| t.inverse())
                .collect();
            let stats = compute_rig_reprojection_stats_per_camera(
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
            let total_count: usize = stats.iter().map(|s| s.count).sum();
            let total_error: f64 = stats.iter().map(|s| s.mean * (s.count as f64)).sum();
            let mean = total_error / total_count as f64;
            RigOpt {
                cameras: result.params.cameras,
                cam_to_rig: result.params.cam_to_rig,
                rig_from_target: result.params.rig_from_target,
                sensors: None,
                per_cam_reproj_errors: stats.iter().map(|s| s.mean).collect(),
                mean_reproj_error: mean,
                final_cost: result.report.final_cost,
            }
        }
        SensorMode::Scheimpflug {
            refine_scheimpflug_in_rig_ba,
            ..
        } => {
            let sensors = session
                .state
                .per_cam_sensors
                .clone()
                .ok_or_else(|| Error::not_available("per-camera Scheimpflug sensors"))?;
            let initial = RigExtrinsicsScheimpflugParams {
                cameras,
                sensors,
                cam_to_rig,
                rig_from_target,
            };
            let scheimpflug_fix = if *refine_scheimpflug_in_rig_ba {
                ScheimpflugFixMask::default()
            } else {
                ScheimpflugFixMask {
                    tilt_x: true,
                    tilt_y: true,
                }
            };
            let solve_opts = RigExtrinsicsScheimpflugSolveOptions {
                robust_loss: config.solver.robust_loss,
                default_fix: fix_intrinsics,
                camera_overrides: Vec::new(),
                default_scheimpflug_fix: scheimpflug_fix,
                scheimpflug_overrides: Vec::new(),
                fix_extrinsics,
                fix_rig_poses,
            };
            let result = match optimize_rig_extrinsics_scheimpflug(
                rig_input_no_meta.clone(),
                initial,
                solve_opts,
                backend_opts,
            ) {
                Ok(r) => r,
                Err(e) => {
                    session.log_failure("rig_optimize", e.to_string());
                    return Err(Error::from(e));
                }
            };
            // Scheimpflug: trust the optim's per-camera errors (computed with
            // the tilted projection chain). The shared core helper is
            // type-locked to `IdentitySensor`.
            RigOpt {
                cameras: result.params.cameras,
                cam_to_rig: result.params.cam_to_rig,
                rig_from_target: result.params.rig_from_target,
                sensors: Some(result.params.sensors),
                per_cam_reproj_errors: result.per_cam_reproj_errors,
                mean_reproj_error: result.mean_reproj_error,
                final_cost: result.report.final_cost,
            }
        }
    };

    let cam_se3_rig: Vec<Iso3> = opt.cam_to_rig.iter().map(|t| t.inverse()).collect();
    let mean_reproj_error = opt.mean_reproj_error;
    let per_cam_reproj_errors = opt.per_cam_reproj_errors.clone();
    let final_cost = opt.final_cost;
    session.state.rig_ba_cam_se3_rig = Some(cam_se3_rig);
    session.state.rig_ba_rig_se3_target = Some(opt.rig_from_target);
    session.state.rig_ba_reproj_error = Some(mean_reproj_error);
    session.state.rig_ba_per_cam_reproj_errors = Some(opt.per_cam_reproj_errors);
    session.state.per_cam_intrinsics = Some(opt.cameras);
    if opt.sensors.is_some() {
        session.state.per_cam_sensors = opt.sensors;
    }

    session.log_success_with_notes(
        "rig_optimize",
        format!("final_cost={final_cost:.2e}, mean_reproj_err={mean_reproj_error:.3}px"),
    );

    Ok(RigHandeyeRigOptimizeResult {
        mean_reproj_error,
        per_cam_reproj_errors,
        cam_se3_rig: session
            .state
            .rig_ba_cam_se3_rig
            .clone()
            .ok_or_else(|| Error::not_available("cam_se3_rig from rig BA"))?,
        rig_se3_target: session
            .state
            .rig_ba_rig_se3_target
            .clone()
            .ok_or_else(|| Error::not_available("rig_se3_target from rig BA"))?,
    })
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
pub fn step_handeye_init_with_seed(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: RigHandeyeHandeyeManualInit,
    opts: Option<HandeyeInitOptions>,
) -> Result<RigHandeyeHandeyeInitResult, Error> {
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

    Ok(RigHandeyeHandeyeInitResult {
        initial_handeye: handeye,
        initial_mode_target_pose: Some(mode_target_pose),
    })
}

/// Initialize hand-eye transform using the linear Tsai-Lenz DLT path.
///
/// Convenience wrapper around [`step_handeye_init_with_seed`] with default seeds.
pub fn step_handeye_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<RigHandeyeHandeyeInitResult, Error> {
    step_handeye_init_with_seed(session, RigHandeyeHandeyeManualInit::default(), opts)
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
) -> Result<RigHandeyeHandeyeOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?.clone();

    if !session.state.has_handeye_init() {
        return Err(Error::not_available(
            "hand-eye initialization (call step_handeye_init first)",
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

    // Don't refine intrinsics in final BA.
    let fix_intrinsics = CameraFixMask::all_fixed();

    let fix_extrinsics: Vec<bool> = if config.handeye_ba.refine_cam_se3_rig_in_handeye_ba {
        (0..input.num_cameras)
            .map(|i| i == config.rig.reference_camera_idx)
            .collect()
    } else {
        vec![true; input.num_cameras]
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..Default::default()
    };

    let output = match &config.sensor {
        SensorMode::Pinhole => {
            let initial = HandEyeParams {
                cameras,
                cam_to_rig,
                handeye,
                target_poses: vec![mode_target_pose],
            };

            let solve_opts = HandEyeSolveOptions {
                robust_loss: config.solver.robust_loss,
                default_fix: fix_intrinsics,
                camera_overrides: Vec::new(),
                fix_extrinsics,
                fix_handeye: false,
                fix_target_poses: Vec::new(),
                relax_target_poses: false,
                refine_robot_poses: config.handeye_ba.refine_robot_poses,
                robot_rot_sigma: config.handeye_ba.robot_rot_sigma,
                robot_trans_sigma: config.handeye_ba.robot_trans_sigma,
            };

            let handeye_dataset = HandEyeDataset::new(
                input.views.clone(),
                input.num_cameras,
                config.handeye_init.handeye_mode,
            )?;

            match optimize_handeye(handeye_dataset, initial, solve_opts, backend_opts) {
                Ok(r) => RigHandeyeOutput::Pinhole(r),
                Err(e) => {
                    session.log_failure("handeye_optimize", e.to_string());
                    return Err(Error::from(e));
                }
            }
        }
        SensorMode::Scheimpflug { .. } => {
            let sensors = session
                .state
                .per_cam_sensors
                .clone()
                .ok_or_else(|| Error::not_available("per-camera Scheimpflug sensors"))?;
            let initial = HandEyeScheimpflugParams {
                cameras,
                sensors,
                cam_to_rig,
                handeye,
                target_poses: vec![mode_target_pose],
            };

            let scheimpflug_fix = if config.handeye_ba.refine_scheimpflug_in_handeye_ba {
                ScheimpflugFixMask::default()
            } else {
                ScheimpflugFixMask {
                    tilt_x: true,
                    tilt_y: true,
                }
            };

            let solve_opts = HandEyeScheimpflugSolveOptions {
                robust_loss: config.solver.robust_loss,
                default_fix: fix_intrinsics,
                camera_overrides: Vec::new(),
                default_scheimpflug_fix: scheimpflug_fix,
                scheimpflug_overrides: Vec::new(),
                fix_extrinsics,
                fix_handeye: false,
                fix_target_poses: Vec::new(),
                relax_target_poses: false,
                refine_robot_poses: config.handeye_ba.refine_robot_poses,
                robot_rot_sigma: config.handeye_ba.robot_rot_sigma,
                robot_trans_sigma: config.handeye_ba.robot_trans_sigma,
            };

            let handeye_dataset = HandEyeScheimpflugDataset::new(
                input.views.clone(),
                input.num_cameras,
                config.handeye_init.handeye_mode,
            )?;

            match optimize_handeye_scheimpflug(handeye_dataset, initial, solve_opts, backend_opts) {
                Ok(r) => RigHandeyeOutput::Scheimpflug(r),
                Err(e) => {
                    session.log_failure("handeye_optimize", e.to_string());
                    return Err(Error::from(e));
                }
            }
        }
    };

    let final_cost = output.final_cost();
    let mean_reproj_error = final_cost.sqrt();
    session.state.final_cost = Some(final_cost);
    session.state.final_reproj_error = Some(mean_reproj_error);

    session.set_output(output);

    session.log_success_with_notes("handeye_optimize", format!("final_cost={final_cost:.2e}"));

    Ok(RigHandeyeHandeyeOptimizeResult {
        mean_reproj_error,
        final_cost,
    })
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
    let manual = session
        .config
        .intrinsics
        .manual_init
        .clone()
        .unwrap_or_default();
    let _ = step_intrinsics_init_all_with_seed(session, manual, None)?;
    let _ = step_intrinsics_optimize_all(session, None)?;
    let _ = step_rig_init(session)?;
    let _ = step_rig_optimize(session, None)?;
    let _ = step_handeye_init(session, None)?;
    let _ = step_handeye_optimize(session, None)?;
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
    fn run_calibration_uses_configured_intrinsics_manual_init() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        session
            .set_config(super::super::problem::RigHandeyeConfig {
                intrinsics: super::super::problem::RigHandeyeIntrinsicsConfig {
                    manual_init: Some(RigHandeyeIntrinsicsManualInit {
                        per_cam_intrinsics: Some(vec![
                            FxFyCxCySkew {
                                fx: 800.0,
                                fy: 780.0,
                                cx: 640.0,
                                cy: 360.0,
                                skew: 0.0,
                            },
                            FxFyCxCySkew {
                                fx: 810.0,
                                fy: 790.0,
                                cx: 640.0,
                                cy: 360.0,
                                skew: 0.0,
                            },
                        ]),
                        per_cam_distortion: Some(vec![
                            BrownConrady5::default(),
                            BrownConrady5::default(),
                        ]),
                        per_cam_sensors: None,
                    }),
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();

        run_calibration(&mut session).unwrap();
        let notes = session.log()[0]
            .notes
            .as_ref()
            .expect("intrinsics init logs seed provenance");
        assert!(
            notes.contains("manual: per_cam_intrinsics, per_cam_distortion"),
            "unexpected intrinsics-init notes: {notes}"
        );
    }

    #[test]
    fn step_set_handeye_init_eye_to_hand_recovers_target_on_gripper() {
        // Regression test: in EyeToHand mode the
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
        step_handeye_init_with_seed(&mut session, manual, None).unwrap();

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
