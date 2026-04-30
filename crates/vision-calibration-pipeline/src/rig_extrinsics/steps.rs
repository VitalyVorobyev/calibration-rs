//! Step functions for multi-camera rig extrinsics calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigExtrinsicsProblem>` to perform calibration.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraFixMask, DistortionFixMask, FxFyCxCySkew, IntrinsicsFixMask, Iso3, NoMeta,
    Real, ScheimpflugParams, View, compute_rig_reprojection_stats_per_camera, make_pinhole_camera,
};
use vision_calibration_linear::estimate_extrinsics_from_cam_target_poses;
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RigExtrinsicsParams,
    RigExtrinsicsScheimpflugParams, RigExtrinsicsScheimpflugSolveOptions,
    RigExtrinsicsSolveOptions, ScheimpflugIntrinsicsParams, ScheimpflugIntrinsicsSolveOptions,
    optimize_planar_intrinsics, optimize_rig_extrinsics, optimize_rig_extrinsics_scheimpflug,
    optimize_scheimpflug_intrinsics,
};

use crate::rig_family::{
    RigIntrinsicsSeeds, SensorFlavour, bootstrap_rig_intrinsics, format_init_source,
    views_to_planar_dataset,
};
use crate::session::CalibrationSession;

use super::problem::{RigExtrinsicsInput, RigExtrinsicsOutput, RigExtrinsicsProblem, SensorMode};

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

/// Manual seeds for the **per-camera intrinsics stage** of rig extrinsics
/// calibration.
///
/// All fields are `Option<T>`. When `per_cam_intrinsics` is `Some`, the bootstrap
/// Zhang's auto-fit is skipped for *every* camera; per-camera target poses are
/// then recovered from per-camera homographies using the seeded intrinsics. When
/// `None`, the existing per-camera auto-init runs.
///
/// `per_cam_distortion` follows the same convention as `PlanarManualInit`: when
/// intrinsics are seeded but distortion is not, distortion defaults to zeros. When
/// intrinsics are auto-init'd, the bootstrap fitted distortion is used unless
/// distortion is seeded explicitly.
///
/// `per_cam_sensors` is consulted only when [`SensorMode::Scheimpflug`] is
/// configured; for [`SensorMode::Pinhole`] it is silently ignored.
///
/// Per-camera vectors must have length equal to `input.num_cameras`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigIntrinsicsManualInit {
    /// Per-camera intrinsics seeds. `None` runs Zhang's per camera.
    pub per_cam_intrinsics: Option<Vec<FxFyCxCySkew<Real>>>,
    /// Per-camera distortion seeds.
    pub per_cam_distortion: Option<Vec<BrownConrady5<Real>>>,
    /// Per-camera Scheimpflug sensor seeds (Scheimpflug mode only).
    #[serde(default)]
    pub per_cam_sensors: Option<Vec<ScheimpflugParams>>,
}

/// Manual seeds for the **rig extrinsics stage**.
///
/// `cam_se3_rig` and `rig_se3_target` are geometrically coupled — providing one
/// without the other is ambiguous. Per ADR 0011, both must be `Some` or both must
/// be `None`. A mismatched configuration returns `Error::InvalidInput`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RigExtrinsicsManualInit {
    /// Per-camera `T_C_R` (camera-from-rig). `None` runs the linear extrinsics fit.
    pub cam_se3_rig: Option<Vec<Iso3>>,
    /// Per-view `T_R_T` (rig-from-target). Must be paired with `cam_se3_rig`.
    pub rig_se3_target: Option<Vec<Iso3>>,
}

/// Options for rig BA optimization.
#[derive(Debug, Clone, Default)]
pub struct RigOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Extract views for a single camera from the rig dataset.
///
/// Input-type-specific; the rest of the per-camera bootstrap chain
/// (`views_to_planar_dataset`, `estimate_target_pose`) is shared via
/// [`crate::rig_family`].
fn extract_camera_views(input: &RigExtrinsicsInput, cam_idx: usize) -> Vec<Option<View<NoMeta>>> {
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

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics for all cameras from any combination of manual seeds
/// and auto-estimation (Zhang's method per camera).
///
/// This is the load-bearing intrinsics-stage init. [`step_intrinsics_init_all`]
/// is a thin delegate with `RigIntrinsicsManualInit::default()`.
///
/// See [`RigIntrinsicsManualInit`] for partial-seed semantics. Per-camera target
/// poses are *always* recovered from per-camera homographies — using the seeded
/// intrinsics if provided, otherwise the auto-fit intrinsics.
///
/// # Errors
///
/// - Input not set, or any camera has fewer than 3 views with observations.
/// - `manual.per_cam_intrinsics` (or `per_cam_distortion`) length mismatches
///   `input.num_cameras`.
pub fn step_set_intrinsics_init_all(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    manual: RigIntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(config.intrinsics_init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: config.fix_k3,
            fix_tangential: config.fix_tangential,
            iters: 8,
        },
        zero_skew: config.zero_skew,
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

    session.state.per_cam_intrinsics = Some(bootstrap.bundle.cameras);
    session.state.per_cam_sensors = bootstrap.bundle.scheimpflug;
    session.state.per_cam_target_poses = Some(bootstrap.per_cam_target_poses);

    let source = format_init_source(&bootstrap.manual_fields, &bootstrap.auto_fields);
    session.log_success_with_notes(
        "intrinsics_init_all",
        format!("initialized {num_cameras} cameras {source}"),
    );

    Ok(())
}

/// Initialize intrinsics for all cameras using full auto-init (Zhang's per camera).
///
/// Convenience wrapper around [`step_set_intrinsics_init_all`] with default seeds.
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<(), Error> {
    step_set_intrinsics_init_all(session, RigIntrinsicsManualInit::default(), opts)
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
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
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

    let max_iters = opts.max_iters.unwrap_or(config.max_iters);
    let verbosity = opts.verbosity.unwrap_or(config.verbosity);

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);
    let mut optimized_sensors = match &config.sensor {
        SensorMode::Pinhole => None,
        SensorMode::Scheimpflug { .. } => Some(Vec::with_capacity(input.num_cameras)),
    };

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
                    .ok_or_else(|| Error::not_available("initial pose"))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        match &config.sensor {
            SensorMode::Pinhole => {
                let initial_params =
                    PlanarIntrinsicsParams::new(per_cam_intrinsics[cam_idx].clone(), initial_poses)
                        .map_err(|e| {
                            Error::numerical(format!(
                                "failed to build params for camera {cam_idx}: {e}"
                            ))
                        })?;

                let solve_opts = PlanarIntrinsicsSolveOptions {
                    robust_loss: config.robust_loss,
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

                optimized_cameras.push(result.params.camera.clone());
                per_cam_reproj_errors.push(result.mean_reproj_error);
            }
            SensorMode::Scheimpflug {
                fix_scheimpflug_in_intrinsics,
                ..
            } => {
                let cam = &per_cam_intrinsics[cam_idx];
                let sensor = per_cam_sensors_in.as_ref().unwrap()[cam_idx];
                let initial_params =
                    ScheimpflugIntrinsicsParams::new(cam.k, cam.dist, sensor, initial_poses)?;

                // Radial-only distortion (k1, k2 free; k3, p1, p2 fixed). Tangential
                // distortion can absorb tilt-like geometric signal and interfere with
                // Scheimpflug tilt optimization (matches the original
                // rig_scheimpflug_extrinsics behaviour).
                let solve_opts = ScheimpflugIntrinsicsSolveOptions {
                    robust_loss: config.robust_loss,
                    fix_intrinsics: IntrinsicsFixMask::default(),
                    fix_distortion: DistortionFixMask::radial_only(),
                    fix_scheimpflug: *fix_scheimpflug_in_intrinsics,
                    fix_poses: vec![0],
                };

                let backend_opts = BackendSolveOptions {
                    max_iters,
                    verbosity,
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
                        "Scheimpflug intrinsics optimization failed for camera {cam_idx}: {e}"
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
                optimized_sensors
                    .as_mut()
                    .unwrap()
                    .push(result.params.sensor);
                per_cam_reproj_errors.push(result.mean_reproj_error);
            }
        }
    }

    session.state.per_cam_intrinsics = Some(optimized_cameras);
    session.state.per_cam_sensors = optimized_sensors;
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

/// Initialize rig extrinsics from any combination of manual seeds and auto-
/// estimation (linear extrinsics fit).
///
/// This is the load-bearing rig-stage init. [`step_rig_init`] is a thin delegate
/// with `RigExtrinsicsManualInit::default()`.
///
/// **Coupling rule** (ADR 0011): `cam_se3_rig` and `rig_se3_target` are
/// geometrically coupled. Both must be `Some` or both `None`. Mismatched
/// configurations return `Error::InvalidInput`.
///
/// # Errors
///
/// - Input not set, or per-camera intrinsics not computed.
/// - Coupling rule violated (one of `cam_se3_rig` / `rig_se3_target` seeded
///   without the other).
/// - Vector length mismatches: `cam_se3_rig.len() != input.num_cameras`, or
///   `rig_se3_target.len() != input.num_views()`.
/// - Auto-init linear estimation fails.
pub fn step_set_rig_init(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
    manual: RigExtrinsicsManualInit,
) -> Result<(), Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_init_all first)",
        ));
    }

    let num_views = input.num_views();
    let num_cameras = input.num_cameras;
    let reference_camera_idx = session.config.reference_camera_idx;

    // Coupling rule: both-or-neither.
    match (&manual.cam_se3_rig, &manual.rig_se3_target) {
        (Some(_), None) | (None, Some(_)) => {
            let msg = "RigExtrinsicsManualInit: cam_se3_rig and rig_se3_target must both be Some or \
                 both be None (geometrically coupled per ADR 0011)";
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
                let msg = format!(
                    "manual cam_se3_rig length ({}) does not match num_cameras ({})",
                    cam_se3_rig.len(),
                    num_cameras
                );
                session.log_failure("rig_init", msg.clone());
                return Err(Error::invalid_input(msg));
            }
            if rig_se3_target.len() != num_views {
                let msg = format!(
                    "manual rig_se3_target length ({}) does not match num_views ({})",
                    rig_se3_target.len(),
                    num_views
                );
                session.log_failure("rig_init", msg.clone());
                return Err(Error::invalid_input(msg));
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

/// Initialize rig extrinsics using full auto-init (linear extrinsics fit).
///
/// Convenience wrapper around [`step_set_rig_init`] with default seeds.
pub fn step_rig_init(session: &mut CalibrationSession<RigExtrinsicsProblem>) -> Result<(), Error> {
    step_set_rig_init(session, RigExtrinsicsManualInit::default())
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
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
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

    let fix_intrinsics = if config.refine_intrinsics_in_rig_ba {
        CameraFixMask::default()
    } else {
        CameraFixMask::all_fixed()
    };

    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == config.reference_camera_idx)
        .collect();

    let fix_rig_poses = if config.fix_first_rig_pose {
        vec![0]
    } else {
        Vec::new()
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.verbosity),
        ..Default::default()
    };

    let output = match &config.sensor {
        SensorMode::Pinhole => {
            let initial = RigExtrinsicsParams {
                cameras,
                cam_to_rig,
                rig_from_target,
            };
            let solve_opts = RigExtrinsicsSolveOptions {
                robust_loss: config.robust_loss,
                default_fix: fix_intrinsics,
                camera_overrides: Vec::new(),
                fix_extrinsics,
                fix_rig_poses,
            };
            match optimize_rig_extrinsics(input.clone(), initial, solve_opts, backend_opts) {
                Ok(r) => RigExtrinsicsOutput::Pinhole(r),
                Err(e) => {
                    session.log_failure("rig_optimize", e.to_string());
                    return Err(Error::from(e));
                }
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
                vision_calibration_optim::ScheimpflugFixMask::default()
            } else {
                vision_calibration_optim::ScheimpflugFixMask {
                    tilt_x: true,
                    tilt_y: true,
                }
            };
            let solve_opts = RigExtrinsicsScheimpflugSolveOptions {
                robust_loss: config.robust_loss,
                default_fix: fix_intrinsics,
                camera_overrides: Vec::new(),
                default_scheimpflug_fix: scheimpflug_fix,
                scheimpflug_overrides: Vec::new(),
                fix_extrinsics,
                fix_rig_poses,
            };
            match optimize_rig_extrinsics_scheimpflug(
                input.clone(),
                initial,
                solve_opts,
                backend_opts,
            ) {
                Ok(r) => RigExtrinsicsOutput::Scheimpflug(r),
                Err(e) => {
                    session.log_failure("rig_optimize", e.to_string());
                    return Err(Error::from(e));
                }
            }
        }
    };

    let cam_se3_rig: Vec<Iso3> = output.cam_to_rig().iter().map(|t| t.inverse()).collect();
    // For pinhole rigs we recompute per-camera stats via the shared core helper
    // (matches the original rig_extrinsics behaviour). For Scheimpflug rigs the
    // helper is type-locked to `IdentitySensor`, so we trust the optim's
    // already-computed mean and per-camera errors (computed with the tilted
    // projection chain).
    let (per_cam_errors, mean_reproj_error) = match &output {
        RigExtrinsicsOutput::Pinhole(e) => {
            let stats = compute_rig_reprojection_stats_per_camera(
                &e.params.cameras,
                session.require_input()?,
                &cam_se3_rig,
                &e.params.rig_from_target,
            )
            .map_err(|err| {
                Error::numerical(format!(
                    "failed to compute per-camera rig BA reprojection error: {err}"
                ))
            })?;
            let total_count: usize = stats.iter().map(|s| s.count).sum();
            let total_error: f64 = stats.iter().map(|s| s.mean * (s.count as f64)).sum();
            let mean = total_error / (total_count as f64);
            let per_cam: Vec<f64> = stats.iter().map(|s| s.mean).collect();
            (per_cam, mean)
        }
        RigExtrinsicsOutput::Scheimpflug(e) => {
            (e.per_cam_reproj_errors.clone(), e.mean_reproj_error)
        }
    };
    session.state.rig_ba_final_cost = Some(output.final_cost());
    session.state.rig_ba_reproj_error = Some(mean_reproj_error);
    session.state.rig_ba_per_cam_reproj_errors = Some(per_cam_errors);

    let final_cost = output.final_cost();
    session.set_output(output);

    session.log_success_with_notes(
        "rig_optimize",
        format!("final_cost={final_cost:.2e}, mean_reproj_err={mean_reproj_error:.3}px"),
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Function
// ─────────────────────────────────────────────────────────────────────────────

/// Run the full rig extrinsics calibration pipeline.
///
/// Runs: intrinsics_init_all → intrinsics_optimize_all → rig_init → rig_optimize.
///
/// # Errors
///
/// Any error from the constituent steps.
pub fn run_calibration(
    session: &mut CalibrationSession<RigExtrinsicsProblem>,
) -> Result<(), Error> {
    step_intrinsics_init_all(session, None)?;
    step_intrinsics_optimize_all(session, None)?;
    step_rig_init(session)?;
    step_rig_optimize(session, None)?;
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

    fn make_test_input() -> RigExtrinsicsInput {
        // Ground truth cameras and rig
        let cam0 = make_test_camera(0.0);
        let cam1 = make_test_camera(10.0);
        let cam0_se3_rig = Iso3::identity();
        let cam1_se3_rig = make_iso((0.0, 0.0, 0.1), (0.2, 0.0, 0.0));

        // Board points
        let board_pts: Vec<Pt3> = (0..6)
            .flat_map(|i| (0..5).map(move |j| Pt3::new(i as f64 * 0.05, j as f64 * 0.05, 0.0)))
            .collect();

        // Rig poses
        let rig_poses = [
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            make_iso((0.1, 0.0, 0.0), (0.1, 0.0, 1.0)),
            make_iso((0.0, 0.1, 0.0), (0.0, 0.1, 1.0)),
            make_iso((0.05, 0.05, 0.0), (-0.1, 0.0, 1.0)),
        ];

        let views: Vec<RigView<NoMeta>> = rig_poses
            .iter()
            .map(|rig_se3_target| {
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
                    meta: NoMeta,
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
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();

        step_intrinsics_init_all(&mut session, None).unwrap();

        assert!(session.state.has_per_cam_intrinsics());
        let cameras = session.state.per_cam_intrinsics.as_ref().unwrap();
        assert_eq!(cameras.len(), 2);
    }

    #[test]
    fn step_rig_init_requires_intrinsics() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();

        let result = step_rig_init(&mut session);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("intrinsics"));
    }

    #[test]
    fn step_rig_optimize_requires_init() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();
        step_intrinsics_optimize_all(&mut session, None).unwrap();

        let result = step_rig_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rig"));
    }

    #[test]
    fn set_input_clears_state() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init_all(&mut session, None).unwrap();

        assert!(session.state.has_per_cam_intrinsics());

        session.set_input(make_test_input()).unwrap();
        assert!(!session.state.has_per_cam_intrinsics());
    }

    #[test]
    fn set_config_keeps_output() {
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();

        assert!(session.has_output());

        session
            .set_config(super::super::problem::RigExtrinsicsConfig {
                max_iters: 100,
                ..Default::default()
            })
            .unwrap();

        assert!(session.has_output());
    }

    #[test]
    fn json_roundtrip() {
        let mut session =
            CalibrationSession::<RigExtrinsicsProblem>::with_description("Test rig extrinsics");
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<RigExtrinsicsProblem>::from_json(&json).unwrap();

        assert_eq!(
            restored.metadata.description,
            Some("Test rig extrinsics".to_string())
        );
        assert!(restored.has_input());
        assert!(restored.has_output());
        assert_eq!(restored.exports.len(), 1);
    }

    #[test]
    fn rig_optimize_keeps_reprojection_error_reasonable() {
        let input = make_test_input();
        let mut session = CalibrationSession::<RigExtrinsicsProblem>::new();
        session.set_input(input.clone()).unwrap();

        run_calibration(&mut session).unwrap();
        let output = session.require_output().unwrap().clone();

        let cam_se3_rig: Vec<Iso3> = output.cam_to_rig().iter().map(|t| t.inverse()).collect();
        let reproj_stats = vision_calibration_core::compute_rig_reprojection_stats(
            output.cameras(),
            &input,
            &cam_se3_rig,
            output.rig_from_target(),
        )
        .unwrap();
        let per_cam_stats = compute_rig_reprojection_stats_per_camera(
            output.cameras(),
            &input,
            &cam_se3_rig,
            output.rig_from_target(),
        )
        .unwrap();
        let mean_err = reproj_stats.mean;
        assert!(
            mean_err.is_finite() && mean_err < 1.0e-3,
            "mean reprojection error too large: {} px",
            mean_err
        );

        let state_mean = session.state.rig_ba_reproj_error.unwrap();
        assert!(
            (state_mean - mean_err).abs() < 1e-12,
            "state reprojection error mismatch: state={}, computed={}",
            state_mean,
            mean_err
        );

        let state_per_cam = session.state.rig_ba_per_cam_reproj_errors.as_ref().unwrap();
        assert_eq!(state_per_cam.len(), per_cam_stats.len());
        for (i, (state_mean, computed)) in state_per_cam
            .iter()
            .zip(per_cam_stats.iter().map(|s| s.mean))
            .enumerate()
        {
            assert!(
                (state_mean - computed).abs() < 1e-12,
                "camera {} error mismatch: state={}, computed={}",
                i,
                state_mean,
                computed
            );
        }

        // Export should use `cam_se3_rig` convention.
        let export = session.export().unwrap();
        assert_eq!(export.cam_se3_rig.len(), cam_se3_rig.len());
        for (a, b) in export.cam_se3_rig.iter().zip(cam_se3_rig.iter()) {
            let dt = (a.translation.vector - b.translation.vector).norm();
            assert!(dt < 1e-10, "export translation error {}", dt);
        }
    }
}
