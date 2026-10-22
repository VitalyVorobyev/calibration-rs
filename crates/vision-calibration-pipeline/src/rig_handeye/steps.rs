//! Step functions for multi-camera rig hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigHandeyeProblem>` to perform calibration.

use crate::Error;
use nalgebra::{Quaternion, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraModelSpec, CameraParams, CorrespondenceView, DistortionKind,
    DistortionModel, FxFyCxCy, IntrinsicsModel, Iso3, NoMeta, PlanarDataset, Pt2, RigDataset,
    RigView, RigViewObs, ScheimpflugParams, SensorKind, SensorModel, SensorParams, Uncertainty,
    View,
};
use vision_calibration_linear::extrinsics::estimate_extrinsics_from_cam_target_poses;
use vision_calibration_linear::handeye::{estimate_gripper_se3_target_dlt, estimate_handeye_dlt};
use vision_calibration_linear::prelude::*;
use vision_calibration_optim::{
    BackendSolveOptions, CameraBounds, HandeyeDataset, HandeyeMode, HandeyeParams,
    HandeyeSolveOptions, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RigExtrinsicsParams, RigExtrinsicsSolveOptions,
    ScheimpflugStagedInitOptions, SolveReport, optimize_handeye, optimize_planar_intrinsics,
    optimize_rig_extrinsics, optimize_scheimpflug_intrinsics_staged,
};

use crate::planar_family::{camera_solve_opts, embed_distortion, optimize_camera};
use crate::rig_family::{
    RigIntrinsicsSeeds, bootstrap_rig_intrinsics, check_camera_models, format_init_source,
    guard_percam_reproj_errors, rig_ba_fix, rig_distortion, rig_tilt, tilted_planar_params,
    views_to_planar_dataset,
};
use crate::session::CalibrationSession;

use super::problem::{IntrinsicsManualInit, RigHandeyeInput, RigHandeyeProblem};

/// The model of the cameras the nominal recovery serves: Brown-Conrady
/// distortion and a Scheimpflug sensor.
const TILTED_BROWN_CONRADY: CameraModelSpec =
    CameraModelSpec::pinhole(DistortionKind::BrownConrady5, SensorKind::Scheimpflug);

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::common::{
    HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions, IntrinsicsOptimizeOptions,
    RigInitOptions, RigOptimizeOptions,
};

/// Manual seeds for the **rig extrinsics stage** of rig hand-eye calibration.
///
/// `cam_se3_rig` and `rig_se3_target` are coupled — both must be
/// `Some` or both `None`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigManualInit {
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
pub struct HandeyeManualInit {
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
    initial: PlanarIntrinsicsParams,
    solve_opts: PlanarIntrinsicsSolveOptions,
}

#[derive(Debug, Clone, Copy)]
struct NominalScheimpflugSeed {
    intrinsics: FxFyCxCy<f64>,
    distortion: BrownConrady5<f64>,
    sensor: ScheimpflugParams,
}

#[derive(Debug, Clone)]
struct ScoredScheimpflugSeed {
    score: f64,
    params: PlanarIntrinsicsParams,
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
/// Structurally similar to [`crate::rig_extrinsics::IntrinsicsInitResult`]
/// but lives in this module so the two problem types can evolve independently.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsInitResult {
    /// Per-camera initial parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera target poses: `[view][cam] -> Option<Iso3>` (`camera_se3_target`).
    /// Inner `None` marks views where that camera did not observe the target.
    pub per_cam_target_poses: Vec<Vec<Option<Iso3>>>,
}

/// Typed return value of [`step_intrinsics_optimize_all`] for rig hand-eye
/// calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsOptimizeResult {
    /// Per-camera refined parameters, each in its camera's model.
    pub cameras: Vec<CameraParams>,
    /// Per-camera mean reprojection error in pixels.
    pub per_cam_reproj_errors: Vec<f64>,
    /// Per-camera target poses refined alongside the intrinsics:
    /// `[view][cam] -> Option<Iso3>` (`camera_se3_target`). This is what the
    /// unseeded rig init consumes; seeded rig inits (e.g.
    /// [`crate::device_seed::rig_layout_seed`]) should anchor on these, not
    /// on the coarser init-time poses.
    pub per_cam_target_poses: Vec<Vec<Option<Iso3>>>,
}

/// Typed return value of [`step_rig_init`] / [`step_rig_init_with_seed`] for rig
/// hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigInitResult {
    /// Initial per-camera `T_C_R` — camera-from-rig (reference camera is identity).
    pub initial_cam_se3_rig: Vec<Iso3>,
    /// Initial per-view `T_R_T` — rig-from-target.
    pub initial_rig_se3_target: Vec<Iso3>,
}

/// Typed return value of [`step_rig_optimize`] for rig hand-eye calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RigOptimizeResult {
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
/// [`HandeyeManualInit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeInitResult {
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeOptimizeResult {
    /// Mean reprojection error in pixels after the final hand-eye BA.
    pub mean_reproj_error: f64,
    /// Final solver cost from the hand-eye BA.
    pub final_cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics for all cameras from any combination of manual seeds
/// and auto-estimation, each camera in its own model (`config.camera_for(i)`).
///
/// Runs Zhang's method with iterative distortion estimation on each camera
/// independently (the tilted variant for a Scheimpflug camera). An unset
/// `intrinsics.fix_tangential` holds the tangential terms of each
/// Scheimpflug camera and keeps them free for every other camera. See `rig_extrinsics::step_intrinsics_init_all_with_seed` for the
/// seed semantics.
///
/// # Errors
///
/// - Input not set
/// - Any camera has fewer than 3 views with observations
/// - A seed's length mismatches the camera count, or a non-zero tilt seed is
///   given for an untilted camera
pub fn step_intrinsics_init_all_with_seed(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    session.validate()?;
    let input = session.require_input()?;
    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    let models: Vec<CameraModelSpec> = (0..num_cameras)
        .map(|i| config.camera_for(i).model)
        .collect();

    let seeds = RigIntrinsicsSeeds {
        per_cam_intrinsics: manual.per_cam_intrinsics,
        per_cam_distortion: manual.per_cam_distortion,
        per_cam_sensors: manual.per_cam_sensors,
    };

    let bootstrap = match bootstrap_rig_intrinsics(
        &models,
        num_views,
        |cam_idx| extract_camera_views(input, cam_idx),
        seeds,
        &config.intrinsics,
        opts.iterations,
    ) {
        Ok(b) => b,
        Err(e) => {
            session.log_failure("intrinsics_init_all", e.to_string());
            return Err(e);
        }
    };

    let cameras = bootstrap.cameras;
    let per_cam_target_poses = bootstrap.per_cam_target_poses;

    session.state.cameras = Some(cameras.clone());
    session.state.per_cam_target_poses = Some(per_cam_target_poses.clone());
    session.state.per_cam_intrinsics_auto = bootstrap.manual_fields.is_empty();

    let source = format_init_source(&bootstrap.manual_fields, &bootstrap.auto_fields);
    session.log_success_with_notes(
        "intrinsics_init_all",
        format!("initialized {num_cameras} cameras {source}"),
    );

    Ok(IntrinsicsInitResult {
        cameras,
        per_cam_target_poses,
    })
}

/// Initialize intrinsics for all cameras using full auto-init (Zhang's per camera).
///
/// Convenience wrapper around [`step_intrinsics_init_all_with_seed`] with default seeds.
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    step_intrinsics_init_all_with_seed(session, IntrinsicsManualInit::default(), opts)
}

/// Optimize intrinsics for all cameras.
///
/// Refines each camera independently in its own model, holding its `fix`
/// lists, with every pose free. An untilted camera runs a plain planar
/// solve. A Scheimpflug camera runs the staged multi-start tilt solve;
/// after a full auto-init it runs a wider multi-start over
/// linear tilt seeds instead, and a Brown-Conrady tilted camera that still
/// fits badly is re-solved from a nominal seed built from the well-fitted
/// ones.
///
/// Requires [`step_intrinsics_init_all`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Initialization not run, or the camera models changed since
/// - Optimization fails for any camera
pub fn step_intrinsics_optimize_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<IntrinsicsOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        return Err(Error::not_available(
            "per-camera intrinsics (call step_intrinsics_init_all first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let cameras = session
        .state
        .cameras
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    check_camera_models(&cameras, |i| config.camera_for(i).model)?;
    let mut per_cam_target_poses = session
        .state
        .per_cam_target_poses
        .clone()
        .ok_or_else(|| Error::not_available("per-camera target poses"))?;

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..config.solver.backend_options()
    };
    let auto_init = session.state.per_cam_intrinsics_auto;

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);
    let mut scheimpflug_work_items = Vec::new();

    for (cam_idx, camera) in cameras.iter().enumerate() {
        let camera_config = config.camera_for(cam_idx);
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
        let initial_params =
            PlanarIntrinsicsParams::new(camera.clone(), initial_poses).map_err(|e| {
                Error::numerical(format!("failed to build params for camera {cam_idx}: {e}"))
            })?;

        let tilted = camera_config.model.sensor == SensorKind::Scheimpflug;
        let result = if tilted && auto_init {
            let solve_opts = camera_solve_opts(&camera_config, &config.solver);
            if camera_config.model == TILTED_BROWN_CONRADY {
                scheimpflug_work_items.push(ScheimpflugCameraWork {
                    cam_idx,
                    dataset: planar_dataset.clone(),
                    valid_indices: valid_indices.clone(),
                    initial: initial_params.clone(),
                    solve_opts: solve_opts.clone(),
                });
            }
            optimize_scheimpflug_intrinsics_auto_multistart(
                cam_idx,
                &planar_dataset,
                &initial_params,
                solve_opts,
                backend_opts.clone(),
                config.intrinsics.iterations,
                config.intrinsics.fix_k3,
            )?
        } else {
            optimize_camera(
                cam_idx,
                &planar_dataset,
                &initial_params,
                &camera_config,
                &config.solver,
                backend_opts.clone(),
            )?
        };

        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            per_cam_target_poses[global_idx][cam_idx] =
                Some(result.params.camera_se3_target[local_idx]);
        }
        optimized_cameras.push(result.params.camera);
        per_cam_reproj_errors.push(result.mean_reproj_error);
    }

    if !scheimpflug_work_items.is_empty() {
        recover_bad_scheimpflug_cameras_from_nominal(
            &scheimpflug_work_items,
            &mut optimized_cameras,
            &mut per_cam_reproj_errors,
            &mut per_cam_target_poses,
            backend_opts,
            config.rig.reference_camera_idx,
        )?;
    }

    // Fail fast on any diverged camera before its garbage intrinsics poison the
    // linear rig init and the joint rig + hand-eye bundle adjustment.
    guard_percam_reproj_errors(&per_cam_reproj_errors)?;

    session.state.cameras = Some(optimized_cameras.clone());
    session.state.per_cam_target_poses = Some(per_cam_target_poses.clone());
    session.state.per_cam_reproj_errors = Some(per_cam_reproj_errors.clone());

    let avg_error: f64 =
        per_cam_reproj_errors.iter().sum::<f64>() / per_cam_reproj_errors.len() as f64;
    session.log_success_with_notes(
        "intrinsics_optimize_all",
        format!("avg_reproj_err={avg_error:.3}px"),
    );

    Ok(IntrinsicsOptimizeResult {
        cameras: optimized_cameras,
        per_cam_reproj_errors,
        per_cam_target_poses,
    })
}

#[allow(clippy::too_many_arguments)]
fn optimize_scheimpflug_intrinsics_auto_multistart(
    cam_idx: usize,
    dataset: &PlanarDataset,
    initial: &PlanarIntrinsicsParams,
    mut solve_opts: PlanarIntrinsicsSolveOptions,
    backend_opts: BackendSolveOptions,
    init_iterations: usize,
    fix_k3_in_init: bool,
) -> Result<PlanarIntrinsicsEstimate, Error> {
    solve_opts.fix_poses.clear();
    let distortion_kind = initial
        .camera
        .spec()
        .map_or(DistortionKind::BrownConrady5, |s| s.distortion);
    let mut best: Option<PlanarIntrinsicsEstimate> = None;
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
        // Each linear seed starts from zero distortion in the camera's model.
        let params = PlanarIntrinsicsParams::new(
            CameraParams {
                distortion: embed_distortion(
                    &BrownConrady5 {
                        iters: 8,
                        ..BrownConrady5::default()
                    },
                    distortion_kind,
                ),
                sensor: SensorParams::Scheimpflug {
                    params: init.sensor,
                },
                intrinsics: init.camera.k,
                ..initial.camera.clone()
            },
            init.poses,
        )?;
        if !seeds.iter().any(|(s, _)| {
            (rig_tilt(&s.camera).tilt_x - rig_tilt(&params.camera).tilt_x).abs() < 1e-4
                && (rig_tilt(&s.camera).tilt_y - rig_tilt(&params.camera).tilt_y).abs() < 1e-4
                && (s.camera.intrinsics.fx - params.camera.intrinsics.fx).abs() < 1.0
                && (s.camera.intrinsics.fy - params.camera.intrinsics.fy).abs() < 1.0
        }) {
            seeds.push((params, false));
        }
    }

    for (params, _) in seeds {
        let staged = ScheimpflugStagedInitOptions {
            tilt_x_seeds: vec![rig_tilt(&params.camera).tilt_x],
            tilt_y_seeds: vec![rig_tilt(&params.camera).tilt_y],
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

/// Re-solve each badly fitted tilted camera from a nominal seed built from
/// the well-fitted ones (and, failing that, against their rig). The
/// recovery is a Brown-Conrady heuristic: it serves cameras of
/// [`TILTED_BROWN_CONRADY`] only.
fn recover_bad_scheimpflug_cameras_from_nominal(
    work_items: &[ScheimpflugCameraWork],
    cameras: &mut [CameraParams],
    per_cam_reproj_errors: &mut [f64],
    per_cam_target_poses: &mut [Vec<Option<Iso3>>],
    backend_opts: BackendSolveOptions,
    reference_camera_idx: usize,
) -> Result<(), Error> {
    let good_indices = good_nominal_scheimpflug_indices(cameras, per_cam_reproj_errors);
    if good_indices.len() < 2 {
        return Ok(());
    };
    let Some(nominal) = build_nominal_scheimpflug_seed(cameras, &good_indices) else {
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
            per_cam_target_poses.len(),
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
            cameras[cam_idx] = candidate.params.camera.clone();
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
    cameras: &[CameraParams],
    good_indices: &[usize],
) -> Option<NominalScheimpflugSeed> {
    let k = |idx: usize| cameras[idx].intrinsics;
    let dist = |idx: usize| rig_distortion(&cameras[idx]);
    let tilt = |idx: usize| rig_tilt(&cameras[idx]);
    let intrinsics = FxFyCxCy {
        fx: median_of(good_indices.iter().map(|&idx| k(idx).fx))?,
        fy: median_of(good_indices.iter().map(|&idx| k(idx).fy))?,
        cx: median_of(good_indices.iter().map(|&idx| k(idx).cx))?,
        cy: median_of(good_indices.iter().map(|&idx| k(idx).cy))?,
    };
    let distortion = BrownConrady5 {
        k1: median_of(good_indices.iter().map(|&idx| dist(idx).k1))?,
        k2: median_of(good_indices.iter().map(|&idx| dist(idx).k2))?,
        k3: median_of(good_indices.iter().map(|&idx| dist(idx).k3))?,
        p1: median_of(good_indices.iter().map(|&idx| dist(idx).p1))?,
        p2: median_of(good_indices.iter().map(|&idx| dist(idx).p2))?,
        iters: dist(good_indices[0]).iters.max(8),
    };
    let sensor = ScheimpflugParams {
        tilt_x: median_of(good_indices.iter().map(|&idx| tilt(idx).tilt_x))?,
        tilt_y: median_of(good_indices.iter().map(|&idx| tilt(idx).tilt_y))?,
    };

    Some(NominalScheimpflugSeed {
        intrinsics,
        distortion,
        sensor,
    })
}

fn good_nominal_scheimpflug_indices(cameras: &[CameraParams], errors: &[f64]) -> Vec<usize> {
    cameras
        .iter()
        .zip(errors.iter())
        .enumerate()
        .filter_map(|(idx, (camera, error))| {
            (camera.spec() == Some(TILTED_BROWN_CONRADY)
                && is_good_nominal_scheimpflug_source(camera, *error))
            .then_some(idx)
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

fn is_good_nominal_scheimpflug_source(camera: &CameraParams, error: f64) -> bool {
    let k = camera.intrinsics;
    let dist = rig_distortion(camera);
    let sensor = rig_tilt(camera);
    error.is_finite()
        && error <= SCHEIMPFLUG_NOMINAL_GOOD_REPROJ_PX
        && k.fx.is_finite()
        && k.fy.is_finite()
        && k.cx.is_finite()
        && k.cy.is_finite()
        && (100.0..10_000.0).contains(&k.fx)
        && (100.0..10_000.0).contains(&k.fy)
        && dist.k1.is_finite()
        && dist.k2.is_finite()
        && dist.k3.is_finite()
        && dist.p1.is_finite()
        && dist.p2.is_finite()
        && dist.k1.abs() < 2.0
        && dist.k2.abs() < 2.0
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

fn recover_one_scheimpflug_camera_from_nominal(
    item: &ScheimpflugCameraWork,
    nominal: NominalScheimpflugSeed,
    rig_context: Option<&GoodRigContext>,
    cameras: &[CameraParams],
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<Option<PlanarIntrinsicsEstimate>, Error> {
    let solve_opts = &item.solve_opts;
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
        let candidate = optimize_planar_intrinsics(
            &item.dataset,
            &scored_seed.params,
            nominal_pose_only_solve_opts(solve_opts.clone()),
            BackendSolveOptions {
                max_iters: backend_opts.max_iters.max(60),
                ..backend_opts.clone()
            },
        )
        .unwrap_or_else(|_| scheimpflug_estimate_from_scored_seed(scored_seed));

        let candidate = optimize_planar_intrinsics(
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
                num_views,
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
    params: &PlanarIntrinsicsParams,
    context: &GoodRigContext,
) -> Option<f64> {
    let cam_to_rig =
        estimate_candidate_cam_to_rig(params, &item.valid_indices, &context.rig_from_target)?;
    let cam_from_rig = cam_to_rig.inverse();
    let Ok(camera) = params.camera.build() else {
        return None;
    };
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

fn optimize_bad_scheimpflug_camera_against_good_rig(
    item: &ScheimpflugCameraWork,
    candidate_params: &PlanarIntrinsicsParams,
    context: &GoodRigContext,
    cameras: &[CameraParams],
    num_views: usize,
    backend_opts: &BackendSolveOptions,
) -> Result<Option<PlanarIntrinsicsEstimate>, Error> {
    let Some(initial_cam_to_rig) = estimate_candidate_cam_to_rig(
        candidate_params,
        &item.valid_indices,
        &context.rig_from_target,
    ) else {
        return Ok(None);
    };
    let dataset = bad_camera_only_scheimpflug_rig_dataset(item, cameras.len(), num_views)?;

    let mut rig_cameras = cameras.to_vec();
    rig_cameras[item.cam_idx] = candidate_params.camera.clone();
    let mut cam_to_rig = context.cam_to_rig.clone();
    if cam_to_rig.len() != cameras.len() {
        return Ok(None);
    }
    cam_to_rig[item.cam_idx] = initial_cam_to_rig;

    let mut fix_extrinsics = vec![true; cameras.len()];
    fix_extrinsics[item.cam_idx] = false;
    let fix = rig_cameras
        .iter()
        .map(|c| c.spec().map(CameraFix::all))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::invalid_input("a rig camera has a general homography sensor"))?;
    let initial = RigExtrinsicsParams {
        cameras: rig_cameras,
        cam_to_rig,
        rig_from_target: context.rig_from_target.clone(),
    };
    let solve_opts = RigExtrinsicsSolveOptions {
        robust_loss: item.solve_opts.robust_loss,
        fix,
        fix_extrinsics,
        fix_rig_poses: (0..num_views).collect(),
    };
    let Ok(result) = optimize_rig_extrinsics(
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
    let cam_from_rig = result.params.cam_to_rig[item.cam_idx].inverse();
    let mut poses = Vec::with_capacity(item.valid_indices.len());
    for &global_idx in &item.valid_indices {
        if global_idx >= result.params.rig_from_target.len() {
            return Ok(None);
        }
        poses.push(cam_from_rig * result.params.rig_from_target[global_idx]);
    }
    let params = PlanarIntrinsicsParams::new(camera.clone(), poses)?;
    Ok(Some(PlanarIntrinsicsEstimate {
        params,
        report: result.report,
        mean_reproj_error: mean,
        // Derived from the rig solve, not solved on its own.
        uncertainty: Uncertainty::default(),
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
    params: &PlanarIntrinsicsParams,
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
                let mut dist = rig_distortion(&base.params.camera);
                dist.k1 += dk1;
                dist.k2 += dk2;
                score_scheimpflug_seed_parts(
                    dataset,
                    base.params.camera.intrinsics,
                    dist,
                    rig_tilt(&base.params.camera),
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
    params: &PlanarIntrinsicsParams,
    scored: &mut Vec<ScoredScheimpflugSeed>,
) {
    let dist = rig_distortion(&params.camera);
    if let Ok(poses) = recover_scheimpflug_poses_for_seed(
        dataset,
        &params.camera.intrinsics,
        &dist,
        rig_tilt(&params.camera),
    ) && let Ok(params) = PlanarIntrinsicsParams::new(params.camera.clone(), poses)
    {
        let score = mean_scheimpflug_reproj(dataset, &params);
        if score.is_finite() {
            scored.push(ScoredScheimpflugSeed { score, params });
        }
    }
}

fn score_scheimpflug_seed_parts(
    dataset: &PlanarDataset,
    intrinsics: FxFyCxCy<f64>,
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
    let Ok(params) = tilted_planar_params(intrinsics, distortion, sensor, poses) else {
        return;
    };
    let score = mean_scheimpflug_reproj(dataset, &params);
    if score.is_finite() {
        scored.push(ScoredScheimpflugSeed { score, params });
    }
}

fn nominal_pose_only_solve_opts(
    mut opts: PlanarIntrinsicsSolveOptions,
) -> PlanarIntrinsicsSolveOptions {
    opts.fix = CameraFix::all(TILTED_BROWN_CONRADY);
    opts.fix_poses.clear();
    opts.bounds = None;
    opts
}

fn nominal_tight_refine_solve_opts(
    mut opts: PlanarIntrinsicsSolveOptions,
    nominal: NominalScheimpflugSeed,
) -> PlanarIntrinsicsSolveOptions {
    let k = nominal.intrinsics;
    opts.fix.intrinsics = Some(Vec::new());
    opts.fix_poses.clear();
    opts.bounds = Some(CameraBounds {
        fx: Some((0.94 * k.fx, 1.06 * k.fx)),
        fy: Some((0.94 * k.fy, 1.06 * k.fy)),
        cx: Some((k.cx - 60.0, k.cx + 60.0)),
        cy: Some((k.cy - 60.0, k.cy + 60.0)),
        tilt_x: Some((nominal.sensor.tilt_x - 0.06, nominal.sensor.tilt_x + 0.06)),
        tilt_y: Some((nominal.sensor.tilt_y - 0.04, nominal.sensor.tilt_y + 0.04)),
    });
    opts
}

fn scheimpflug_estimate_from_scored_seed(seed: ScoredScheimpflugSeed) -> PlanarIntrinsicsEstimate {
    PlanarIntrinsicsEstimate {
        params: seed.params,
        report: SolveReport {
            final_cost: seed.score * seed.score,
            num_iters: 0,
        },
        mean_reproj_error: seed.score,
        uncertainty: Uncertainty::default(),
    }
}

fn mean_scheimpflug_reproj(dataset: &PlanarDataset, params: &PlanarIntrinsicsParams) -> f64 {
    let Ok(camera) = params.camera.build() else {
        return f64::INFINITY;
    };
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
    intrinsics: &FxFyCxCy<f64>,
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
    manual: RigManualInit,
    opts: Option<RigInitOptions>,
) -> Result<RigInitResult, Error> {
    let RigInitOptions {} = opts.unwrap_or_default();
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
            let msg = "RigManualInit: cam_se3_rig and rig_se3_target must both be Some \
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

    Ok(RigInitResult {
        initial_cam_se3_rig: cam_se3_rig,
        initial_rig_se3_target: rig_se3_target,
    })
}

/// Initialize rig extrinsics using full auto-init (linear extrinsics fit).
///
/// Convenience wrapper around [`step_rig_init_with_seed`] with default seeds.
pub fn step_rig_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<RigInitOptions>,
) -> Result<RigInitResult, Error> {
    step_rig_init_with_seed(session, RigManualInit::default(), opts)
}

/// Optimize rig extrinsics using bundle adjustment.
///
/// Jointly optimizes:
/// - Per-camera extrinsics (camera-to-rig)
/// - Per-view rig poses (rig-to-target)
/// - Optionally: per-camera intrinsics (if `refine_intrinsics` is true)
///
/// The reference camera (`rig.reference_camera_idx`) alone removes the rig's
/// full 6-DOF gauge freedom; no rig-from-target pose is fixed; an extra
/// per-view pose constraint would be redundant and mildly pessimizing.
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
) -> Result<RigOptimizeResult, Error> {
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
        .cameras
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    check_camera_models(&cameras, |i| config.camera_for(i).model)?;
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

    let fix = (0..input.num_cameras)
        .map(|i| {
            rig_ba_fix(
                &config.camera_for(i),
                config.rig.refine_intrinsics,
                config.rig.refine_sensor,
            )
        })
        .collect();

    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == config.rig.reference_camera_idx)
        .collect();

    // The reference-camera fix above removes the full 6-DOF rig gauge on
    // its own; pinning a rig-from-target pose as well is redundant and
    // slightly worsens the fit (`docs/notes/rig-extrinsics.md` §Gauge).
    let fix_rig_poses: Vec<usize> = Vec::new();

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..config.solver.backend_options()
    };

    // The rig BA takes a dataset without robot metadata.
    let rig_input_no_meta: RigDataset<NoMeta> = RigDataset::new(
        input
            .views
            .iter()
            .map(|v| RigView {
                meta: NoMeta,
                obs: v.obs.clone(),
            })
            .collect(),
        input.num_cameras,
    )?;

    let initial = RigExtrinsicsParams {
        cameras,
        cam_to_rig,
        rig_from_target,
    };
    let solve_opts = RigExtrinsicsSolveOptions {
        robust_loss: config.solver.robust_loss,
        fix,
        fix_extrinsics,
        fix_rig_poses,
    };
    let opt = match optimize_rig_extrinsics(rig_input_no_meta, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("rig_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    let cam_se3_rig: Vec<Iso3> = opt.params.cam_to_rig.iter().map(|t| t.inverse()).collect();
    let mean_reproj_error = opt.mean_reproj_error;
    let per_cam_reproj_errors = opt.per_cam_reproj_errors.clone();
    let final_cost = opt.report.final_cost;
    session.state.rig_ba_cam_se3_rig = Some(cam_se3_rig);
    session.state.rig_ba_rig_se3_target = Some(opt.params.rig_from_target);
    session.state.rig_ba_reproj_error = Some(mean_reproj_error);
    session.state.rig_ba_per_cam_reproj_errors = Some(opt.per_cam_reproj_errors);
    session.state.cameras = Some(opt.params.cameras);

    session.log_success_with_notes(
        "rig_optimize",
        format!("final_cost={final_cost:.2e}, mean_reproj_err={mean_reproj_error:.3}px"),
    );

    Ok(RigOptimizeResult {
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
    manual: HandeyeManualInit,
    opts: Option<HandeyeInitOptions>,
) -> Result<HandeyeInitResult, Error> {
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
                HandeyeMode::EyeInHand => {
                    let target_se3_rig: Vec<Iso3> =
                        rig_se3_target.iter().map(|t| t.inverse()).collect();
                    estimate_handeye_dlt(&robot_poses, &target_se3_rig, min_angle)
                        .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))
                        .map_err(|e| {
                            Error::numerical(format!("linear hand-eye estimation failed: {e}"))
                        })?
                }
                HandeyeMode::EyeToHand => {
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
                HandeyeMode::EyeInHand => robot_poses[0] * handeye * rig_se3_target[0],
                HandeyeMode::EyeToHand => {
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

    Ok(HandeyeInitResult {
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
) -> Result<HandeyeInitResult, Error> {
    step_handeye_init_with_seed(session, HandeyeManualInit::default(), opts)
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
) -> Result<HandeyeOptimizeResult, Error> {
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
        .cameras
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    check_camera_models(&cameras, |i| config.camera_for(i).model)?;
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

    let fix = (0..input.num_cameras)
        .map(|i| {
            rig_ba_fix(
                &config.camera_for(i),
                config.handeye_optimize.refine_intrinsics,
                config.handeye_optimize.refine_sensor,
            )
        })
        .collect();

    let fix_extrinsics: Vec<bool> = if config.handeye_optimize.refine_cam_se3_rig {
        (0..input.num_cameras)
            .map(|i| i == config.rig.reference_camera_idx)
            .collect()
    } else {
        vec![true; input.num_cameras]
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.solver.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.solver.verbosity),
        ..config.solver.backend_options()
    };

    let initial = HandeyeParams {
        cameras,
        cam_to_rig,
        handeye,
        target_poses: vec![mode_target_pose],
    };
    let solve_opts = HandeyeSolveOptions {
        robust_loss: config.solver.robust_loss,
        fix,
        fix_extrinsics,
        fix_handeye: false,
        fix_target_poses: Vec::new(),
        relax_target_poses: false,
        refine_robot_poses: config.handeye_optimize.robot_poses.refine,
        robot_rot_sigma: config.handeye_optimize.robot_poses.rot_sigma,
        robot_trans_sigma: config.handeye_optimize.robot_poses.trans_sigma,
    };
    let handeye_dataset = HandeyeDataset::new(
        input.views.clone(),
        input.num_cameras,
        config.handeye_init.handeye_mode,
    )?;
    let output = match optimize_handeye(handeye_dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    let final_cost = output.report.final_cost;
    let mean_reproj_error = output.mean_reproj_error;
    session.state.final_cost = Some(final_cost);
    session.state.final_reproj_error = Some(mean_reproj_error);

    session.set_output(output);

    session.log_success_with_notes("handeye_optimize", format!("final_cost={final_cost:.2e}"));

    Ok(HandeyeOptimizeResult {
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
    let manual = session.config.manual_init.clone().unwrap_or_default();
    let _ = step_intrinsics_init_all_with_seed(session, manual, None)?;
    let _ = step_intrinsics_optimize_all(session, None)?;
    let _ = step_rig_init(session, None)?;
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
        BrownConrady5, CorrespondenceView, FxFyCxCy, PinholeCamera, Pt2, Pt3, RigDataset, RigView,
        RigViewObs, make_pinhole_camera,
    };
    use vision_calibration_optim::RobotPoseMeta;

    fn make_test_camera(offset: f64) -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0 + offset,
                fy: 780.0 + offset,
                cx: 640.0,
                cy: 360.0,
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
        make_noisy_test_input(0.0)
    }

    /// The test scene with each pixel moved by `±noise_px` in a fixed
    /// pattern, so the solve keeps a nonzero residual.
    fn make_noisy_test_input(noise_px: f64) -> RigHandeyeInput {
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
                        .enumerate()
                        .map(|(k, p)| {
                            let p_cam = pose.transform_point(p);
                            let px = cam.project_point_c(&p_cam.coords).unwrap();
                            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                            Pt2::new(px.x + sign * noise_px, px.y - sign * noise_px)
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
        let cameras = session.state.cameras.as_ref().unwrap();
        assert_eq!(cameras.len(), 2);
    }

    #[test]
    fn step_rig_init_requires_intrinsics() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();

        let result = step_rig_init(&mut session, None);
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
        step_rig_init(&mut session, None).unwrap();

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
        step_rig_init(&mut session, None).unwrap();
        step_rig_optimize(&mut session, None).unwrap();

        let result = step_handeye_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hand-eye"));
    }

    #[test]
    fn step_handeye_optimize_reports_the_mean_reprojection_error() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_noisy_test_input(0.3)).unwrap();
        let manual = session.config.manual_init.clone().unwrap_or_default();
        step_intrinsics_init_all_with_seed(&mut session, manual, None).unwrap();
        step_intrinsics_optimize_all(&mut session, None).unwrap();
        step_rig_init(&mut session, None).unwrap();
        step_rig_optimize(&mut session, None).unwrap();
        step_handeye_init(&mut session, None).unwrap();
        let result = step_handeye_optimize(&mut session, None).unwrap();

        let export = session.export().unwrap();
        assert!(
            (result.mean_reproj_error - export.mean_reproj_error).abs() < 1e-12,
            "step reports {} px, export {} px",
            result.mean_reproj_error,
            export.mean_reproj_error
        );
        // The pattern moves every pixel by 0.3 px per axis, about 0.42 px.
        assert!(result.mean_reproj_error > 0.05 && result.mean_reproj_error < 1.0);
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
                solver: crate::common::config::SolverConfig {
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
                manual_init: Some(IntrinsicsManualInit {
                    per_cam_intrinsics: Some(vec![
                        FxFyCxCy {
                            fx: 800.0,
                            fy: 780.0,
                            cx: 640.0,
                            cy: 360.0,
                        },
                        FxFyCxCy {
                            fx: 810.0,
                            fy: 790.0,
                            cx: 640.0,
                            cy: 360.0,
                        },
                    ]),
                    per_cam_distortion: Some(vec![
                        BrownConrady5::default(),
                        BrownConrady5::default(),
                    ]),
                    per_cam_sensors: None,
                }),
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
                handeye_init: crate::common::config::HandeyeInitConfig {
                    handeye_mode: HandeyeMode::EyeToHand,
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
        let manual = HandeyeManualInit {
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
