//! Step functions for multi-camera rig hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigHandeyeProblem>` to perform calibration.

use anyhow::{Context, Result};
use calib_core::{make_pinhole_camera, CameraFixMask, Iso3, NoMeta, PlanarDataset, View};
use calib_linear::estimate_extrinsics_from_cam_target_poses;
use calib_linear::estimate_handeye_dlt;
use calib_linear::prelude::*;
use calib_optim::{
    optimize_handeye, optimize_planar_intrinsics, optimize_rig_extrinsics, BackendSolveOptions,
    HandEyeDataset, HandEyeParams, HandEyeSolveOptions, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RigExtrinsicsParams, RigExtrinsicsSolveOptions,
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
pub struct IntrinsicsOptimOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Options for rig BA optimization.
#[derive(Debug, Clone, Default)]
pub struct RigOptimOptions {
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

/// Options for hand-eye optimization.
#[derive(Debug, Clone, Default)]
pub struct HandeyeOptimOptions {
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
fn views_to_planar_dataset(views: &[Option<View<NoMeta>>]) -> Result<(PlanarDataset, Vec<usize>)> {
    let (valid_views, indices): (Vec<_>, Vec<_>) = views
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.as_ref().map(|view| (view.clone(), i)))
        .unzip();

    if valid_views.len() < 3 {
        anyhow::bail!(
            "need at least 3 views for intrinsics calibration, got {}",
            valid_views.len()
        );
    }

    let dataset = PlanarDataset::new(valid_views)?;
    Ok((dataset, indices))
}

/// Estimate initial target pose from camera intrinsics and view observations.
fn estimate_target_pose(
    k_matrix: &calib_core::Mat3,
    obs: &calib_core::CorrespondenceView,
) -> Result<Iso3> {
    let board_2d: Vec<calib_core::Pt2> = obs
        .points_3d
        .iter()
        .map(|p| calib_core::Pt2::new(p.x, p.y))
        .collect();
    let pixel_2d: Vec<calib_core::Pt2> = obs
        .points_2d
        .iter()
        .map(|v| calib_core::Pt2::new(v.x, v.y))
        .collect();

    let h = dlt_homography(&board_2d, &pixel_2d).context("failed to compute homography")?;
    estimate_planar_pose_from_h(k_matrix, &h).context("failed to recover pose from homography")
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
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<()> {
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

    // Copy values we need to avoid borrow conflicts
    let num_cameras = input.num_cameras;
    let num_views = input.num_views();

    let mut per_cam_intrinsics = Vec::with_capacity(num_cameras);
    let mut per_cam_target_poses: Vec<Vec<Option<Iso3>>> = vec![vec![None; num_cameras]; num_views];

    for (cam_idx, item) in per_cam_target_poses
        .iter_mut()
        .enumerate()
        .take(num_cameras)
    {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views)
            .with_context(|| format!("camera {} has insufficient views", cam_idx))?;

        // Estimate intrinsics
        let camera = estimate_intrinsics_iterative(&planar_dataset, init_opts)
            .with_context(|| format!("intrinsics estimation failed for camera {}", cam_idx))?;

        // Compute K matrix for pose estimation
        let k_matrix = calib_core::Mat3::new(
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

        // Estimate target poses for valid views
        for (local_idx, &global_idx) in valid_indices.iter().enumerate() {
            let view = &planar_dataset.views[local_idx];
            let pose = estimate_target_pose(&k_matrix, &view.obs).with_context(|| {
                format!(
                    "pose estimation failed for cam {} view {}",
                    cam_idx, global_idx
                )
            })?;
            item[cam_idx] = Some(pose);
        }

        per_cam_intrinsics.push(make_pinhole_camera(camera.k, camera.dist));
    }

    // Update state
    session.state.per_cam_intrinsics = Some(per_cam_intrinsics);
    session.state.per_cam_target_poses = Some(per_cam_target_poses);

    session.log_success_with_notes(
        "intrinsics_init_all",
        format!("initialized {} cameras", num_cameras),
    );

    Ok(())
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
    opts: Option<IntrinsicsOptimOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        anyhow::bail!("intrinsics initialization not run - call step_intrinsics_init_all first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let per_cam_intrinsics = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial intrinsics"))?;
    let mut per_cam_target_poses = session
        .state
        .per_cam_target_poses
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial target poses"))?;

    let mut optimized_cameras = Vec::with_capacity(input.num_cameras);
    let mut per_cam_reproj_errors = Vec::with_capacity(input.num_cameras);

    for cam_idx in 0..input.num_cameras {
        let cam_views = extract_camera_views(input, cam_idx);
        let (planar_dataset, valid_indices) = views_to_planar_dataset(&cam_views)
            .with_context(|| format!("camera {} has insufficient views", cam_idx))?;

        // Get initial poses for this camera
        let initial_poses: Vec<Iso3> = valid_indices
            .iter()
            .map(|&global_idx| {
                per_cam_target_poses[global_idx][cam_idx]
                    .ok_or_else(|| anyhow::anyhow!("missing initial pose"))
            })
            .collect::<Result<Vec<_>>>()?;

        // Build params
        let initial_params =
            PlanarIntrinsicsParams::new(per_cam_intrinsics[cam_idx].clone(), initial_poses)
                .with_context(|| format!("failed to build params for camera {}", cam_idx))?;

        // Optimize
        let solve_opts = PlanarIntrinsicsSolveOptions {
            robust_loss: config.robust_loss,
            fix_intrinsics: Default::default(),
            fix_distortion: Default::default(),
            fix_poses: Vec::new(),
        };

        let backend_opts = BackendSolveOptions {
            max_iters: opts.max_iters.unwrap_or(config.max_iters),
            verbosity: opts.verbosity.unwrap_or(config.verbosity),
            ..Default::default()
        };

        let result =
            optimize_planar_intrinsics(&planar_dataset, &initial_params, solve_opts, backend_opts)
                .with_context(|| format!("optimization failed for camera {}", cam_idx))?;

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
pub fn step_rig_init(session: &mut CalibrationSession<RigHandeyeProblem>) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_per_cam_intrinsics() {
        anyhow::bail!(
            "per-camera intrinsics not computed - call step_intrinsics_optimize_all first"
        );
    }

    // Copy values we need before modifying state
    let num_views = input.num_views();
    let reference_camera_idx = session.config.reference_camera_idx;

    let per_cam_target_poses = session
        .state
        .per_cam_target_poses
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no per-camera target poses"))?;

    // Use calib_linear's rig extrinsics initialization
    let extrinsic_result =
        estimate_extrinsics_from_cam_target_poses(&per_cam_target_poses, reference_camera_idx)
            .context("rig extrinsics initialization failed")?;

    // Update state
    session.state.initial_cam_se3_rig = Some(extrinsic_result.cam_to_rig);
    session.state.initial_rig_se3_target = Some(extrinsic_result.rig_from_target);

    session.log_success_with_notes(
        "rig_init",
        format!("ref_cam={}, {} views", reference_camera_idx, num_views),
    );

    Ok(())
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
    opts: Option<RigOptimOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_rig_init() {
        anyhow::bail!("rig initialization not run - call step_rig_init first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let cameras = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no per-camera intrinsics"))?;
    let cam_to_rig = session
        .state
        .initial_cam_se3_rig
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial cam_se3_rig"))?;
    let rig_from_target = session
        .state
        .initial_rig_se3_target
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial rig_se3_target"))?;

    // Build initial params
    let initial = RigExtrinsicsParams {
        cameras,
        cam_to_rig,
        rig_from_target,
    };

    // Configure solve options
    let fix_intrinsics = if config.refine_intrinsics_in_rig_ba {
        CameraFixMask::default()
    } else {
        CameraFixMask::all_fixed()
    };

    // Reference camera has fixed extrinsics (identity)
    let fix_extrinsics: Vec<bool> = (0..input.num_cameras)
        .map(|i| i == config.reference_camera_idx)
        .collect();

    let fix_rig_poses = if config.fix_first_rig_pose {
        vec![0]
    } else {
        Vec::new()
    };

    let solve_opts = RigExtrinsicsSolveOptions {
        robust_loss: config.robust_loss,
        default_fix: fix_intrinsics,
        camera_overrides: Vec::new(),
        fix_extrinsics,
        fix_rig_poses,
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.verbosity),
        ..Default::default()
    };

    // Convert input to NoMeta format for rig extrinsics optimization
    let rig_input_no_meta: calib_core::RigDataset<NoMeta> = calib_core::RigDataset::new(
        input
            .views
            .iter()
            .map(|v| calib_core::RigView {
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
            return Err(e);
        }
    };

    // Update state with refined rig extrinsics
    session.state.rig_ba_cam_se3_rig = Some(result.params.cam_to_rig.clone());
    session.state.rig_ba_rig_se3_target = Some(result.params.rig_from_target.clone());
    session.state.rig_ba_reproj_error = Some(result.report.final_cost.sqrt());
    // Also update cameras in case intrinsics were refined
    session.state.per_cam_intrinsics = Some(result.params.cameras);

    session.log_success_with_notes(
        "rig_optimize",
        format!("reproj_err={:.3}px", result.report.final_cost.sqrt()),
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
pub fn step_handeye_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_rig_optimized() {
        anyhow::bail!("rig optimization not run - call step_rig_optimize first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let min_angle = opts
        .min_motion_angle_deg
        .unwrap_or(config.min_motion_angle_deg);

    // Get robot poses and rig-to-target poses
    let robot_poses: Vec<Iso3> = input.views.iter().map(|v| v.meta.robot_pose).collect();
    let rig_se3_target = session
        .state
        .rig_ba_rig_se3_target
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no rig_se3_target from rig BA"))?;

    // For hand-eye, we need rig poses (not target poses)
    // rig_se3_target = T_R_T, so target_se3_rig = (T_R_T)^-1
    // But for linear hand-eye we need "camera poses" which are actually rig poses here
    // The linear solver expects: robot_poses (T_B_G) and cam_target_poses (T_C_T)
    // In our case, cam = rig, so we use rig_se3_target directly

    // Linear hand-eye estimation
    // Note: estimate_handeye_dlt expects cam_se3_target poses
    // Since our "camera" is the rig, we use rig_se3_target
    let handeye = match estimate_handeye_dlt(&robot_poses, &rig_se3_target, min_angle) {
        Ok(h) => h,
        Err(e) => {
            session.log_failure("handeye_init", e.to_string());
            return Err(e).context("linear hand-eye estimation failed");
        }
    };

    // Estimate initial target pose in base frame
    // For EyeInHand: T_B_T = T_B_G * T_G_R * T_R_T = robot_pose * handeye * rig_se3_target
    let target_se3_base = match config.handeye_mode {
        calib_optim::HandEyeMode::EyeInHand => {
            // handeye = T_G_R, we need T_B_T = T_B_G * T_G_R * T_R_T
            robot_poses[0] * handeye * rig_se3_target[0]
        }
        calib_optim::HandEyeMode::EyeToHand => {
            // For EyeToHand: handeye = T_R_B (rig in base frame, camera is fixed)
            // T_B_T = T_B_R * T_R_T = handeye^-1 * rig_se3_target
            handeye.inverse() * rig_se3_target[0]
        }
    };

    // Update state
    session.state.initial_handeye = Some(handeye);
    session.state.initial_target_se3_base = Some(target_se3_base);

    session.log_success_with_notes(
        "handeye_init",
        format!("translation_norm={:.4}m", handeye.translation.vector.norm()),
    );

    Ok(())
}

/// Optimize hand-eye calibration using bundle adjustment.
///
/// This step jointly optimizes:
/// - Per-camera intrinsics
/// - Per-camera extrinsics (cam_se3_rig) - optionally
/// - Hand-eye transform (gripper_se3_rig)
/// - Target pose in base frame
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
    opts: Option<HandeyeOptimOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?.clone();

    if !session.state.has_handeye_init() {
        anyhow::bail!("hand-eye initialization not run - call step_handeye_init first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get all initial estimates
    let cameras = session
        .state
        .per_cam_intrinsics
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no per-camera intrinsics"))?;
    let cam_to_rig = session
        .state
        .rig_ba_cam_se3_rig
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no cam_se3_rig from rig BA"))?;
    let handeye = session
        .state
        .initial_handeye
        .ok_or_else(|| anyhow::anyhow!("no initial handeye"))?;
    let target_se3_base = session
        .state
        .initial_target_se3_base
        .ok_or_else(|| anyhow::anyhow!("no initial target pose"))?;

    // Build initial params for hand-eye optimization
    let initial = HandEyeParams {
        cameras,
        cam_to_rig: cam_to_rig.clone(),
        handeye,
        target_poses: vec![target_se3_base], // Single static target
    };

    // Configure solve options
    let fix_intrinsics = CameraFixMask::all_fixed(); // Don't refine intrinsics in final BA

    // Fix cam_se3_rig unless explicitly enabled
    let fix_extrinsics: Vec<bool> = if config.refine_cam_se3_rig_in_handeye_ba {
        // Only fix reference camera
        (0..input.num_cameras)
            .map(|i| i == config.reference_camera_idx)
            .collect()
    } else {
        // Fix all extrinsics
        vec![true; input.num_cameras]
    };

    let solve_opts = HandEyeSolveOptions {
        robust_loss: config.robust_loss,
        default_fix: fix_intrinsics,
        camera_overrides: Vec::new(),
        fix_extrinsics,
        fix_handeye: false,
        fix_target_poses: Vec::new(),
        relax_target_poses: false, // Single fixed target
        refine_robot_poses: config.refine_robot_poses,
        robot_rot_sigma: config.robot_rot_sigma,
        robot_trans_sigma: config.robot_trans_sigma,
    };

    let backend_opts = BackendSolveOptions {
        max_iters: opts.max_iters.unwrap_or(config.max_iters),
        verbosity: opts.verbosity.unwrap_or(config.verbosity),
        ..Default::default()
    };

    // Convert input to HandEyeDataset
    let handeye_dataset =
        HandEyeDataset::new(input.views.clone(), input.num_cameras, config.handeye_mode)?;

    // Run optimization
    let result = match optimize_handeye(handeye_dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(e);
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
pub fn run_calibration(session: &mut CalibrationSession<RigHandeyeProblem>) -> Result<()> {
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
    use calib_core::{
        make_pinhole_camera, BrownConrady5, CorrespondenceView, FxFyCxCySkew, PinholeCamera, Pt2,
        Pt3, RigDataset, RigView, RigViewObs,
    };
    use calib_optim::RobotPoseMeta;
    use nalgebra::{Rotation3, Translation3};

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
                        robot_pose: *robot_pose,
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
                max_iters: 100,
                ..Default::default()
            })
            .unwrap();

        assert!(session.has_output());
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
