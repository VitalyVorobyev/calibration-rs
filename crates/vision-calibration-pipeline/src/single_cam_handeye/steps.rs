//! Step functions for single-camera hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<SingleCamHandeyeProblem>` to perform calibration.
//!
//! # Example
//!
//! ```no_run
//! use vision_calibration_pipeline::session::CalibrationSession;
//! use vision_calibration_pipeline::single_cam_handeye::{
//!     SingleCamHandeyeProblem, step_intrinsics_init, step_intrinsics_optimize,
//!     step_handeye_init, step_handeye_optimize,
//! };
//! # fn main() -> anyhow::Result<()> {
//! # let input = unimplemented!();
//!
//! let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
//! session.set_input(input)?;
//!
//! step_intrinsics_init(&mut session, None)?;
//! step_intrinsics_optimize(&mut session, None)?;
//! step_handeye_init(&mut session, None)?;
//! step_handeye_optimize(&mut session, None)?;
//!
//! let export = session.export()?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result, ensure};
use vision_calibration_core::{
    CameraFixMask, CorrespondenceView, Iso3, NoMeta, PlanarDataset, RigView, RigViewObs, View,
};
use vision_calibration_linear::prelude::*;
use vision_calibration_linear::{estimate_gripper_se3_target_dlt, estimate_handeye_dlt};
use vision_calibration_optim::{
    BackendSolveOptions, HandEyeDataset, HandEyeParams, HandEyeSolveOptions,
    PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions, RobotPoseMeta, optimize_handeye,
    optimize_planar_intrinsics,
};

use crate::session::CalibrationSession;

use super::problem::{SingleCamHandeyeInput, SingleCamHandeyeProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options and Manual Init
// ─────────────────────────────────────────────────────────────────────────────

/// Options for intrinsics initialization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsInitOptions {
    /// Override the number of iterations.
    pub iterations: Option<usize>,
}

/// Options for intrinsics optimization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Options for hand-eye initialization step.
#[derive(Debug, Clone, Default)]
pub struct HandeyeInitOptions {
    /// Override minimum motion angle (degrees).
    pub min_motion_angle_deg: Option<f64>,
}

/// Options for hand-eye optimization step.
#[derive(Debug, Clone, Default)]
pub struct HandeyeOptimizeOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
}

/// Manual initialization seeds for the intrinsics stage of hand-eye calibration.
///
/// Each field is `Option<T>`. A `None` field is auto-initialized.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IntrinsicsManualInit {
    /// Manually provided initial camera (intrinsics + distortion). If `None`: auto-initialized
    /// via Zhang's method.
    pub camera: Option<vision_calibration_core::PinholeCamera>,
    /// Manually provided per-view target poses (`cam_se3_target`). If `None`: auto-recovered
    /// from homographies using the chosen camera's intrinsics.
    /// Must match the number of views if provided.
    pub target_poses: Option<Vec<Iso3>>,
}

/// Manual initialization seeds for the hand-eye stage of hand-eye calibration.
///
/// Each field is `Option<T>`. A `None` field is auto-initialized.
///
/// The semantics of each field depend on the configured hand-eye mode:
/// - `EyeInHand`: `handeye` is `gripper_se3_camera`; `mode_target_pose` is `base_se3_target`.
/// - `EyeToHand`: `handeye` is `camera_se3_base`; `mode_target_pose` is `gripper_se3_target`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct HandeyeManualInit {
    /// Manually provided hand-eye transform (mode-dependent). If `None`: auto-initialized
    /// via Tsai-Lenz linear estimation.
    pub handeye: Option<Iso3>,
    /// Manually provided fixed target pose (mode-dependent). If `None`: derived from
    /// the handeye transform and the first view using the optimized camera poses.
    /// Requires intrinsics optimization to have been run.
    pub mode_target_pose: Option<Iso3>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert single-cam input to PlanarDataset for intrinsics calibration.
fn input_to_planar_dataset(input: &SingleCamHandeyeInput) -> Result<PlanarDataset> {
    let views: Vec<View<NoMeta>> = input
        .views
        .iter()
        .map(|v| View::without_meta(v.obs.clone()))
        .collect();
    PlanarDataset::new(views)
}

/// Estimate initial target pose from camera intrinsics and view observations.
fn estimate_target_pose(
    k_matrix: &vision_calibration_core::Mat3,
    view: &CorrespondenceView,
) -> Result<Iso3> {
    // Compute homography
    let board_2d: Vec<vision_calibration_core::Pt2> = view
        .points_3d
        .iter()
        .map(|p| vision_calibration_core::Pt2::new(p.x, p.y))
        .collect();
    let pixel_2d: Vec<vision_calibration_core::Pt2> = view
        .points_2d
        .iter()
        .map(|v| vision_calibration_core::Pt2::new(v.x, v.y))
        .collect();

    let h = dlt_homography(&board_2d, &pixel_2d).context("failed to compute homography")?;

    // Recover pose from homography
    estimate_planar_pose_from_h(k_matrix, &h).context("failed to recover pose from homography")
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Seed intrinsics and/or target poses with known values, auto-initializing the rest.
///
/// This step is an expert alternative to [`step_intrinsics_init`].
///
/// # Errors
///
/// - Input not set
/// - `manual.target_poses` length does not match the number of views
/// - Auto-initialization fails
pub fn step_set_intrinsics_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let num_views = input.views.len();
    // Capture before consuming
    let had_camera = manual.camera.is_some();
    let had_target_poses = manual.target_poses.is_some();

    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(config.intrinsics_init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: config.fix_k3,
            fix_tangential: config.fix_tangential,
            iters: 8,
        },
        zero_skew: config.zero_skew,
    };

    let need_camera = manual.camera.is_none();
    let need_poses = manual.target_poses.is_none();

    let (auto_camera, auto_poses) = if need_camera {
        let planar_dataset = input_to_planar_dataset(input)?;
        let camera = match estimate_intrinsics_iterative(&planar_dataset, init_opts) {
            Ok(c) => c,
            Err(e) => {
                session.log_failure("intrinsics_init", e.to_string());
                return Err(e);
            }
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
        let poses: Vec<Iso3> = input
            .views
            .iter()
            .enumerate()
            .map(|(idx, view)| {
                estimate_target_pose(&k_matrix, &view.obs)
                    .with_context(|| format!("failed to estimate pose for view {}", idx))
            })
            .collect::<Result<Vec<_>>>()?;
        (Some(camera), Some(poses))
    } else if need_poses {
        // Manual camera provided but poses needed: recover from homographies.
        let cam = manual.camera.as_ref().unwrap();
        let k_matrix = vision_calibration_core::Mat3::new(
            cam.k.fx, cam.k.skew, cam.k.cx, 0.0, cam.k.fy, cam.k.cy, 0.0, 0.0, 1.0,
        );
        let poses: Vec<Iso3> = input
            .views
            .iter()
            .enumerate()
            .map(|(idx, view)| {
                estimate_target_pose(&k_matrix, &view.obs)
                    .with_context(|| format!("failed to estimate pose for view {}", idx))
            })
            .collect::<Result<Vec<_>>>()?;
        (None, Some(poses))
    } else {
        (None, None)
    };

    let camera = manual.camera.or(auto_camera).expect("camera must be Some");

    let target_poses = if let Some(p) = manual.target_poses {
        ensure!(
            p.len() == num_views,
            "manual target_poses length ({}) does not match view count ({})",
            p.len(),
            num_views
        );
        p
    } else {
        auto_poses.expect("auto_poses must be Some")
    };

    let mut manual_fields: Vec<&str> = Vec::new();
    if had_camera {
        manual_fields.push("camera");
    }
    if had_target_poses {
        manual_fields.push("target_poses");
    }
    let log_note = if manual_fields.is_empty() {
        format!("fx={:.1}, fy={:.1} (auto)", camera.k.fx, camera.k.fy)
    } else {
        format!(
            "fx={:.1}, fy={:.1} (manual: {})",
            camera.k.fx,
            camera.k.fy,
            manual_fields.join(", ")
        )
    };

    session.state.initial_camera = Some(camera);
    session.state.initial_target_poses = Some(target_poses);

    session.log_success_with_notes("intrinsics_init", log_note);

    Ok(())
}

/// Initialize intrinsics from observations.
///
/// This step computes:
/// 1. Initial intrinsics using Zhang's method with iterative distortion estimation
/// 2. Initial target poses from homographies and estimated intrinsics
///
/// To use prior knowledge about the camera, call [`step_set_intrinsics_init`] instead.
///
/// # Errors
///
/// - Input not set
/// - Fewer than 3 views
/// - Intrinsics estimation fails
pub fn step_intrinsics_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<()> {
    step_set_intrinsics_init(session, IntrinsicsManualInit::default(), opts)
}

/// Optimize intrinsics using non-linear least squares.
///
/// This step refines the initial intrinsics estimates by minimizing reprojection error.
///
/// Requires [`step_intrinsics_init`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Initialization not run
/// - Optimization fails
pub fn step_intrinsics_optimize(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<IntrinsicsOptimizeOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_intrinsics_init() {
        anyhow::bail!("intrinsics initialization not run - call step_intrinsics_init first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get initial estimates
    let initial_camera = session
        .state
        .initial_camera
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial camera"))?;
    let initial_poses = session
        .state
        .initial_target_poses
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial poses"))?;

    // Build initial params
    let initial_params = PlanarIntrinsicsParams::new(initial_camera, initial_poses)
        .context("failed to build params")?;

    // Convert to PlanarDataset
    let planar_dataset = input_to_planar_dataset(input)?;

    // Configure optimization
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

    // Run optimization
    let result = match optimize_planar_intrinsics(
        &planar_dataset,
        &initial_params,
        solve_opts,
        backend_opts,
    ) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("intrinsics_optimize", e.to_string());
            return Err(e);
        }
    };

    // Update state
    session.state.optimized_camera = Some(result.params.camera.clone());
    session.state.optimized_target_poses = Some(result.params.poses().to_vec());
    session.state.intrinsics_reproj_error = Some(result.mean_reproj_error);

    session.log_success_with_notes(
        "intrinsics_optimize",
        format!(
            "reproj_err={:.3}px, cost={:.2e}",
            result.mean_reproj_error, result.report.final_cost
        ),
    );

    Ok(())
}

/// Seed hand-eye transform and/or target pose with known values, auto-initializing the rest.
///
/// This step is an expert alternative to [`step_handeye_init`]. It accepts a
/// [`HandeyeManualInit`] where each field is `Option<Iso3>`:
/// - `Some(value)` — use the provided transform directly.
/// - `None` — auto-initialize via Tsai-Lenz linear estimation.
///
/// When `handeye` is `Some` but `mode_target_pose` is `None`, the target pose is derived
/// analytically from the first view (an approximation that the optimizer will refine).
///
/// Requires [`step_intrinsics_optimize`] to have been run first.
///
/// # Errors
///
/// - Input not set
/// - Intrinsics optimization not run
/// - Auto hand-eye estimation fails (when `handeye` is `None`)
pub fn step_set_handeye_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    manual: HandeyeManualInit,
    opts: Option<HandeyeInitOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_intrinsics_optimized() {
        anyhow::bail!("intrinsics optimization not run - call step_intrinsics_optimize first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let min_angle = opts
        .min_motion_angle_deg
        .unwrap_or(config.min_motion_angle_deg);

    let robot_poses: Vec<Iso3> = input
        .views
        .iter()
        .map(|v| v.meta.base_se3_gripper)
        .collect();
    let cam_se3_target = session
        .state
        .optimized_target_poses
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no optimized target poses"))?;

    // Clear previous init results (allows re-running with a different mode)
    session.state.initial_gripper_se3_camera = None;
    session.state.initial_camera_se3_base = None;
    session.state.initial_base_se3_target = None;
    session.state.initial_gripper_se3_target = None;

    let (log_pose, log_label) = match config.handeye_mode {
        vision_calibration_optim::HandEyeMode::EyeInHand => {
            let gripper_se3_camera = if let Some(he) = manual.handeye {
                he
            } else {
                let target_se3_camera: Vec<Iso3> =
                    cam_se3_target.iter().map(|t| t.inverse()).collect();
                estimate_handeye_dlt(&robot_poses, &target_se3_camera, min_angle)
                    .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))
                    .context("linear hand-eye estimation failed")?
            };
            let base_se3_target = manual
                .mode_target_pose
                .unwrap_or_else(|| robot_poses[0] * gripper_se3_camera * cam_se3_target[0]);

            session.state.initial_gripper_se3_camera = Some(gripper_se3_camera);
            session.state.initial_base_se3_target = Some(base_se3_target);

            (gripper_se3_camera, "gripper_se3_camera")
        }
        vision_calibration_optim::HandEyeMode::EyeToHand => {
            let gripper_se3_target = if let Some(he) = manual.handeye {
                he
            } else {
                estimate_gripper_se3_target_dlt(&robot_poses, &cam_se3_target, min_angle)
                    .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))
                    .context("linear gripper->target estimation failed")?
            };
            let camera_se3_base = manual.mode_target_pose.unwrap_or_else(|| {
                // T_C_T = T_C_B * T_B_G * T_G_T  =>  T_C_B = T_C_T * (T_B_G * T_G_T)^-1
                cam_se3_target[0] * (robot_poses[0] * gripper_se3_target).inverse()
            });

            session.state.initial_camera_se3_base = Some(camera_se3_base);
            session.state.initial_gripper_se3_target = Some(gripper_se3_target);

            (camera_se3_base, "camera_se3_base")
        }
    };

    let manual_note = match (manual.handeye.is_some(), manual.mode_target_pose.is_some()) {
        (true, true) => " (manual: handeye, mode_target_pose)",
        (true, false) => " (manual: handeye)",
        (false, true) => " (manual: mode_target_pose)",
        (false, false) => " (auto)",
    };

    session.log_success_with_notes(
        "handeye_init",
        format!(
            "{} |t|={:.4}m{}",
            log_label,
            log_pose.translation.vector.norm(),
            manual_note
        ),
    );

    Ok(())
}

/// Initialize hand-eye transform from robot poses and camera-target poses.
///
/// Uses Tsai-Lenz linear initialization.
///
/// Requires [`step_intrinsics_optimize`] to be run first.
///
/// To provide a known hand-eye transform, call [`step_set_handeye_init`] instead.
///
/// # Errors
///
/// - Input not set
/// - Intrinsics optimization not run
/// - Linear hand-eye estimation fails
pub fn step_handeye_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<()> {
    step_set_handeye_init(session, HandeyeManualInit::default(), opts)
}

/// Optimize hand-eye calibration using bundle adjustment.
///
/// This step jointly optimizes:
/// - Camera intrinsics and distortion
/// - Hand-eye transform
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
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<HandeyeOptimizeOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_handeye_init() {
        anyhow::bail!("hand-eye initialization not run - call step_handeye_init first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get initial estimates
    let camera = session
        .state
        .optimized_camera
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no optimized camera"))?;
    let (handeye, target_pose) = match config.handeye_mode {
        vision_calibration_optim::HandEyeMode::EyeInHand => {
            let handeye = session
                .state
                .initial_gripper_se3_camera
                .ok_or_else(|| anyhow::anyhow!("no initial gripper_se3_camera"))?;
            let target_pose = session
                .state
                .initial_base_se3_target
                .ok_or_else(|| anyhow::anyhow!("no initial base_se3_target"))?;
            (handeye, target_pose)
        }
        vision_calibration_optim::HandEyeMode::EyeToHand => {
            let handeye = session
                .state
                .initial_camera_se3_base
                .ok_or_else(|| anyhow::anyhow!("no initial camera_se3_base"))?;
            let target_pose = session
                .state
                .initial_gripper_se3_target
                .ok_or_else(|| anyhow::anyhow!("no initial gripper_se3_target"))?;
            (handeye, target_pose)
        }
    };

    // Build HandEyeDataset
    // For single camera: cam_to_rig is identity (camera IS the rig)
    let views: Vec<RigView<RobotPoseMeta>> = input
        .views
        .iter()
        .map(|v| RigView {
            meta: RobotPoseMeta {
                base_se3_gripper: v.meta.base_se3_gripper,
            },
            obs: RigViewObs {
                cameras: vec![Some(v.obs.clone())],
            },
        })
        .collect();

    let dataset = HandEyeDataset::new(views, 1, config.handeye_mode)?;

    // Build initial params
    let initial = HandEyeParams {
        cameras: vec![camera],
        cam_to_rig: vec![Iso3::identity()], // Single cam = rig frame
        handeye,
        target_poses: vec![target_pose],
    };

    // Configure solve options
    let solve_opts = HandEyeSolveOptions {
        robust_loss: config.robust_loss,
        default_fix: CameraFixMask::default(),
        camera_overrides: Vec::new(),
        fix_extrinsics: vec![true], // Fix cam_to_rig for single camera
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

    // Run optimization
    let result = match optimize_handeye(dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(e);
        }
    };

    // Update state metrics
    session.state.handeye_final_cost = Some(result.report.final_cost);
    session.state.handeye_reproj_error = Some(result.mean_reproj_error);

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

/// Run the full calibration pipeline.
///
/// Runs: intrinsics_init → intrinsics_optimize → handeye_init → handeye_optimize.
///
/// # Errors
///
/// Any error from the constituent steps.
pub fn run_calibration(session: &mut CalibrationSession<SingleCamHandeyeProblem>) -> Result<()> {
    step_intrinsics_init(session, None)?;
    step_intrinsics_optimize(session, None)?;
    step_handeye_init(session, None)?;
    step_handeye_optimize(session, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{BrownConrady5, FxFyCxCySkew, Pt2, Pt3, make_pinhole_camera};
    use vision_calibration_optim::HandEyeMode;

    fn make_test_camera() -> vision_calibration_core::PinholeCamera {
        make_pinhole_camera(
            FxFyCxCySkew {
                fx: 800.0,
                fy: 780.0,
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

    fn make_test_input() -> SingleCamHandeyeInput {
        // Ground truth
        let camera_gt = make_test_camera();
        let handeye_gt = make_iso((0.1, -0.05, 0.02), (0.05, -0.03, 0.1));
        let target_in_base_gt = make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 1.0));

        // Board points
        let board_pts: Vec<Pt3> = (0..6)
            .flat_map(|i| (0..5).map(move |j| Pt3::new(i as f64 * 0.05, j as f64 * 0.05, 0.0)))
            .collect();

        // Generate views with different robot poses
        let robot_poses = [
            make_iso((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            make_iso((0.1, 0.0, 0.0), (0.1, 0.0, 0.0)),
            make_iso((0.0, 0.1, 0.0), (0.0, 0.1, 0.0)),
            make_iso((0.05, 0.05, 0.0), (0.05, -0.05, 0.0)),
        ];

        let views: Vec<_> = robot_poses
            .iter()
            .map(|robot_pose| {
                // Camera pose: T_C_T = (T_B_G * T_G_C)^-1 * T_B_T
                let cam_pose = (robot_pose * handeye_gt).inverse() * target_in_base_gt;

                // Project points
                let points_2d: Vec<Pt2> = board_pts
                    .iter()
                    .map(|p| {
                        let p_cam = cam_pose.transform_point(p);
                        camera_gt.project_point_c(&p_cam.coords).unwrap()
                    })
                    .collect();

                super::super::problem::SingleCamHandeyeView {
                    obs: CorrespondenceView::new(board_pts.clone(), points_2d).unwrap(),
                    meta: super::super::problem::HandeyeMeta {
                        base_se3_gripper: *robot_pose,
                    },
                }
            })
            .collect();

        SingleCamHandeyeInput { views }
    }

    #[test]
    fn step_intrinsics_init_computes_estimate() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();

        step_intrinsics_init(&mut session, None).unwrap();

        assert!(session.state.has_intrinsics_init());
        let k = session.state.initial_camera.unwrap().k;
        // Check intrinsics are reasonable (within 20% of ground truth)
        assert!((k.fx - 800.0).abs() < 160.0);
        assert!((k.fy - 780.0).abs() < 160.0);
    }

    #[test]
    fn step_intrinsics_optimize_requires_init() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();

        let result = step_intrinsics_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("init"));
    }

    #[test]
    fn step_handeye_init_requires_intrinsics() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();

        let result = step_handeye_init(&mut session, None);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("intrinsics optimization")
        );
    }

    #[test]
    fn step_handeye_optimize_requires_init() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init(&mut session, None).unwrap();
        step_intrinsics_optimize(&mut session, None).unwrap();

        let result = step_handeye_optimize(&mut session, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hand-eye"));
    }

    #[test]
    fn set_input_clears_state() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        step_intrinsics_init(&mut session, None).unwrap();

        assert!(session.state.has_intrinsics_init());

        // Set new input should clear state
        session.set_input(make_test_input()).unwrap();
        assert!(!session.state.has_intrinsics_init());
    }

    #[test]
    fn set_config_keeps_output() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();

        assert!(session.has_output());

        // Change config
        session
            .set_config(super::super::problem::SingleCamHandeyeConfig {
                max_iters: 100,
                handeye_mode: HandEyeMode::EyeToHand,
                ..Default::default()
            })
            .unwrap();

        // Output should still be there
        assert!(session.has_output());
    }

    #[test]
    fn json_roundtrip() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::with_description(
            "Test single-cam hand-eye",
        );
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();
        session.export().unwrap();

        let json = session.to_json().unwrap();
        let restored = CalibrationSession::<SingleCamHandeyeProblem>::from_json(&json).unwrap();

        assert_eq!(
            restored.metadata.description,
            Some("Test single-cam hand-eye".to_string())
        );
        assert!(restored.has_input());
        assert!(restored.has_output());
        assert_eq!(restored.exports.len(), 1);
    }
}
