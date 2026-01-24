//! Step functions for single-camera hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<SingleCamHandeyeProblem>` to perform calibration.
//!
//! # Example
//!
//! ```ignore
//! use calib_pipeline::session::v2::CalibrationSession;
//! use calib_pipeline::single_cam_handeye::{
//!     SingleCamHandeyeProblem, step_intrinsics_init, step_intrinsics_optimize,
//!     step_handeye_init, step_handeye_optimize,
//! };
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
//! ```

use anyhow::{Context, Result};
use calib_core::{
    make_pinhole_camera, CameraFixMask, CorrespondenceView, Iso3, NoMeta, PlanarDataset, RigView,
    RigViewObs, View,
};
use calib_linear::estimate_handeye_dlt;
use calib_linear::prelude::*;
use calib_optim::{
    optimize_handeye, optimize_planar_intrinsics, BackendSolveOptions, HandEyeDataset,
    HandEyeParams, HandEyeSolveOptions, PlanarIntrinsicsParams, PlanarIntrinsicsSolveOptions,
    RobotPoseMeta,
};

use crate::session::CalibrationSession;

use super::problem::{SingleCamHandeyeInput, SingleCamHandeyeProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

/// Options for intrinsics initialization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsInitOptions {
    /// Override the number of iterations.
    pub iterations: Option<usize>,
}

/// Options for intrinsics optimization step.
#[derive(Debug, Clone, Default)]
pub struct IntrinsicsOptimOptions {
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
pub struct HandeyeOptimOptions {
    /// Override the maximum number of iterations.
    pub max_iters: Option<usize>,
    /// Override verbosity level.
    pub verbosity: Option<usize>,
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
fn estimate_target_pose(k_matrix: &calib_core::Mat3, view: &CorrespondenceView) -> Result<Iso3> {
    // Compute homography
    let board_2d: Vec<calib_core::Pt2> = view
        .points_3d
        .iter()
        .map(|p| calib_core::Pt2::new(p.x, p.y))
        .collect();
    let pixel_2d: Vec<calib_core::Pt2> = view
        .points_2d
        .iter()
        .map(|v| calib_core::Pt2::new(v.x, v.y))
        .collect();

    let h = dlt_homography(&board_2d, &pixel_2d).context("failed to compute homography")?;

    // Recover pose from homography
    estimate_planar_pose_from_h(k_matrix, &h).context("failed to recover pose from homography")
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize intrinsics from observations.
///
/// This step computes:
/// 1. Initial intrinsics using Zhang's method with iterative distortion estimation
/// 2. Initial target poses from homographies and estimated intrinsics
///
/// Updates `session.state` with intermediate results.
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
    session.validate()?;
    let input = session.require_input()?;

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Build init options
    let init_opts = IterativeIntrinsicsOptions {
        iterations: opts.iterations.unwrap_or(config.intrinsics_init_iterations),
        distortion_opts: DistortionFitOptions {
            fix_k3: config.fix_k3,
            fix_tangential: config.fix_tangential,
            iters: 8,
        },
        zero_skew: config.zero_skew,
    };

    // Convert to PlanarDataset
    let planar_dataset = input_to_planar_dataset(input)?;

    // Estimate intrinsics
    let camera = match estimate_intrinsics_iterative(&planar_dataset, init_opts) {
        Ok(c) => c,
        Err(e) => {
            session.log_failure("intrinsics_init", e.to_string());
            return Err(e);
        }
    };

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

    // Estimate initial target poses
    let initial_poses: Vec<Iso3> = input
        .views
        .iter()
        .enumerate()
        .map(|(idx, view)| {
            estimate_target_pose(&k_matrix, &view.obs)
                .with_context(|| format!("failed to estimate pose for view {}", idx))
        })
        .collect::<Result<Vec<_>>>()?;

    // Update state
    session.state.initial_intrinsics = Some(camera.k);
    session.state.initial_distortion = Some(camera.dist);
    session.state.initial_target_poses = Some(initial_poses);

    session.log_success_with_notes(
        "intrinsics_init",
        format!(
            "fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
            camera.k.fx, camera.k.fy, camera.k.cx, camera.k.cy
        ),
    );

    Ok(())
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
    opts: Option<IntrinsicsOptimOptions>,
) -> Result<()> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_intrinsics_init() {
        anyhow::bail!("intrinsics initialization not run - call step_intrinsics_init first");
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get initial estimates
    let initial_intrinsics = session.state.initial_intrinsics.unwrap();
    let initial_distortion = session.state.initial_distortion.unwrap_or_default();
    let initial_poses = session
        .state
        .initial_target_poses
        .clone()
        .ok_or_else(|| anyhow::anyhow!("no initial poses"))?;

    // Build initial params
    let camera = make_pinhole_camera(initial_intrinsics, initial_distortion);
    let initial_params =
        PlanarIntrinsicsParams::new(camera, initial_poses).context("failed to build params")?;

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

/// Initialize hand-eye transform from robot poses and camera-target poses.
///
/// Uses Tsai-Lenz linear initialization.
///
/// Requires [`step_intrinsics_optimize`] to be run first.
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

    // Get robot poses and camera-target poses
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

    // calib-linear expects `target_se3_camera` (camera -> target), while planar intrinsics
    // produces `cam_se3_target` (target -> camera).
    let target_se3_camera: Vec<Iso3> = cam_se3_target.iter().map(|t| t.inverse()).collect();

    // Linear hand-eye estimation
    let handeye = match estimate_handeye_dlt(&robot_poses, &target_se3_camera, min_angle) {
        Ok(h) => h,
        Err(e) => {
            session.log_failure("handeye_init", e.to_string());
            return Err(e).context("linear hand-eye estimation failed");
        }
    };

    // Estimate initial target pose in base frame
    // For EyeInHand: T_B_T = T_B_G * T_G_C * T_C_T = base_se3_gripper * handeye * cam_target_pose
    // Take the first view as reference
    let target_se3_base = match config.handeye_mode {
        calib_optim::HandEyeMode::EyeInHand => {
            // handeye = T_G_C, we need T_B_T = T_B_G * T_G_C * T_C_T
            robot_poses[0] * handeye * cam_se3_target[0]
        }
        calib_optim::HandEyeMode::EyeToHand => {
            // For EyeToHand: handeye = T_C_B
            // T_G_T = T_G_B * T_B_C * T_C_T = robot^-1 * handeye^-1 * cam_target
            // But we want target_se3_gripper for this mode
            handeye.inverse() * cam_se3_target[0]
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
    opts: Option<HandeyeOptimOptions>,
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
    let handeye = session
        .state
        .initial_handeye
        .ok_or_else(|| anyhow::anyhow!("no initial handeye"))?;
    let target_se3_base = session
        .state
        .initial_target_se3_base
        .ok_or_else(|| anyhow::anyhow!("no initial target pose"))?;

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
        target_poses: vec![target_se3_base],
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
    use calib_core::{make_pinhole_camera, BrownConrady5, FxFyCxCySkew, Pt2, Pt3};
    use calib_optim::HandEyeMode;
    use nalgebra::{Rotation3, Translation3};

    fn make_test_camera() -> calib_core::PinholeCamera {
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
        let k = session.state.initial_intrinsics.unwrap();
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
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("intrinsics optimization"));
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
