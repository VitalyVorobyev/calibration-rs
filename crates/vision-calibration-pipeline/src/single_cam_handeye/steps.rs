//! Step functions for single-camera hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<SingleCamHandeyeProblem>` to perform calibration.
//!
//! [`SingleCamHandeyeProblem`](super::SingleCamHandeyeProblem) has a
//! runnable example of the sequence.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{
    CameraParams, DistortionParams, FxFyCxCy, Iso3, NoMeta, PlanarDataset, Real, RigView,
    RigViewObs, ScheimpflugParams, View,
};
use vision_calibration_optim::{
    HandeyeDataset, HandeyeMode, HandeyeParams, PlanarIntrinsicsParams, RobotPoseMeta,
    optimize_handeye,
};

use crate::planar_family::{
    CameraSeed, Provenance, bootstrap_camera, check_camera_model, check_camera_seed,
    optimize_camera,
};
use crate::rig_family::{handeye_solve_opts, linear_handeye};
use crate::session::CalibrationSession;

use super::problem::{SingleCamHandeyeInput, SingleCamHandeyeProblem};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

use crate::common::{
    HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions, IntrinsicsOptimizeOptions,
};

/// Manual initialization seeds for the **intrinsics stage** of single-camera
/// hand-eye calibration.
///
/// All fields are `Option<T>`. `None` means auto-init via Zhang's method (same path
/// as plain `step_intrinsics_init`); `Some` means use the provided value. The
/// seeds are values in the configured camera model.
///
/// Partial-seed semantics mirror `IntrinsicsManualInit`: when `intrinsics` is seeded
/// but `target_poses` is `None`, poses are recovered from per-view homographies
/// using the manual intrinsics. When intrinsics are seeded and distortion is not,
/// distortion starts at zero. A distortion seed's kind must equal the
/// camera's configured distortion kind (a typed error otherwise) and is used
/// as given. `sensor` seeds
/// the tilt of a Scheimpflug camera (a seeded tilt replaces the tilt sweep);
/// an untilted camera takes only a zero tilt.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsManualInit {
    /// Manual intrinsics seed.
    pub intrinsics: Option<FxFyCxCy<Real>>,
    /// Manual distortion seed, of the kind `camera.model.distortion` names.
    pub distortion: Option<DistortionParams>,
    /// Manual sensor tilt seed (Scheimpflug camera; zero otherwise).
    pub sensor: Option<ScheimpflugParams>,
    /// Manual per-view target poses (`camera_se3_target`).
    pub target_poses: Option<Vec<Iso3>>,
}

/// Manual initialization seeds for the **hand-eye stage** of single-camera
/// hand-eye calibration.
///
/// Fields are mode-dependent — see `session.config.handeye_mode`:
/// - `HandeyeMode::EyeInHand` consumes `gripper_se3_camera` and `base_se3_target`.
/// - `HandeyeMode::EyeToHand` consumes `camera_se3_base` and `gripper_se3_target`.
///
/// Fields irrelevant to the active mode are ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeManualInit {
    /// (EyeInHand) Camera mounted on gripper: `T_G_C`.
    pub gripper_se3_camera: Option<Iso3>,
    /// (EyeInHand) Target attached in base frame: `T_B_T`.
    pub base_se3_target: Option<Iso3>,
    /// (EyeToHand) Camera fixed in scene observing gripper-mounted target: `T_C_B`.
    pub camera_se3_base: Option<Iso3>,
    /// (EyeToHand) Target attached to gripper: `T_G_T`.
    pub gripper_se3_target: Option<Iso3>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Convert single-cam input to PlanarDataset for intrinsics calibration.
fn input_to_planar_dataset(input: &SingleCamHandeyeInput) -> Result<PlanarDataset, Error> {
    let views: Vec<View<NoMeta>> = input
        .views
        .iter()
        .map(|v| View::without_meta(v.obs.clone()))
        .collect();
    PlanarDataset::new(views).map_err(Error::Core)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

/// Typed return value of [`step_intrinsics_init`] / [`step_intrinsics_init_with_seed`].
///
/// Mirrors the values written into `session.state` so consumers can read them
/// directly without unwrapping `Option` fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsInitResult {
    /// Initial camera estimate, in the configured model.
    pub camera: CameraParams,
    /// Per-view initial target poses (`camera_se3_target`).
    pub target_poses: Vec<Iso3>,
}

/// Typed return value of [`step_intrinsics_optimize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsOptimizeResult {
    /// Refined camera, in the configured model.
    pub camera: CameraParams,
    /// Refined per-view target poses (`camera_se3_target`).
    pub target_poses: Vec<Iso3>,
    /// Mean reprojection error in pixels after intrinsics optimization.
    pub mean_reproj_error: f64,
}

/// Typed return value of [`step_handeye_init`] / [`step_handeye_init_with_seed`].
///
/// The pose fields are mutually-exclusive by hand-eye mode:
/// - `EyeInHand` populates `gripper_se3_camera` and `base_se3_target`; the
///   `EyeToHand` fields are `None`.
/// - `EyeToHand` populates `camera_se3_base` and `gripper_se3_target`; the
///   `EyeInHand` fields are `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeInitResult {
    /// (EyeInHand) Hand-eye transform `T_G_C` — gripper-from-camera.
    pub gripper_se3_camera: Option<Iso3>,
    /// (EyeInHand) Fixed target pose `T_B_T` — base-from-target.
    pub base_se3_target: Option<Iso3>,
    /// (EyeToHand) Hand-eye transform `T_C_B` — camera-from-base.
    pub camera_se3_base: Option<Iso3>,
    /// (EyeToHand) Target attached to gripper `T_G_T` — gripper-from-target.
    pub gripper_se3_target: Option<Iso3>,
}

/// Typed return value of [`step_handeye_optimize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeOptimizeResult {
    /// Mean reprojection error in pixels after hand-eye BA.
    pub mean_reproj_error: f64,
    /// Final solver cost from the hand-eye BA.
    pub final_cost: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Functions
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize the camera and per-view target poses from any combination of
/// manual seeds and auto-estimation.
///
/// This is the load-bearing intrinsics-stage init. [`step_intrinsics_init`] is a
/// thin delegate with `IntrinsicsManualInit::default()`.
///
/// The camera's model is `session.config.camera.model`: Zhang's method, or for a
/// Scheimpflug sensor its tilted variant with a linear tilt sweep. See
/// [`IntrinsicsManualInit`] for partial-seed semantics.
///
/// # Errors
///
/// - Input not set; a non-zero tilt seed for an untilted camera; manual
///   target poses whose count differs from the view count.
/// - The linear initialization fails.
pub fn step_intrinsics_init_with_seed(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    manual: IntrinsicsManualInit,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    session.validate()?;
    let input = session.require_input()?;
    let view_count = input.views.len();
    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let model = config.camera.model;
    let seed = CameraSeed {
        intrinsics: manual.intrinsics,
        distortion: manual.distortion,
        sensor: manual.sensor,
    };
    if let Err(e) = check_camera_seed(
        model,
        &seed,
        manual.target_poses.as_deref(),
        "target_poses",
        view_count,
    ) {
        session.log_failure("intrinsics_init", e.to_string());
        return Err(e);
    }
    let mut provenance = Provenance::of_camera_seed(model, &seed);
    provenance.note("target_poses", manual.target_poses.is_some());

    let init_opts = config
        .intrinsics
        .camera_init_opts(opts.iterations, &config.camera);
    let planar_dataset = input_to_planar_dataset(input)?;
    let bootstrap = match bootstrap_camera(0, model, &planar_dataset, &seed, init_opts) {
        Ok(b) => b,
        Err(e) => {
            session.log_failure("intrinsics_init", e.to_string());
            return Err(e);
        }
    };
    let camera = bootstrap.camera;
    let initial_poses = manual.target_poses.unwrap_or(bootstrap.poses);

    session.state.initial_camera = Some(camera.clone());
    session.state.initial_target_poses = Some(initial_poses.clone());

    let k = camera.intrinsics;
    session.log_success_with_notes(
        "intrinsics_init",
        format!(
            "fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1} {}",
            k.fx,
            k.fy,
            k.cx,
            k.cy,
            provenance.source()
        ),
    );

    Ok(IntrinsicsInitResult {
        camera,
        target_poses: initial_poses,
    })
}

/// Initialize intrinsics from observations using full auto-init.
///
/// Convenience wrapper around [`step_intrinsics_init_with_seed`] with default seeds.
pub fn step_intrinsics_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    step_intrinsics_init_with_seed(session, IntrinsicsManualInit::default(), opts)
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
) -> Result<IntrinsicsOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_intrinsics_init() {
        return Err(Error::not_available(
            "intrinsics init (call step_intrinsics_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get initial estimates
    let initial_camera = session
        .state
        .initial_camera
        .clone()
        .ok_or_else(|| Error::not_available("initial camera"))?;
    let initial_poses = session
        .state
        .initial_target_poses
        .clone()
        .ok_or_else(|| Error::not_available("initial poses"))?;

    check_camera_model(&initial_camera, config.camera.model, "step_intrinsics_init")?;

    // Build initial params
    let initial_params = PlanarIntrinsicsParams::new(initial_camera, initial_poses)
        .map_err(|e| Error::numerical(format!("failed to build params: {e}")))?;

    // Convert to PlanarDataset
    let planar_dataset = input_to_planar_dataset(input)?;

    let backend_opts =
        config
            .solver
            .step_backend_options(session.control(), opts.max_iters, opts.verbosity);

    // Run optimization: a plain planar solve, or the staged tilt
    // multi-start for a Scheimpflug sensor.
    let result = match optimize_camera(
        0,
        &planar_dataset,
        &initial_params,
        &config.camera,
        &config.solver,
        backend_opts,
    ) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("intrinsics_optimize", e.to_string());
            return Err(e);
        }
    };

    // Update state
    let camera = result.params.camera.clone();
    let target_poses = result.params.camera_se3_target.to_vec();
    let mean_reproj_error = result.mean_reproj_error;
    session.state.optimized_camera = Some(camera.clone());
    session.state.optimized_target_poses = Some(target_poses.clone());
    session.state.intrinsics_reproj_error = Some(mean_reproj_error);

    session.log_success_with_notes(
        "intrinsics_optimize",
        format!(
            "reproj_err={:.3}px, cost={:.2e}",
            mean_reproj_error, result.report.final_cost
        ),
    );

    Ok(IntrinsicsOptimizeResult {
        camera,
        target_poses,
        mean_reproj_error,
    })
}

/// Initialize the hand-eye stage from any combination of manual seeds and
/// auto-estimation (the closed-form solver of `handeye_init.method`).
///
/// This is the load-bearing handeye-stage init. [`step_handeye_init`] is a thin
/// delegate with `HandeyeManualInit::default()`.
///
/// Mode-aware behavior — `session.config.handeye_init.handeye_mode` selects which
/// seed fields are consumed:
/// - `EyeInHand`: `gripper_se3_camera` and `base_se3_target`, both from the
///   linear solve when not seeded; a seeded `gripper_se3_camera` closes
///   `base_se3_target` through the first view.
/// - `EyeToHand`: `gripper_se3_target` and `camera_se3_base`, likewise.
///
/// Fields irrelevant to the active mode are ignored.
///
/// # Errors
///
/// - Input not set, intrinsics optimization not run.
/// - The linear solve fails (when not seeded).
pub fn step_handeye_init_with_seed(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    manual: HandeyeManualInit,
    opts: Option<HandeyeInitOptions>,
) -> Result<HandeyeInitResult, Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_intrinsics_optimized() {
        return Err(Error::not_available(
            "intrinsics optimization (call step_intrinsics_optimize first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let min_angle = opts
        .min_motion_angle_deg
        .unwrap_or(config.handeye_init.min_motion_angle_deg);
    let method = opts.method.unwrap_or(config.handeye_init.method);

    let robot_poses: Vec<Iso3> = input
        .views
        .iter()
        .map(|v| v.meta.base_se3_gripper)
        .collect();
    let cam_se3_target = session
        .state
        .optimized_target_poses
        .clone()
        .ok_or_else(|| Error::not_available("optimized target poses"))?;

    // Clear previous init results (allows re-running with a different mode).
    session.state.initial_gripper_se3_camera = None;
    session.state.initial_camera_se3_base = None;
    session.state.initial_base_se3_target = None;
    session.state.initial_gripper_se3_target = None;

    let mode = config.handeye_init.handeye_mode;
    let (seeded_first, seeded_second) = match mode {
        HandeyeMode::EyeInHand => (manual.gripper_se3_camera, manual.base_se3_target),
        HandeyeMode::EyeToHand => (manual.gripper_se3_target, manual.camera_se3_base),
    };
    // Each mode seeds one pose and closes the other through the first view.
    let mut provenance = Provenance::default();
    let (first, second) = match mode {
        HandeyeMode::EyeInHand => ("gripper_se3_camera", "base_se3_target"),
        HandeyeMode::EyeToHand => ("gripper_se3_target", "camera_se3_base"),
    };
    provenance.note(first, seeded_first.is_some());
    provenance.note(second, seeded_second.is_some());

    // The linear solve gives both poses.
    let (first_pose, estimated_second) = match seeded_first {
        Some(t) => (t, None),
        None => {
            let (handeye, target) =
                linear_handeye(mode, method, &robot_poses, &cam_se3_target, min_angle)
                    .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))?;
            match mode {
                HandeyeMode::EyeInHand => (handeye, Some(target)),
                HandeyeMode::EyeToHand => (target, Some(handeye)),
            }
        }
    };
    let (log_pose, log_label, result) = match mode {
        HandeyeMode::EyeInHand => {
            let gripper_se3_camera = first_pose;
            let base_se3_target = seeded_second
                .or(estimated_second)
                .unwrap_or_else(|| robot_poses[0] * gripper_se3_camera * cam_se3_target[0]);
            session.state.initial_gripper_se3_camera = Some(gripper_se3_camera);
            session.state.initial_base_se3_target = Some(base_se3_target);
            (
                gripper_se3_camera,
                "gripper_se3_camera",
                HandeyeInitResult {
                    gripper_se3_camera: Some(gripper_se3_camera),
                    base_se3_target: Some(base_se3_target),
                    camera_se3_base: None,
                    gripper_se3_target: None,
                },
            )
        }
        HandeyeMode::EyeToHand => {
            let gripper_se3_target = first_pose;
            // T_C_T = T_C_B * T_B_G * T_G_T => T_C_B = T_C_T * (T_B_G * T_G_T)^-1.
            let camera_se3_base = seeded_second.or(estimated_second).unwrap_or_else(|| {
                cam_se3_target[0] * (robot_poses[0] * gripper_se3_target).inverse()
            });
            session.state.initial_camera_se3_base = Some(camera_se3_base);
            session.state.initial_gripper_se3_target = Some(gripper_se3_target);
            (
                camera_se3_base,
                "camera_se3_base",
                HandeyeInitResult {
                    gripper_se3_camera: None,
                    base_se3_target: None,
                    camera_se3_base: Some(camera_se3_base),
                    gripper_se3_target: Some(gripper_se3_target),
                },
            )
        }
    };

    session.log_success_with_notes(
        "handeye_init",
        format!(
            "{} |t|={:.4}m {}",
            log_label,
            log_pose.translation.vector.norm(),
            provenance.source()
        ),
    );

    Ok(result)
}

/// Initialize the hand-eye transform using full auto-init (the closed-form
/// solver of `handeye_init.method`).
///
/// Convenience wrapper around [`step_handeye_init_with_seed`] with default seeds.
pub fn step_handeye_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<HandeyeInitResult, Error> {
    step_handeye_init_with_seed(session, HandeyeManualInit::default(), opts)
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
) -> Result<HandeyeOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?;

    if !session.state.has_handeye_init() {
        return Err(Error::not_available(
            "handeye init (call step_handeye_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    // Get initial estimates
    let camera = session
        .state
        .optimized_camera
        .clone()
        .ok_or_else(|| Error::not_available("optimized camera"))?;
    check_camera_model(&camera, config.camera.model, "step_intrinsics_init")?;
    let (handeye, target_pose) = match config.handeye_init.handeye_mode {
        HandeyeMode::EyeInHand => {
            let handeye = session
                .state
                .initial_gripper_se3_camera
                .ok_or_else(|| Error::not_available("initial gripper_se3_camera"))?;
            let target_pose = session
                .state
                .initial_base_se3_target
                .ok_or_else(|| Error::not_available("initial base_se3_target"))?;
            (handeye, target_pose)
        }
        HandeyeMode::EyeToHand => {
            let handeye = session
                .state
                .initial_camera_se3_base
                .ok_or_else(|| Error::not_available("initial camera_se3_base"))?;
            let target_pose = session
                .state
                .initial_gripper_se3_target
                .ok_or_else(|| Error::not_available("initial gripper_se3_target"))?;
            (handeye, target_pose)
        }
    };

    // Build HandeyeDataset
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

    let dataset = HandeyeDataset::new(views, 1, config.handeye_init.handeye_mode)?;

    // Build initial params
    let initial = HandeyeParams {
        cameras: vec![camera],
        cam_to_rig: vec![Iso3::identity()], // Single cam = rig frame
        handeye,
        target_poses: vec![target_pose],
    };

    // The camera is the rig: its extrinsics stay the identity.
    let solve_opts = handeye_solve_opts(
        config.solver.robust_loss,
        vec![config.camera.fix.clone()],
        vec![true],
        &config.robot_poses,
    );
    let backend_opts =
        config
            .solver
            .step_backend_options(session.control(), opts.max_iters, opts.verbosity);

    // Run optimization
    let result = match optimize_handeye(dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    // Update state metrics
    let final_cost = result.report.final_cost;
    let mean_reproj_error = result.mean_reproj_error;
    session.state.handeye_final_cost = Some(final_cost);
    session.state.handeye_reproj_error = Some(mean_reproj_error);

    // Set output
    session.set_output(result);

    session.log_success_with_notes("handeye_optimize", format!("final_cost={final_cost:.2e}"));

    Ok(HandeyeOptimizeResult {
        mean_reproj_error,
        final_cost,
    })
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
pub fn run_calibration(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
) -> Result<(), Error> {
    session.begin_step("intrinsics_init")?;
    let _ = step_intrinsics_init(session, None)?;
    session.begin_step("intrinsics_optimize")?;
    let _ = step_intrinsics_optimize(session, None)?;
    session.begin_step("handeye_init")?;
    let _ = step_handeye_init(session, None)?;
    session.begin_step("handeye_optimize")?;
    let _ = step_handeye_optimize(session, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCy, Pt2, Pt3, make_pinhole_camera,
    };
    use vision_calibration_optim::HandeyeMode;

    fn make_test_camera() -> vision_calibration_core::PinholeCamera {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 780.0,
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
                    meta: crate::common::RobotPoseMeta {
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
        let k = session.state.initial_camera.unwrap().intrinsics;
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
        assert!(result.unwrap_err().to_string().contains("handeye"));
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
                solver: crate::common::config::SolverConfig {
                    max_iters: 100,
                    ..Default::default()
                },
                handeye_init: crate::common::config::HandeyeInitConfig {
                    handeye_mode: HandeyeMode::EyeToHand,
                    ..Default::default()
                },
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
