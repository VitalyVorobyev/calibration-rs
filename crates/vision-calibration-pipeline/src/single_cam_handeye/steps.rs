//! Step functions for single-camera hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<SingleCamHandeyeProblem>` to perform calibration.
//!
//! [`SingleCamHandeyeProblem`](super::SingleCamHandeyeProblem) has a
//! runnable example of the sequence.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{CameraParams, Iso3, NoMeta, PlanarDataset, View};
use vision_calibration_optim::{HandeyeMode, PlanarIntrinsicsParams};

use crate::planar_family::{check_camera_model, init_camera_with_seed, optimize_camera};
pub use crate::rig_family::HandeyeOptimizeResult;
use crate::rig_family::{init_handeye, optimize_handeye_stage};
use crate::session::CalibrationSession;

use super::problem::{SingleCamHandeyeInput, SingleCamHandeyeProblem, SingleCamHandeyeSolution};

// ─────────────────────────────────────────────────────────────────────────────
// Step Options
// ─────────────────────────────────────────────────────────────────────────────

use crate::common::{
    HandeyeInitOptions, HandeyeOptimizeOptions, IntrinsicsInitOptions, IntrinsicsManualInit,
    IntrinsicsOptimizeOptions,
};

/// Manual initialization seeds for the **hand-eye stage** of single-camera
/// hand-eye calibration.
///
/// One variant per mounting, named like [`SingleCamHandeyeSolution`] and
/// tagged by `mode` in JSON, so a seed reads like the export it seeds. The
/// variant must be the one of `session.config.handeye_init.handeye_mode`; any
/// other is an [`Error::InvalidInput`] before any work.
///
/// A pose left `None` is estimated. With one pose seeded the other closes
/// through the first view, without the closed-form solve; with neither
/// seeded, the closed-form solver of `handeye_init.method` gives both.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HandeyeManualInit {
    /// The camera rides on the gripper; the target stands in the base frame.
    EyeInHand {
        /// Camera mounted on the gripper: `T_G_C`.
        gripper_se3_cam: Option<Iso3>,
        /// Target fixed in the base frame: `T_B_T`.
        base_se3_target: Option<Iso3>,
    },
    /// The camera stands in the base frame; the target rides on the gripper.
    EyeToHand {
        /// Camera fixed in the scene: `T_C_B`.
        cam_se3_base: Option<Iso3>,
        /// Target attached to the gripper: `T_G_T`.
        gripper_se3_target: Option<Iso3>,
    },
}

impl HandeyeManualInit {
    /// The seed of `mode` with nothing seeded: both poses estimated.
    pub fn unseeded(mode: HandeyeMode) -> Self {
        match mode {
            HandeyeMode::EyeInHand => Self::EyeInHand {
                gripper_se3_cam: None,
                base_se3_target: None,
            },
            HandeyeMode::EyeToHand => Self::EyeToHand {
                cam_se3_base: None,
                gripper_se3_target: None,
            },
        }
    }

    /// The mode the seed is for.
    pub fn mode(&self) -> HandeyeMode {
        match self {
            Self::EyeInHand { .. } => HandeyeMode::EyeInHand,
            Self::EyeToHand { .. } => HandeyeMode::EyeToHand,
        }
    }

    /// The seeded `(hand-eye, target)` poses of the mode.
    fn poses(&self) -> (Option<Iso3>, Option<Iso3>) {
        match *self {
            Self::EyeInHand {
                gripper_se3_cam,
                base_se3_target,
            } => (gripper_se3_cam, base_se3_target),
            Self::EyeToHand {
                cam_se3_base,
                gripper_se3_target,
            } => (cam_se3_base, gripper_se3_target),
        }
    }
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
    /// Per-view initial target poses (`cam_se3_target`).
    pub cam_se3_target: Vec<Iso3>,
}

/// Typed return value of [`step_intrinsics_optimize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct IntrinsicsOptimizeResult {
    /// Refined camera, in the configured model.
    pub camera: CameraParams,
    /// Refined per-view target poses (`cam_se3_target`).
    pub cam_se3_target: Vec<Iso3>,
    /// Mean reprojection error in pixels after intrinsics optimization.
    pub mean_reproj_error: f64,
}

/// Typed return value of [`step_handeye_init`] / [`step_handeye_init_with_seed`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeInitResult {
    /// The initial hand-eye and target pose of the configured mode; both
    /// poses are always present.
    pub handeye: SingleCamHandeyeSolution,
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
    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let model = config.camera.model;
    let init_opts = config
        .intrinsics_init
        .camera_init_opts(opts.iterations, &config.camera);
    let planar_dataset = input_to_planar_dataset(input)?;
    let init = match init_camera_with_seed(model, &planar_dataset, manual, init_opts) {
        Ok(init) => init,
        Err(e) => {
            session.log_failure("intrinsics_init", e.to_string());
            return Err(e);
        }
    };

    session.state.initial_camera = Some(init.camera.clone());
    session.state.initial_cam_se3_target = Some(init.cam_se3_target.clone());
    session.log_success_with_notes("intrinsics_init", init.log_note());

    Ok(IntrinsicsInitResult {
        camera: init.camera,
        cam_se3_target: init.cam_se3_target,
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
        .initial_cam_se3_target
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
        config.intrinsics_optimize.reprojection_loss,
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
    let cam_se3_target = result.params.cam_se3_target.to_vec();
    let mean_reproj_error = result.mean_reproj_error;
    session.state.optimized_camera = Some(camera.clone());
    session.state.optimized_cam_se3_target = Some(cam_se3_target.clone());
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
        cam_se3_target,
        mean_reproj_error,
    })
}

/// Initialize the hand-eye stage from any combination of manual seeds and
/// estimation. [`step_handeye_init`] is this step with nothing seeded.
///
/// The seed's variant must be the one of
/// `session.config.handeye_init.handeye_mode`. A seeded pose is used as
/// given. With one pose seeded the other closes through the first view
/// (the mode's chain puts the target where the camera saw it), without the
/// closed-form solve; with neither seeded, the closed-form solver of
/// `handeye_init.method` gives both. The previous initialization is cleared
/// once the step starts its work, so a failed run leaves none behind.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when the seed's variant is not the configured
///   mode's, raised before any work.
/// - Input not set, intrinsics optimization not run.
/// - The closed-form solve fails (nothing seeded).
pub fn step_handeye_init_with_seed(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    manual: HandeyeManualInit,
    opts: Option<HandeyeInitOptions>,
) -> Result<HandeyeInitResult, Error> {
    let handeye = init_handeye(session, manual.mode(), manual.poses(), opts)?;
    Ok(HandeyeInitResult { handeye })
}

/// Initialize the hand-eye transform using full auto-init (the closed-form
/// solver of `handeye_init.method`).
///
/// Convenience wrapper around [`step_handeye_init_with_seed`] with nothing
/// seeded.
pub fn step_handeye_init(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<HandeyeInitResult, Error> {
    let manual = HandeyeManualInit::unseeded(session.config.handeye_init.handeye_mode);
    step_handeye_init_with_seed(session, manual, opts)
}

/// Optimize hand-eye calibration using bundle adjustment.
///
/// This step jointly optimizes:
/// - The camera (intrinsics, distortion, sensor), holding its `fix` lists
/// - The hand-eye transform
/// - The target pose of the mode
/// - Optionally: per-view robot pose corrections
///
/// Requires [`step_handeye_init`] to be run first.
///
/// # Errors
///
/// - Input not set
/// - Hand-eye initialization not run, or run in another mode
/// - Optimization fails
pub fn step_handeye_optimize(
    session: &mut CalibrationSession<SingleCamHandeyeProblem>,
    opts: Option<HandeyeOptimizeOptions>,
) -> Result<HandeyeOptimizeResult, Error> {
    optimize_handeye_stage(session, opts)
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

    #[test]
    fn a_seed_of_the_other_mode_is_an_invalid_input_before_any_work() {
        // Nothing has run: the mode check comes before the stage gates.
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        let seed = HandeyeManualInit::unseeded(HandeyeMode::EyeToHand);
        assert_eq!(
            session.config.handeye_init.handeye_mode,
            HandeyeMode::EyeInHand
        );
        let err = step_handeye_init_with_seed(&mut session, seed, None).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    #[test]
    fn handeye_seed_json_is_tagged_like_the_export() {
        let seed = HandeyeManualInit::EyeToHand {
            cam_se3_base: Some(Iso3::identity()),
            gripper_se3_target: None,
        };
        let json = serde_json::to_value(&seed).unwrap();
        assert_eq!(json["mode"], "eye_to_hand");
        let back: HandeyeManualInit = serde_json::from_value(json).unwrap();
        assert_eq!(back, seed);
        // Unseeded poses may be left out.
        let bare: HandeyeManualInit = serde_json::from_str(r#"{"mode":"eye_in_hand"}"#).unwrap();
        assert_eq!(bare, HandeyeManualInit::unseeded(HandeyeMode::EyeInHand));
    }
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
    fn set_config_keeps_output_in_the_mode_it_was_solved_in() {
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();

        assert!(session.has_output());
        let solved = session.export_peek().unwrap().handeye;
        assert_eq!(solved.mode(), HandeyeMode::EyeInHand);

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

        // The output stays, and still reads as the eye-in-hand solution it
        // is: the export does not reinterpret it in the new config's mode.
        assert!(session.has_output());
        assert_eq!(session.export_peek().unwrap().handeye, solved);
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
