//! Step functions for multi-camera rig hand-eye calibration.
//!
//! This module provides step functions that operate on
//! `CalibrationSession<RigHandeyeProblem>` to perform calibration. The first
//! four are the rig stages every rig problem type shares.

use crate::Error;
use serde::{Deserialize, Serialize};
use vision_calibration_core::{CameraParams, Iso3};
use vision_calibration_optim::{HandeyeDataset, HandeyeMode, HandeyeParams, optimize_handeye};

use crate::common::{
    HandeyeInitOptions, HandeyeOptimizeOptions, HandeyeSolution, IntrinsicsInitOptions,
    IntrinsicsOptimizeOptions, RigInitOptions, RigOptimizeOptions,
};
use crate::planar_family::Provenance;
use crate::rig_family::{
    self, check_camera_models, handeye_solve_opts, inverse_poses, linear_handeye, rig_ba_fix,
    rig_ba_notes,
};
pub use crate::rig_family::{
    IntrinsicsInitResult, IntrinsicsManualInit, RigInitResult, RigManualInit, RigPoses,
};
use crate::session::CalibrationSession;

use super::problem::RigHandeyeProblem;

// ─────────────────────────────────────────────────────────────────────────────
// Manual seeds
// ─────────────────────────────────────────────────────────────────────────────

/// Manual seeds for the **hand-eye stage** of rig hand-eye calibration.
///
/// One variant per mounting, named like [`HandeyeSolution`] and tagged by
/// `mode` in JSON, so a seed reads like the export it seeds. The variant must
/// be the one of `session.config.handeye_init.handeye_mode`; any other is an
/// [`Error::InvalidInput`] before any work.
///
/// A pose left `None` is estimated: with the hand-eye seeded and the target
/// pose not, the target pose closes through the first view; otherwise the
/// closed-form solver of `handeye_init.method` provides what is missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
#[non_exhaustive]
pub enum HandeyeManualInit {
    /// The rig rides on the gripper; the target stands in the base frame.
    EyeInHand {
        /// The rig mounted on the gripper: `T_G_R`.
        gripper_se3_rig: Option<Iso3>,
        /// Target fixed in the base frame: `T_B_T`.
        base_se3_target: Option<Iso3>,
    },
    /// The rig stands in the base frame; the target rides on the gripper.
    EyeToHand {
        /// The base's pose in the rig: `T_R_B`.
        rig_se3_base: Option<Iso3>,
        /// Target attached to the gripper: `T_G_T`.
        gripper_se3_target: Option<Iso3>,
    },
}

impl HandeyeManualInit {
    /// The seed of `mode` with nothing seeded: both poses estimated.
    pub fn unseeded(mode: HandeyeMode) -> Self {
        match mode {
            HandeyeMode::EyeInHand => Self::EyeInHand {
                gripper_se3_rig: None,
                base_se3_target: None,
            },
            HandeyeMode::EyeToHand => Self::EyeToHand {
                rig_se3_base: None,
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
                gripper_se3_rig,
                base_se3_target,
            } => (gripper_se3_rig, base_se3_target),
            Self::EyeToHand {
                rig_se3_base,
                gripper_se3_target,
            } => (rig_se3_base, gripper_se3_target),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step Results
// ─────────────────────────────────────────────────────────────────────────────

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
    /// `[view][cam] -> Option<Iso3>` (`cam_se3_target`). This is what the
    /// unseeded rig init consumes; seeded rig inits (e.g.
    /// [`crate::device_seed::rig_layout_seed`]) should anchor on these, not
    /// on the coarser init-time poses.
    pub cam_se3_target: Vec<Vec<Option<Iso3>>>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeInitResult {
    /// The initial hand-eye and target pose of the configured mode; both
    /// poses are always present.
    pub handeye: HandeyeSolution,
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
/// independently (the tilted variant for a Scheimpflug camera). The linear
/// stage fits only the distortion terms each camera's solves estimate. See
/// `rig_extrinsics::step_intrinsics_init_all_with_seed` for the seed
/// semantics. `manual` is the seed of this call: it overrides the config's
/// `manual_init`, which [`step_intrinsics_init_all`] and `run_calibration`
/// use.
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
    rig_family::init_intrinsics(session, manual, opts)
}

/// Initialize intrinsics for all cameras, seeded by the config's
/// `manual_init` where it has one (Zhang's per camera for everything else).
///
/// Convenience wrapper around [`step_intrinsics_init_all_with_seed`] with the
/// config's seed (the default, a full auto-init, when it has none).
pub fn step_intrinsics_init_all(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<IntrinsicsInitOptions>,
) -> Result<IntrinsicsInitResult, Error> {
    let manual = session.config.manual_init.clone().unwrap_or_default();
    step_intrinsics_init_all_with_seed(session, manual, opts)
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
    let estimate = rig_family::optimize_intrinsics(session, opts)?;
    Ok(IntrinsicsOptimizeResult {
        cameras: estimate.cameras,
        per_cam_reproj_errors: estimate.per_cam_reproj_errors,
        cam_se3_target: estimate.cam_se3_target,
    })
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
    rig_family::init_rig(session, manual, opts)
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
    let output = rig_family::optimize_rig(session, opts)?;
    let notes = rig_ba_notes(&output);
    let cam_se3_rig = inverse_poses(&output.params.rig_se3_cam);
    let result = RigOptimizeResult {
        mean_reproj_error: output.mean_reproj_error,
        per_cam_reproj_errors: output.per_cam_reproj_errors,
        cam_se3_rig: cam_se3_rig.clone(),
        rig_se3_target: output.params.rig_se3_target.clone(),
    };
    session.state.rig_ba_cam_se3_rig = Some(cam_se3_rig);
    session.state.rig_ba_rig_se3_target = Some(output.params.rig_se3_target);
    session.state.rig.cameras = Some(output.params.cameras);
    session.log_success_with_notes("rig_optimize", notes);
    Ok(result)
}

/// Initialize hand-eye transform from robot poses and rig poses.
///
/// Uses the closed-form solver of `handeye_init.method`.
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
    let handeye_mode = session.config.handeye_init.handeye_mode;
    if manual.mode() != handeye_mode {
        return Err(Error::invalid_input(format!(
            "hand-eye seed is {:?}, the config's handeye_init.handeye_mode is {handeye_mode:?}",
            manual.mode()
        )));
    }
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
    let method = opts.method.unwrap_or(session.config.handeye_init.method);

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

    let (seeded_handeye, seeded_target) = manual.poses();
    let (handeye_label, target_label) = match handeye_mode {
        HandeyeMode::EyeInHand => ("gripper_se3_rig", "base_se3_target"),
        HandeyeMode::EyeToHand => ("rig_se3_base", "gripper_se3_target"),
    };
    let mut provenance = Provenance::default();
    provenance.note(handeye_label, seeded_handeye.is_some());
    provenance.note(target_label, seeded_target.is_some());

    // The linear solve gives the hand-eye and the mode's target pose
    // together; a seeded hand-eye closes the target pose through the first view.
    let (handeye, estimated_target_pose) = match seeded_handeye {
        Some(t) => (t, None),
        None => {
            let (handeye, target) = linear_handeye(
                handeye_mode,
                method,
                &robot_poses,
                &rig_se3_target,
                min_angle,
            )
            .inspect_err(|e| session.log_failure("handeye_init", e.to_string()))?;
            (handeye, Some(target))
        }
    };

    let target_pose = seeded_target.unwrap_or_else(|| {
        estimated_target_pose.unwrap_or_else(|| match handeye_mode {
            HandeyeMode::EyeInHand => robot_poses[0] * handeye * rig_se3_target[0],
            HandeyeMode::EyeToHand => {
                // handeye = T_R_B. Chain: T_R_T = T_R_B * T_B_G * T_G_T.
                // T_G_T = (T_B_G)^-1 * (T_R_B)^-1 * T_R_T = robot_poses[0]^-1 * handeye^-1 * T_R_T.
                robot_poses[0].inverse() * handeye.inverse() * rig_se3_target[0]
            }
        })
    });

    let solution = HandeyeSolution::new(handeye_mode, handeye, target_pose);
    session.state.initial_handeye = Some(solution);

    session.log_success_with_notes(
        "handeye_init",
        format!(
            "translation_norm={:.4}m {}",
            handeye.translation.vector.norm(),
            provenance.source()
        ),
    );

    Ok(HandeyeInitResult { handeye: solution })
}

/// Initialize hand-eye transform using the closed-form solver of
/// `handeye_init.method`.
///
/// Convenience wrapper around [`step_handeye_init_with_seed`] with default seeds.
pub fn step_handeye_init(
    session: &mut CalibrationSession<RigHandeyeProblem>,
    opts: Option<HandeyeInitOptions>,
) -> Result<HandeyeInitResult, Error> {
    let manual = HandeyeManualInit::unseeded(session.config.handeye_init.handeye_mode);
    step_handeye_init_with_seed(session, manual, opts)
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
    let input = session.require_input()?;

    if !session.state.has_handeye_init() {
        return Err(Error::not_available(
            "hand-eye initialization (call step_handeye_init first)",
        ));
    }

    let opts = opts.unwrap_or_default();
    let config = &session.config;

    let cameras = session
        .state
        .rig
        .cameras
        .clone()
        .ok_or_else(|| Error::not_available("per-camera intrinsics"))?;
    check_camera_models(&cameras, |i| config.camera_for(i).model)?;
    let cam_se3_rig = session
        .state
        .rig_ba_cam_se3_rig
        .as_ref()
        .ok_or_else(|| Error::not_available("cam_se3_rig from rig BA"))?;
    let solution = session
        .state
        .initial_handeye
        .ok_or_else(|| Error::not_available("initial handeye"))?;
    if solution.mode() != config.handeye_init.handeye_mode {
        return Err(Error::not_available(
            "initial handeye of the configured mode (call step_handeye_init again)",
        ));
    }
    let (handeye, mode_target_pose) = (*solution.handeye().pose(), *solution.target_pose());

    let refine = &config.handeye_optimize;
    let fix = (0..input.num_cameras)
        .map(|i| {
            rig_ba_fix(
                &config.camera_for(i),
                refine.refine_intrinsics,
                refine.refine_sensor,
            )
        })
        .collect();
    let fix_extrinsics: Vec<bool> = if refine.refine_cam_se3_rig {
        (0..input.num_cameras)
            .map(|i| i == config.rig.reference_camera_idx)
            .collect()
    } else {
        vec![true; input.num_cameras]
    };

    let initial = HandeyeParams::new(
        cameras,
        inverse_poses(cam_se3_rig),
        handeye,
        vec![mode_target_pose],
    );
    let solve_opts = handeye_solve_opts(
        refine.reprojection_loss,
        fix,
        fix_extrinsics,
        &refine.robot_poses,
    );
    let backend_opts =
        config
            .solver
            .step_backend_options(session.control(), opts.max_iters, opts.verbosity);
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
    session.begin_step("intrinsics_init_all")?;
    let _ = step_intrinsics_init_all(session, None)?;
    session.begin_step("intrinsics_optimize_all")?;
    let _ = step_intrinsics_optimize_all(session, None)?;
    session.begin_step("rig_init")?;
    let _ = step_rig_init(session, None)?;
    session.begin_step("rig_optimize")?;
    let _ = step_rig_optimize(session, None)?;
    session.begin_step("handeye_init")?;
    let _ = step_handeye_init(session, None)?;
    session.begin_step("handeye_optimize")?;
    let _ = step_handeye_optimize(session, None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::problem::RigHandeyeInput;
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, DistortionParams, FxFyCxCy, PinholeCamera, Pt2, Pt3,
        RigDataset, RigView, RigViewObs, make_pinhole_camera,
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

        assert!(session.state.rig.has_per_cam_intrinsics());
        let cameras = session.state.rig.cameras.as_ref().unwrap();
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

        assert!(session.state.rig.has_per_cam_intrinsics());

        session.set_input(make_test_input()).unwrap();
        assert!(!session.state.rig.has_per_cam_intrinsics());
    }

    #[test]
    fn set_config_keeps_output_in_the_mode_it_was_solved_in() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        session.set_input(make_test_input()).unwrap();
        run_calibration(&mut session).unwrap();

        assert!(session.has_output());
        let solved = session.export_peek().unwrap().handeye;
        assert_eq!(solved.mode(), HandeyeMode::EyeInHand);

        session
            .set_config(super::super::problem::RigHandeyeConfig {
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

        // The output stays and still reads in the mode it was solved in.
        assert!(session.has_output());
        assert_eq!(session.export_peek().unwrap().handeye, solved);
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
                        DistortionParams::BrownConrady5 {
                            params: BrownConrady5::default(),
                        };
                        2
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
    fn step_intrinsics_init_all_uses_the_configured_seed_and_an_explicit_seed_overrides_it() {
        let seed = |fx: f64| IntrinsicsManualInit {
            per_cam_intrinsics: Some(vec![
                FxFyCxCy {
                    fx,
                    fy: fx,
                    cx: 640.0,
                    cy: 360.0,
                };
                2
            ]),
            per_cam_distortion: Some(vec![
                DistortionParams::BrownConrady5 {
                    params: BrownConrady5::default(),
                };
                2
            ]),
            per_cam_sensors: None,
        };
        let session_with = |manual_init| {
            let mut session = CalibrationSession::<RigHandeyeProblem>::new();
            session.set_input(make_test_input()).unwrap();
            session
                .set_config(super::super::problem::RigHandeyeConfig {
                    manual_init,
                    ..Default::default()
                })
                .unwrap();
            session
        };

        // The configured seed reaches the step.
        let mut session = session_with(Some(seed(805.0)));
        step_intrinsics_init_all(&mut session, None).unwrap();
        let notes = session.log()[0].notes.clone().expect("seed provenance");
        assert!(
            notes.contains("manual: per_cam_intrinsics, per_cam_distortion"),
            "{notes}"
        );

        // An explicit seed overrides the config's.
        let mut session = session_with(Some(seed(805.0)));
        let init =
            step_intrinsics_init_all_with_seed(&mut session, IntrinsicsManualInit::default(), None)
                .unwrap();
        let notes = session.log()[0].notes.clone().expect("provenance");
        assert!(!notes.contains("manual: per_cam_intrinsics"), "{notes}");
        assert_eq!(init.cameras.len(), 2);
    }

    #[test]
    fn a_seed_of_the_other_mode_is_an_invalid_input_before_any_work() {
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
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
        let seed = HandeyeManualInit::EyeInHand {
            gripper_se3_rig: Some(Iso3::identity()),
            base_se3_target: None,
        };
        let json = serde_json::to_value(&seed).unwrap();
        assert_eq!(json["mode"], "eye_in_hand");
        let back: HandeyeManualInit = serde_json::from_value(json).unwrap();
        assert_eq!(back, seed);
        let bare: HandeyeManualInit = serde_json::from_str(r#"{"mode":"eye_to_hand"}"#).unwrap();
        assert_eq!(bare, HandeyeManualInit::unseeded(HandeyeMode::EyeToHand));
    }

    #[test]
    fn step_set_handeye_init_eye_to_hand_recovers_target_on_gripper() {
        // Regression test: in EyeToHand mode the
        // auto-derive of the target pose from a manual `handeye` seed must
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
                    ..Default::default()
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
        // target-pose auto-derive arithmetic from DLT precision.
        let manual = HandeyeManualInit::EyeToHand {
            rig_se3_base: Some(t_r_b),
            gripper_se3_target: None,
        };
        step_handeye_init_with_seed(&mut session, manual, None).unwrap();

        let recovered = *session.state.initial_handeye.unwrap().target_pose();
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
