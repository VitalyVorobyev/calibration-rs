//! The hand-eye stages every hand-eye problem type closes with: the
//! initialization of the hand-eye and the target pose (from seeds or the
//! closed-form solve), then the hand-eye bundle adjustment.
//!
//! The observer the robot carries (eye-in-hand) or that watches it
//! (eye-to-hand) is a rig. A single camera is the one-camera rig whose
//! frame is the camera's (`cam_se3_rig` the identity), so the single-camera
//! and the rig problem types run these stages alike. A problem type opts in
//! by implementing [`HandeyeStages`]; its `step_handeye_*` functions delegate
//! here and keep their own documentation and public types.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{CameraFix, CameraParams, Iso3};
use vision_calibration_linear::handeye::{
    HandeyeMethod, estimate_eye_in_hand, estimate_eye_to_hand,
};
use vision_calibration_optim::{
    HandeyeDataset, HandeyeEstimate, HandeyeMode, HandeyeParams, HandeyeSolveOptions, RobustLoss,
    optimize_handeye,
};

use crate::Error;
use crate::common::config::{HandeyeInitConfig, RobotPoseConfig, SolverConfig};
use crate::common::{HandeyeInitOptions, HandeyeOptimizeOptions};
use crate::planar_family::Provenance;
use crate::session::{CalibrationSession, ProblemType};

/// Typed return value of a hand-eye problem's `step_handeye_optimize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct HandeyeOptimizeResult {
    /// Mean reprojection error in pixels after the hand-eye bundle
    /// adjustment.
    pub mean_reproj_error: f64,
    /// Final solver cost of the hand-eye bundle adjustment.
    pub final_cost: f64,
}

/// A hand-eye solution of one mode: the hand-eye pose and the target pose
/// that mode fixes, named as the problem's export names them.
pub(crate) trait HandeyePoses: Copy {
    /// The solution of `mode` from its hand-eye and target pose.
    fn from_poses(mode: HandeyeMode, handeye: Iso3, target: Iso3) -> Self;
    /// The mode the solution is in.
    fn mode(&self) -> HandeyeMode;
    /// The `(hand-eye, target)` poses.
    fn poses(&self) -> (Iso3, Iso3);
    /// The names of the `(hand-eye, target)` poses of `mode`, for log notes.
    fn pose_names(mode: HandeyeMode) -> (&'static str, &'static str);
}

/// A problem type whose pipeline closes with the hand-eye stages.
pub(crate) trait HandeyeStages: ProblemType<Output = HandeyeEstimate> {
    /// The mode-tagged solution the problem's state and export hold.
    type Solution: HandeyePoses;

    /// The hand-eye initialization settings (the mode among them).
    fn handeye_init_config(config: &Self::Config) -> &HandeyeInitConfig;
    /// The solver engine settings.
    fn solver_config(config: &Self::Config) -> &SolverConfig;
    /// The robot's `base_se3_gripper` at each view.
    fn base_se3_gripper(input: &Self::Input) -> Vec<Iso3>;
    /// The observer's target pose at each view, refined by the stage before
    /// the hand-eye; `Error::NotAvailable` naming that stage when it has
    /// not run.
    fn observer_se3_target(state: &Self::State) -> Result<&[Iso3], Error>;
    /// The initial hand-eye.
    fn initial_handeye(state: &Self::State) -> Option<Self::Solution>;
    /// The initial hand-eye, mutably.
    fn initial_handeye_mut(state: &mut Self::State) -> &mut Option<Self::Solution>;
    /// The cameras the bundle adjustment starts from, checked against the
    /// configured models.
    fn handeye_cameras(
        config: &Self::Config,
        state: &Self::State,
    ) -> Result<Vec<CameraParams>, Error>;
    /// Each camera's `rig_se3_cam` the bundle adjustment starts from.
    fn rig_se3_cam(state: &Self::State) -> Result<Vec<Iso3>, Error>;
    /// The input as the bundle adjustment's dataset.
    fn handeye_dataset(input: &Self::Input, mode: HandeyeMode) -> Result<HandeyeDataset, Error>;
    /// The bundle adjustment's loss, scope and robot-pose terms.
    fn handeye_solve_options(config: &Self::Config, num_cameras: usize) -> HandeyeSolveOptions;
    /// Record the bundle adjustment's final cost and mean reprojection
    /// error.
    fn record_handeye_optimize(state: &mut Self::State, final_cost: f64, mean_reproj_error: f64);
}

/// The closed-form hand-eye estimate of `mode` by `method`: the hand-eye
/// transform and the mode's fixed target pose, from the robot poses and
/// the observer's (a camera's, or a rig's) target poses.
///
/// - `EyeInHand`: (`gripper_se3_observer`, `base_se3_target`).
/// - `EyeToHand`: (`observer_se3_base`, `gripper_se3_target`).
///
/// # Errors
///
/// [`Error::Numerical`] when the linear solve fails.
fn linear_handeye(
    mode: HandeyeMode,
    method: HandeyeMethod,
    base_se3_gripper: &[Iso3],
    observer_se3_target: &[Iso3],
    min_motion_angle_deg: f64,
) -> Result<(Iso3, Iso3), Error> {
    match mode {
        HandeyeMode::EyeInHand => estimate_eye_in_hand(
            method,
            base_se3_gripper,
            observer_se3_target,
            min_motion_angle_deg,
        )
        .map(|e| (e.gripper_se3_cam, e.base_se3_target))
        .map_err(|e| Error::numerical(format!("linear hand-eye estimation failed: {e}"))),
        HandeyeMode::EyeToHand => estimate_eye_to_hand(
            method,
            base_se3_gripper,
            observer_se3_target,
            min_motion_angle_deg,
        )
        .map(|e| (e.cam_se3_base, e.gripper_se3_target))
        .map_err(|e| Error::numerical(format!("linear gripper->target estimation failed: {e}"))),
    }
}

/// The hand-eye bundle adjustment of one fixed target with the hand-eye
/// transform free, each camera holding `fix`, the extrinsics held where
/// `fix_extrinsics`, and the robot poses refined as `robot_poses` says.
pub(crate) fn handeye_solve_opts(
    robust_loss: RobustLoss,
    fix: Vec<CameraFix>,
    fix_extrinsics: Vec<bool>,
    robot_poses: &RobotPoseConfig,
) -> HandeyeSolveOptions {
    let (refine_robot_poses, robot_rot_sigma, robot_trans_sigma, estimate_robot_sigma) =
        robot_poses.solver_terms();
    HandeyeSolveOptions {
        robust_loss,
        fix,
        fix_extrinsics,
        fix_handeye: false,
        fix_target_poses: Vec::new(),
        relax_target_poses: false,
        refine_robot_poses,
        robot_rot_sigma,
        robot_trans_sigma,
        estimate_robot_sigma,
    }
}

/// The target pose of `mode` that puts the target where the observer saw
/// it at one view, given the hand-eye.
fn target_through(
    mode: HandeyeMode,
    base_se3_gripper: &Iso3,
    observer_se3_target: &Iso3,
    handeye: &Iso3,
) -> Iso3 {
    match mode {
        // T_O_T = (T_B_G · T_G_O)⁻¹ · T_B_T  ⇒  T_B_T = T_B_G · T_G_O · T_O_T.
        HandeyeMode::EyeInHand => base_se3_gripper * handeye * observer_se3_target,
        // T_O_T = T_O_B · T_B_G · T_G_T  ⇒  T_G_T = T_B_G⁻¹ · T_O_B⁻¹ · T_O_T.
        HandeyeMode::EyeToHand => {
            base_se3_gripper.inverse() * handeye.inverse() * observer_se3_target
        }
    }
}

/// The hand-eye of `mode` that puts the target where the observer saw it
/// at one view, given the target pose.
fn handeye_through(
    mode: HandeyeMode,
    base_se3_gripper: &Iso3,
    observer_se3_target: &Iso3,
    target: &Iso3,
) -> Iso3 {
    match mode {
        // T_G_O = T_B_G⁻¹ · T_B_T · T_O_T⁻¹.
        HandeyeMode::EyeInHand => {
            base_se3_gripper.inverse() * target * observer_se3_target.inverse()
        }
        // T_O_B = T_O_T · (T_B_G · T_G_T)⁻¹.
        HandeyeMode::EyeToHand => observer_se3_target * (base_se3_gripper * target).inverse(),
    }
}

/// Initialize the hand-eye and the target pose of the configured mode.
///
/// A seeded pose is used as given. With one pose seeded the other closes
/// through the first view; with neither, the closed-form solver of
/// `handeye_init.method` gives both. The previous initialization is cleared
/// once the stage's own work starts, so a failed run leaves none behind.
///
/// # Errors
///
/// - [`Error::InvalidInput`] when `seed_mode` is not the configured mode,
///   before any work.
/// - The observer's target poses are not available (the stage before has
///   not run).
/// - The closed-form solve fails (nothing seeded).
pub(crate) fn init_handeye<P: HandeyeStages>(
    session: &mut CalibrationSession<P>,
    seed_mode: HandeyeMode,
    (seeded_handeye, seeded_target): (Option<Iso3>, Option<Iso3>),
    opts: Option<HandeyeInitOptions>,
) -> Result<P::Solution, Error> {
    session.validate()?;
    let init_config = P::handeye_init_config(&session.config);
    let mode = init_config.handeye_mode;
    if seed_mode != mode {
        return Err(Error::invalid_input(format!(
            "hand-eye seed is {seed_mode:?}, the config's handeye_init.handeye_mode is {mode:?}"
        )));
    }
    let opts = opts.unwrap_or_default();
    let min_angle = opts
        .min_motion_angle_deg
        .unwrap_or(init_config.min_motion_angle_deg);
    let method = opts.method.unwrap_or(init_config.method);
    let base_se3_gripper = P::base_se3_gripper(session.require_input()?);
    let observer_se3_target = P::observer_se3_target(&session.state)?.to_vec();

    *P::initial_handeye_mut(&mut session.state) = None;

    let (handeye_name, target_name) = P::Solution::pose_names(mode);
    let mut provenance = Provenance::default();
    provenance.note(handeye_name, seeded_handeye.is_some());
    provenance.note(target_name, seeded_target.is_some());

    let first_view = || {
        base_se3_gripper
            .first()
            .zip(observer_se3_target.first())
            .ok_or(Error::InsufficientData { need: 1, got: 0 })
    };
    let poses = match (seeded_handeye, seeded_target) {
        (Some(handeye), Some(target)) => Ok((handeye, target)),
        (Some(handeye), None) => first_view()
            .map(|(robot, observed)| (handeye, target_through(mode, robot, observed, &handeye))),
        (None, Some(target)) => first_view()
            .map(|(robot, observed)| (handeye_through(mode, robot, observed, &target), target)),
        (None, None) => linear_handeye(
            mode,
            method,
            &base_se3_gripper,
            &observer_se3_target,
            min_angle,
        ),
    };
    let (handeye, target) =
        poses.inspect_err(|e| session.log_failure("handeye_init", e.to_string()))?;

    let solution = P::Solution::from_poses(mode, handeye, target);
    *P::initial_handeye_mut(&mut session.state) = Some(solution);
    session.log_success_with_notes(
        "handeye_init",
        format!(
            "{handeye_name} |t|={:.4}m {}",
            handeye.translation.vector.norm(),
            provenance.source()
        ),
    );
    Ok(solution)
}

/// The hand-eye bundle adjustment from the initial hand-eye: the cameras,
/// the hand-eye, the target pose and the robot poses as the problem's
/// [`HandeyeStages::handeye_solve_options`] say. Sets the session's output.
///
/// # Errors
///
/// - The hand-eye was not initialized, or not in the configured mode.
/// - The cameras or rig of the stages before are not available, or the
///   camera models changed since.
/// - The solve fails.
pub(crate) fn optimize_handeye_stage<P: HandeyeStages>(
    session: &mut CalibrationSession<P>,
    opts: Option<HandeyeOptimizeOptions>,
) -> Result<HandeyeOptimizeResult, Error> {
    session.validate()?;
    let input = session.require_input()?;
    let solution = P::initial_handeye(&session.state).ok_or_else(|| {
        Error::not_available("hand-eye initialization (call step_handeye_init first)")
    })?;

    let opts = opts.unwrap_or_default();
    let config = &session.config;
    let mode = P::handeye_init_config(config).handeye_mode;
    let cameras = P::handeye_cameras(config, &session.state)?;
    let rig_se3_cam = P::rig_se3_cam(&session.state)?;
    if solution.mode() != mode {
        return Err(Error::not_available(
            "initial handeye of the configured mode (call step_handeye_init again)",
        ));
    }
    let (handeye, target) = solution.poses();

    let solve_opts = P::handeye_solve_options(config, cameras.len());
    let initial = HandeyeParams::new(cameras, rig_se3_cam, handeye, vec![target]);
    let backend_opts = P::solver_config(config).step_backend_options(
        session.control(),
        opts.max_iters,
        opts.verbosity,
    );
    let dataset = P::handeye_dataset(input, mode)?;
    let output = match optimize_handeye(dataset, initial, solve_opts, backend_opts) {
        Ok(r) => r,
        Err(e) => {
            session.log_failure("handeye_optimize", e.to_string());
            return Err(Error::from(e));
        }
    };

    let final_cost = output.report.final_cost;
    let mean_reproj_error = output.mean_reproj_error;
    P::record_handeye_optimize(&mut session.state, final_cost, mean_reproj_error);
    session.set_output(output);
    session.log_success_with_notes("handeye_optimize", format!("final_cost={final_cost:.2e}"));

    Ok(HandeyeOptimizeResult {
        mean_reproj_error,
        final_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig_handeye::{self, RigHandeyeProblem};
    use crate::single_cam_handeye::{self, SingleCamHandeyeProblem};
    use vision_calibration_core::synthetic::{planar, poses::make_iso};
    use vision_calibration_core::{
        BrownConrady5, CorrespondenceView, FxFyCxCy, PinholeCamera, RigDataset, RigView,
        RigViewObs, make_pinhole_camera, pinhole_camera_params,
    };
    use vision_calibration_optim::RobotPoseMeta;

    const MODES: [HandeyeMode; 2] = [HandeyeMode::EyeInHand, HandeyeMode::EyeToHand];

    fn camera() -> PinholeCamera {
        make_pinhole_camera(
            FxFyCxCy {
                fx: 800.0,
                fy: 790.0,
                cx: 512.0,
                cy: 384.0,
            },
            BrownConrady5::default(),
        )
    }

    /// The true `(hand-eye, target)` of `mode`.
    fn truth(mode: HandeyeMode) -> (Iso3, Iso3) {
        match mode {
            HandeyeMode::EyeInHand => (
                make_iso((0.10, -0.05, 0.02), (0.05, -0.03, 0.08)),
                make_iso((0.05, -0.08, 0.0), (-0.10, -0.08, 1.0)),
            ),
            HandeyeMode::EyeToHand => (
                make_iso((0.08, -0.06, 0.03), (0.02, 0.04, 1.2)),
                make_iso((0.03, 0.05, -0.02), (0.03, -0.02, 0.02)),
            ),
        }
    }

    /// Four stations that all rotate the gripper about the base's z axis:
    /// the closed-form solvers need two rotation axes, so they cannot run.
    fn one_axis_stations() -> Vec<Iso3> {
        (0..4)
            .map(|i| {
                let i = f64::from(i);
                make_iso((0.0, 0.0, 0.3 * i), (0.04 * i, 0.02 * i, 0.0))
            })
            .collect()
    }

    /// The observer's target pose at each station, exact.
    fn observed(mode: HandeyeMode, stations: &[Iso3]) -> Vec<Iso3> {
        let (handeye, target) = truth(mode);
        stations
            .iter()
            .map(|robot| match mode {
                HandeyeMode::EyeInHand => (robot * handeye).inverse() * target,
                HandeyeMode::EyeToHand => handeye * robot * target,
            })
            .collect()
    }

    fn board_view(observer_se3_target: &Iso3) -> CorrespondenceView {
        let board = planar::grid_points(4, 3, 0.03);
        planar::project_view_all(&camera(), observer_se3_target, &board)
            .expect("the board is in front of the camera")
    }

    /// A single-camera session past its intrinsics stage, at the truth.
    fn single_session(mode: HandeyeMode) -> CalibrationSession<SingleCamHandeyeProblem> {
        let stations = one_axis_stations();
        let observed = observed(mode, &stations);
        let views = stations
            .iter()
            .zip(&observed)
            .map(|(robot, pose)| single_cam_handeye::SingleCamHandeyeView {
                obs: board_view(pose),
                meta: RobotPoseMeta {
                    base_se3_gripper: *robot,
                },
            })
            .collect();
        let mut session = CalibrationSession::<SingleCamHandeyeProblem>::new();
        let mut config = single_cam_handeye::SingleCamHandeyeConfig::default();
        config.handeye_init.handeye_mode = mode;
        session.set_config(config).unwrap();
        let input = single_cam_handeye::SingleCamHandeyeInput::new(views).unwrap();
        session.set_input(input).unwrap();
        session.state.optimized_camera = Some(pinhole_camera_params(&camera()));
        session.state.optimized_cam_se3_target = Some(observed);
        session
    }

    /// A one-camera rig session past its rig stage, at the truth.
    fn rig_session(mode: HandeyeMode) -> CalibrationSession<RigHandeyeProblem> {
        let stations = one_axis_stations();
        let observed = observed(mode, &stations);
        let views = stations
            .iter()
            .zip(&observed)
            .map(|(robot, pose)| RigView {
                meta: RobotPoseMeta {
                    base_se3_gripper: *robot,
                },
                obs: RigViewObs {
                    cameras: vec![Some(board_view(pose))],
                },
            })
            .collect();
        let mut session = CalibrationSession::<RigHandeyeProblem>::new();
        let mut config = rig_handeye::RigHandeyeConfig::default();
        config.handeye_init.handeye_mode = mode;
        session.set_config(config).unwrap();
        session
            .set_input(RigDataset::new(views, 1).unwrap())
            .unwrap();
        session.state.rig_ba_cam_se3_rig = Some(vec![Iso3::identity()]);
        session.state.rig_ba_rig_se3_target = Some(observed);
        session
    }

    fn assert_close(got: &Iso3, want: &Iso3, what: &str) {
        let dt = (got.translation.vector - want.translation.vector).norm();
        let dr = got.rotation.angle_to(&want.rotation);
        assert!(dt < 1e-9 && dr < 1e-9, "{what}: Δt={dt:e} m, Δr={dr:e} rad");
    }

    /// The `(hand-eye, target)` seed of `mode` with the true hand-eye, the
    /// true target pose, both or neither.
    fn seeded(mode: HandeyeMode, handeye: bool, target: bool) -> (Option<Iso3>, Option<Iso3>) {
        let (h, t) = truth(mode);
        (handeye.then_some(h), target.then_some(t))
    }

    /// The single-camera step seeded with `(hand-eye, target)` of `mode`.
    fn init_single(
        session: &mut CalibrationSession<SingleCamHandeyeProblem>,
        mode: HandeyeMode,
        (h, t): (Option<Iso3>, Option<Iso3>),
    ) -> Result<(Iso3, Iso3), Error> {
        use single_cam_handeye::HandeyeManualInit as Seed;
        let seed = match mode {
            HandeyeMode::EyeInHand => Seed::EyeInHand {
                gripper_se3_cam: h,
                base_se3_target: t,
            },
            HandeyeMode::EyeToHand => Seed::EyeToHand {
                cam_se3_base: h,
                gripper_se3_target: t,
            },
        };
        let init = single_cam_handeye::step_handeye_init_with_seed(session, seed, None)?;
        Ok((*init.handeye.handeye().pose(), *init.handeye.target_pose()))
    }

    /// The rig step seeded with `(hand-eye, target)` of `mode`.
    fn init_rig(
        session: &mut CalibrationSession<RigHandeyeProblem>,
        mode: HandeyeMode,
        (h, t): (Option<Iso3>, Option<Iso3>),
    ) -> Result<(Iso3, Iso3), Error> {
        use rig_handeye::HandeyeManualInit as Seed;
        let seed = match mode {
            HandeyeMode::EyeInHand => Seed::EyeInHand {
                gripper_se3_rig: h,
                base_se3_target: t,
            },
            HandeyeMode::EyeToHand => Seed::EyeToHand {
                rig_se3_base: h,
                gripper_se3_target: t,
            },
        };
        let init = rig_handeye::step_handeye_init_with_seed(session, seed, None)?;
        Ok((*init.handeye.handeye().pose(), *init.handeye.target_pose()))
    }

    #[test]
    fn one_seeded_pose_closes_the_other_without_the_closed_form_solve() {
        for mode in MODES {
            let (handeye, target) = truth(mode);
            // The stations defeat the closed-form solve...
            let nothing = seeded(mode, false, false);
            assert!(init_single(&mut single_session(mode), mode, nothing).is_err());
            assert!(init_rig(&mut rig_session(mode), mode, nothing).is_err());
            // ...and either pose alone closes the other, in either problem.
            for (seed_handeye, seed_target) in [(true, false), (false, true)] {
                let seed = seeded(mode, seed_handeye, seed_target);
                for (problem, init) in [
                    ("single", init_single(&mut single_session(mode), mode, seed)),
                    ("rig", init_rig(&mut rig_session(mode), mode, seed)),
                ] {
                    let what = format!("{problem} {mode:?}, hand-eye seeded: {seed_handeye}");
                    let (h, t) = init.unwrap_or_else(|e| panic!("{what}: {e}"));
                    assert_close(&h, &handeye, &what);
                    assert_close(&t, &target, &what);
                }
            }
        }
    }

    #[test]
    fn a_failed_run_leaves_no_initialization() {
        for mode in MODES {
            let (both, nothing) = (seeded(mode, true, true), seeded(mode, false, false));

            let mut session = single_session(mode);
            init_single(&mut session, mode, both).unwrap();
            assert!(session.state.initial_handeye.is_some());
            assert!(init_single(&mut session, mode, nothing).is_err());
            assert!(session.state.initial_handeye.is_none(), "single {mode:?}");
            assert!(single_cam_handeye::step_handeye_optimize(&mut session, None).is_err());

            let mut session = rig_session(mode);
            init_rig(&mut session, mode, both).unwrap();
            assert!(session.state.initial_handeye.is_some());
            assert!(init_rig(&mut session, mode, nothing).is_err());
            assert!(session.state.initial_handeye.is_none(), "rig {mode:?}");
            assert!(rig_handeye::step_handeye_optimize(&mut session, None).is_err());
        }
    }

    #[test]
    fn a_seed_of_another_mode_leaves_the_initialization() {
        let mode = HandeyeMode::EyeInHand;
        let mut session = rig_session(mode);
        init_rig(&mut session, mode, seeded(mode, true, true)).unwrap();
        let before = session.state.initial_handeye;
        let other = rig_handeye::step_handeye_init_with_seed(
            &mut session,
            rig_handeye::HandeyeManualInit::unseeded(HandeyeMode::EyeToHand),
            None,
        );
        assert!(matches!(other, Err(Error::InvalidInput { .. })));
        assert_eq!(session.state.initial_handeye, before);
    }
}
