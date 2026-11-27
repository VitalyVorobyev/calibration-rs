//! Why a solve stopped, through the public result of `problem::solve`, on
//! both linearization engines: convergence by a stopping rule, the same
//! rule met on the last allowed iteration, the iteration budget, a
//! coordinate held on its bound and nothing to solve. A cancelled solve and a residual
//! that is not differentiable at the parameters are still errors, and a
//! report written without the reason reads back as unknown.

#![allow(missing_docs)]

use nalgebra::{DVector, RealField};
use vision_calibration_core::synthetic::planar::grid_points;
use vision_calibration_core::synthetic::poses::{seeded_tilts, tilted_board_poses};
use vision_calibration_core::{
    BrownConrady5, CameraFix, CameraParams, CancelToken, DistortionParams, FxFyCxCy, Iso3,
    ProjectionParams, Pt2, Pt3, RunControl, SensorParams, Uncertainty, UncertaintyUnavailable,
};
use vision_calibration_optim::problem::{
    Manifold, Noise, PixelNoise, PoseChain, Problem, Residual, Solution, SolveOptions, TargetPoint,
    solve,
};
use vision_calibration_optim::{Error, RobustLoss, SolveReport, SolveTermination, SolverBackend};

const BACKENDS: [SolverBackend; 2] = [SolverBackend::TinySolver, SolverBackend::Factrs];

/// A deterministic standard normal draw for `key`.
fn gauss(key: u64) -> f64 {
    let mix = |mut z: u64| {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let uniform = |z: u64| ((mix(z) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let (u1, u2) = (uniform(2 * key), uniform(2 * key + 1));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn camera(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64) -> CameraParams {
    CameraParams::new(
        ProjectionParams::Pinhole,
        DistortionParams::BrownConrady5 {
            params: BrownConrady5 {
                k1,
                iters: 8,
                ..BrownConrady5::default()
            },
        },
        SensorParams::Identity,
        FxFyCxCy { fx, fy, cx, cy },
    )
}

/// A planar calibration: eight views of a board seen by a distorted
/// camera, with 0.3 px of pixel noise, started off the true camera.
fn calibration(fix: impl Fn(&CameraParams) -> CameraFix, fix_poses: bool) -> Problem {
    let truth = camera(820.0, 810.0, 640.0, 360.0, -0.12);
    let start = camera(845.0, 795.0, 646.0, 355.0, 0.0);
    let model = truth.build().unwrap();
    let board = grid_points(9, 6, 0.03);
    let poses: Vec<Iso3> = tilted_board_poses(&seeded_tilts(8, 0.4, 7), (0.12, 0.075), 0.45, 0.02);

    let mut problem = Problem::new();
    let cam = problem.add_camera(&start, &fix(&start)).unwrap();
    for (v, pose) in poses.iter().enumerate() {
        let block = problem.add_pose(format!("pose{v}"), pose).unwrap();
        if fix_poses {
            problem.fix(block).unwrap();
        }
        for (i, point) in board.iter().enumerate() {
            let q = model.project_point(&pose.transform_point(point)).unwrap();
            let key = 1000 * v as u64 + i as u64;
            let pixel = Pt2::new(q.x + 0.3 * gauss(2 * key), q.y + 0.3 * gauss(2 * key + 1));
            problem
                .add_reprojection(
                    cam,
                    PoseChain::Pose(block),
                    TargetPoint::Known(*point),
                    pixel,
                    PixelNoise::Estimated,
                    RobustLoss::None,
                )
                .unwrap();
        }
    }
    problem
}

fn run(problem: &Problem, backend: SolverBackend, max_iters: usize) -> Solution {
    let mut options = SolveOptions::default();
    options.backend = backend;
    options.max_iters = max_iters;
    solve(problem, &options).unwrap()
}

fn reason(solution: &Solution) -> &UncertaintyUnavailable {
    match &solution.uncertainty {
        Uncertainty::Unavailable { reason } => reason,
        Uncertainty::Available(_) => panic!("the covariance is available"),
    }
}

#[test]
fn a_converged_solve_names_its_rule_even_on_its_last_iteration() {
    let problem = calibration(|_| CameraFix::default(), false);
    for backend in BACKENDS {
        let free = run(&problem, backend, 100);
        let rule = free.report.termination.expect("a solve records its reason");
        let n = free.report.num_iters;
        assert!(
            rule.is_converged() && rule != SolveTermination::MaxIters,
            "{backend:?}: {rule:?}"
        );
        assert!(n >= 3, "{backend:?}: {n} iterations");
        assert!(
            matches!(free.uncertainty, Uncertainty::Available(_)),
            "{backend:?}: a converged solve has its covariance"
        );

        // The rule is met on iteration n: a budget of exactly n is still
        // convergence, the same solve to the bit.
        let exact = run(&problem, backend, n);
        assert_eq!(exact.report.termination, Some(rule), "{backend:?}");
        assert_eq!(exact.report.num_iters, n, "{backend:?}");
        assert_eq!(
            exact.report.final_cost.to_bits(),
            free.report.final_cost.to_bits(),
            "{backend:?}"
        );
        assert!(matches!(exact.uncertainty, Uncertainty::Available(_)));

        // One iteration short, the budget ends it.
        let short = run(&problem, backend, n - 1);
        assert_eq!(
            short.report.termination,
            Some(SolveTermination::MaxIters),
            "{backend:?}"
        );
        assert_eq!(short.report.num_iters, n - 1, "{backend:?}");
        assert_eq!(
            reason(&short),
            &UncertaintyUnavailable::NotConverged,
            "{backend:?}"
        );
    }
}

#[test]
fn nothing_free_is_all_parameters_fixed() {
    let problem = calibration(|c| CameraFix::all(c.spec().unwrap()), true);
    for backend in BACKENDS {
        let solution = run(&problem, backend, 100);
        assert_eq!(
            solution.report.termination,
            Some(SolveTermination::AllParametersFixed),
            "{backend:?}"
        );
        assert_eq!(solution.report.num_iters, 0, "{backend:?}");
    }
}

/// A point held on its bound, `x ≤ 0`, by a fixed camera that sees it a
/// thousand metres to the right: every step is clamped back onto the
/// bound. With the coordinate held there is no step left to the minimum the
/// bound allows, so the solve stops on its first iteration at the bound.
#[test]
fn a_coordinate_held_on_its_bound_is_at_its_minimum() {
    let cam_params = camera(820.0, 810.0, 640.0, 360.0, 0.0);
    let mut problem = Problem::new();
    let cam = problem
        .add_camera(&cam_params, &CameraFix::all(cam_params.spec().unwrap()))
        .unwrap();
    let pose = problem
        .add_pose("pose", &Iso3::translation(0.0, 0.0, 1.0))
        .unwrap();
    problem.fix(pose).unwrap();
    let point = problem
        .add_block("point", &[0.0, 0.0, 0.0], Manifold::Euclidean)
        .unwrap();
    problem.fix_components(point, &[1, 2]).unwrap();
    problem.set_bounds(point, 0, -1.0, 0.0).unwrap();
    let seen = cam_params
        .build()
        .unwrap()
        .project_point(&Pt3::new(1000.0, 0.0, 1.0))
        .unwrap();
    problem
        .add_reprojection(
            cam,
            PoseChain::Pose(pose),
            TargetPoint::Block(point),
            seen,
            PixelNoise::Estimated,
            RobustLoss::None,
        )
        .unwrap();

    for backend in BACKENDS {
        let solution = run(&problem, backend, 100);
        assert_eq!(
            solution.report.termination,
            Some(SolveTermination::SigmaStep),
            "{backend:?}"
        );
        assert_eq!(solution.report.num_iters, 1, "{backend:?}");
        assert_eq!(solution.value(point).unwrap(), [0.0, 0.0, 0.0]);
        assert!(
            matches!(reason(&solution), UncertaintyUnavailable::AtBound { .. }),
            "{backend:?}: {:?}",
            solution.uncertainty
        );
    }
}

#[test]
fn a_cancelled_solve_is_still_an_error() {
    let problem = calibration(|_| CameraFix::default(), false);
    for backend in BACKENDS {
        let token = CancelToken::new();
        token.cancel();
        let mut options = SolveOptions::default();
        options.backend = backend;
        options.control = RunControl::new().with_cancel(token);
        let err = solve(&problem, &options).unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{backend:?}: {err:?}");
    }
}

#[test]
fn the_reason_round_trips_and_an_old_report_reads_as_unknown() {
    let old: SolveReport = serde_json::from_str(r#"{"final_cost": 1.5, "num_iters": 7}"#).unwrap();
    assert_eq!(old.termination, None);
    assert_eq!(old.num_iters, 7);
    // Unknown stays unknown: no key is written for it.
    assert_eq!(
        serde_json::to_value(&old).unwrap(),
        serde_json::json!({"final_cost": 1.5, "num_iters": 7})
    );

    let problem = calibration(|_| CameraFix::default(), false);
    let report = run(&problem, SolverBackend::TinySolver, 2).report;
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(json["termination"], "max_iters");
    let back: SolveReport = serde_json::from_value(json).unwrap();
    assert_eq!(back.termination, Some(SolveTermination::MaxIters));

    for (reason, wire) in [
        (SolveTermination::RelativeStep, "relative_step"),
        (SolveTermination::MinError, "min_error"),
        (SolveTermination::MinAbsDecrease, "min_abs_decrease"),
        (SolveTermination::MinRelDecrease, "min_rel_decrease"),
        (SolveTermination::NoAcceptedStep, "no_accepted_step"),
        (SolveTermination::MaxIters, "max_iters"),
        (SolveTermination::AllParametersFixed, "all_parameters_fixed"),
        (SolveTermination::SigmaStep, "sigma_step"),
        (SolveTermination::StalledStep, "stalled_step"),
    ] {
        assert_eq!(serde_json::to_value(reason).unwrap(), wire);
    }
}

#[test]
fn a_cancelled_solve_with_nothing_free_is_an_error() {
    let problem = calibration(|c| CameraFix::all(c.spec().unwrap()), true);
    for backend in BACKENDS {
        let token = CancelToken::new();
        token.cancel();
        let mut options = SolveOptions::default();
        options.backend = backend;
        options.control = RunControl::new().with_cancel(token);
        let err = solve(&problem, &options).unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{backend:?}: {err:?}");
    }
}

/// `√x − 1`, started at `x = 0`: the residual is finite there, its
/// derivative `1 / (2√x)` is not.
struct SqrtShift;

impl Residual for SqrtShift {
    fn dim(&self) -> usize {
        1
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![1]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        DVector::from_element(1, blocks[0][0].clone().sqrt() - T::one())
    }
}

/// On tiny-solver, which zeroes a Jacobian entry that is not finite; factrs
/// passes it on to the shared loop, which rejects it, and runs only the
/// library's residuals.
#[test]
fn a_residual_not_differentiable_at_the_start_is_an_error() {
    let mut problem = Problem::new();
    let x = problem.add_block("x", &[0.0], Manifold::Euclidean).unwrap();
    problem
        .add_residual(SqrtShift, &[x], RobustLoss::None, Noise::Stated(1.0))
        .unwrap();
    let err = solve(&problem, &SolveOptions::default()).unwrap_err();
    assert!(
        matches!(&err, Error::Numerical(message) if message.contains("not differentiable")),
        "{err:?}"
    );
}
