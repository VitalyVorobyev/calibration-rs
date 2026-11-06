//! Progress reports and cancellation through the Levenberg–Marquardt loop:
//! an observer sees every iteration and changes nothing, and a cancelled
//! token stops the solve at its next iteration with `Error::Cancelled`, on
//! both backends.

#![allow(missing_docs)]

use std::sync::{Arc, Mutex};

use nalgebra::{DVector, RealField};
use vision_calibration_core::synthetic::planar;
use vision_calibration_core::{
    BrownConrady5, CancelToken, FxFyCxCy, PlanarDataset, Progress, RunControl, View,
    make_pinhole_camera,
};
use vision_calibration_optim::problem::{Manifold, Noise, Problem, Residual, SolveOptions, solve};
use vision_calibration_optim::{
    BackendSolveOptions, Error, PlanarIntrinsicsEstimate, PlanarIntrinsicsParams,
    PlanarIntrinsicsSolveOptions, RobustLoss, SolverBackend, optimize_planar_intrinsics,
};

/// Each iteration reported, with its cost.
type Seen = Arc<Mutex<Vec<(usize, f64)>>>;

/// The iterations a control saw, and the control. `cancel_at` cancels the
/// token from the observer once that iteration is reported.
fn recording(cancel_at: Option<usize>) -> (RunControl, Seen) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    let token = CancelToken::new();
    let trigger = token.clone();
    let control = RunControl::new()
        .with_cancel(token)
        .with_observer(move |progress| {
            if let Progress::Iteration { iter, cost, .. } = progress {
                log.lock().unwrap().push((iter, cost));
                if cancel_at == Some(iter) {
                    trigger.cancel();
                }
            }
        });
    (control, seen)
}

/// Rosenbrock's valley as two residuals, `10 (y − x²)` and `1 − x`: from
/// `(−1.2, 1)` Levenberg–Marquardt takes a couple of dozen iterations.
struct Rosenbrock;

impl Residual for Rosenbrock {
    fn dim(&self) -> usize {
        2
    }
    fn block_dims(&self) -> Vec<usize> {
        vec![2]
    }
    fn eval<T: RealField>(&self, blocks: &[DVector<T>]) -> DVector<T> {
        let (x, y) = (blocks[0][0].clone(), blocks[0][1].clone());
        let ten = T::from_f64(10.0).unwrap();
        DVector::from_vec(vec![ten * (y - x.clone() * x.clone()), T::one() - x])
    }
}

fn rosenbrock(control: RunControl) -> Result<(Vec<f64>, usize, f64), Error> {
    let mut problem = Problem::new();
    let block = problem.add_block("xy", &[-1.2, 1.0], Manifold::Euclidean)?;
    problem.add_residual(Rosenbrock, &[block], RobustLoss::None, Noise::Stated(1.0))?;
    let mut options = SolveOptions::default();
    options.max_iters = 200;
    options.control = control;
    let solution = solve(&problem, &options)?;
    let value = solution.value(block).unwrap().to_vec();
    Ok((value, solution.report.num_iters, solution.report.final_cost))
}

#[test]
fn an_observer_sees_every_iteration_and_changes_nothing() {
    let (plain, plain_iters, plain_cost) = rosenbrock(RunControl::default()).unwrap();
    let (control, seen) = recording(None);
    let (observed, iters, cost) = rosenbrock(control).unwrap();

    // Bit-identical: the observer is a side channel.
    assert_eq!(plain, observed);
    assert_eq!((plain_iters, plain_cost.to_bits()), (iters, cost.to_bits()));

    let seen = seen.lock().unwrap();
    assert!(
        iters > 5,
        "Rosenbrock should take several iterations, took {iters}"
    );
    let counted: Vec<usize> = seen.iter().map(|&(iter, _)| iter).collect();
    assert_eq!(counted, (1..=iters).collect::<Vec<_>>());
    // The last report is the solve report's cost; accepted steps only lower it.
    assert_eq!(seen.last().unwrap().1.to_bits(), cost.to_bits());
    assert!(seen.windows(2).all(|w| w[1].1 <= w[0].1));
}

#[test]
fn a_cancelled_solve_stops_at_the_next_iteration() {
    let (control, seen) = recording(Some(3));
    let err = rosenbrock(control).unwrap_err();
    assert!(matches!(err, Error::Cancelled), "{err:?}");
    // Cancelled while iteration 3 was reported: iteration 4 never starts.
    let counted: Vec<usize> = seen.lock().unwrap().iter().map(|&(i, _)| i).collect();
    assert_eq!(counted, [1, 2, 3]);

    let token = CancelToken::new();
    token.cancel();
    let err = rosenbrock(RunControl::new().with_cancel(token)).unwrap_err();
    assert!(
        matches!(err, Error::Cancelled),
        "a cancelled token stops before iteration 1"
    );
}

fn planar_scene() -> (PlanarDataset, PlanarIntrinsicsParams) {
    let truth = make_pinhole_camera(
        FxFyCxCy {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
        },
        BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            ..BrownConrady5::default()
        },
    );
    let board = planar::grid_points(8, 6, 0.04);
    let poses = planar::poses_yaw_y_z(6, -0.2, 0.08, 0.5, 0.05);
    let views = planar::project_views_all(&truth, &board, &poses).unwrap();
    let dataset = PlanarDataset::new(views.into_iter().map(View::without_meta).collect()).unwrap();
    let start = make_pinhole_camera(
        FxFyCxCy {
            fx: 760.0,
            fy: 750.0,
            cx: 630.0,
            cy: 350.0,
        },
        BrownConrady5::default(),
    );
    (
        dataset,
        PlanarIntrinsicsParams::from_pinhole(start, poses).unwrap(),
    )
}

fn planar(backend: SolverBackend, control: RunControl) -> Result<PlanarIntrinsicsEstimate, Error> {
    let (dataset, initial) = planar_scene();
    let backend_opts = BackendSolveOptions {
        backend,
        max_iters: 100,
        min_abs_decrease: Some(1e-14),
        min_rel_decrease: Some(1e-14),
        min_error: Some(1e-20),
        control,
        ..BackendSolveOptions::default()
    };
    optimize_planar_intrinsics(
        &dataset,
        &initial,
        PlanarIntrinsicsSolveOptions::default(),
        backend_opts,
    )
}

#[test]
fn both_backends_report_and_cancel_alike() {
    for backend in [SolverBackend::TinySolver, SolverBackend::Factrs] {
        let plain = planar(backend, RunControl::default()).unwrap();
        let (control, seen) = recording(None);
        let observed = planar(backend, control).unwrap();
        assert_eq!(
            serde_json::to_string(&plain.params).unwrap(),
            serde_json::to_string(&observed.params).unwrap(),
            "{backend:?}: an observer changed the result"
        );
        assert_eq!(
            plain.report.final_cost.to_bits(),
            observed.report.final_cost.to_bits()
        );
        let iters = observed.report.num_iters;
        assert!(iters >= 4, "{backend:?}: {iters} iterations");
        let counted: Vec<usize> = seen.lock().unwrap().iter().map(|&(i, _)| i).collect();
        assert_eq!(counted, (1..=iters).collect::<Vec<_>>(), "{backend:?}");

        let (control, seen) = recording(Some(2));
        let err = planar(backend, control).unwrap_err();
        assert!(matches!(err, Error::Cancelled), "{backend:?}: {err:?}");
        assert_eq!(seen.lock().unwrap().len(), 2, "{backend:?}");
    }
}
