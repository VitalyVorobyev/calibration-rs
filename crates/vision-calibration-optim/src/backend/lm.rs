//! The Levenberg–Marquardt loop shared by every backend.
//!
//! A backend supplies a [`LinearizationEngine`]: the robust objective, the
//! loss-corrected residual and Jacobian, and the retraction onto its
//! parameter manifolds. The loop itself — Jacobi column scaling, a clamped
//! Marquardt diagonal, the Ceres gain ratio, damping retries and the stopping
//! rules — is written once, so backends differ only in how they linearize.

use crate::backend::normal_equations::{NormalEquations, jacobi_scaling, unscale};
use crate::backend::{BackendSolveOptions, SolveTermination};
use faer::sparse::SparseColMat;
use nalgebra::DVector;
use vision_calibration_core::Progress;

const LM_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;
const LM_MIN_DAMPING: f64 = 1e-12;
const LM_MAX_STEP_ATTEMPTS: usize = 32;
/// Stop once an accepted step is this small relative to the parameters
/// (`‖dx‖ ≤ ε(‖x‖ + ε)`, Ceres' `parameter_tolerance`, `‖x‖` over the free
/// blocks, [`LinearizationEngine::norm`]): it can no longer move them
/// appreciably.
const LM_RELATIVE_STEP_TOLERANCE: f64 = 1e-8;

/// Defaults for the [`BackendSolveOptions`] thresholds left as `None`.
const DEFAULT_MIN_ABS_DECREASE: f64 = 1e-5;
const DEFAULT_MIN_REL_DECREASE: f64 = 1e-5;
const DEFAULT_MIN_ERROR: f64 = 1e-10;

/// What the loop needs from a backend.
///
/// The local model at `x` is `‖r̃ + J̃·dx‖²`, where `(r̃, J̃)` is what
/// [`linearize`](Self::linearize) returns: the residual and Jacobian with the
/// robust loss folded in, so the model's first-order change matches that of
/// [`cost`](Self::cost).
pub(crate) trait LinearizationEngine {
    /// Parameter values.
    type State: Clone;

    /// The free tangent dimension: the Jacobian's columns less any held
    /// fixed. Zero leaves nothing to solve.
    fn free_dim(&self) -> usize;

    /// The robust objective `Σᵢ ρᵢ(‖rᵢ‖²)` at `x`.
    fn cost(&self, x: &Self::State) -> f64;

    /// Loss-corrected residual (one column) and Jacobian at `x`.
    ///
    /// The Jacobian should have the same sparsity pattern at every `x`, with
    /// each column's rows in ascending order: the normal-equation plan is
    /// built for one pattern and rebuilt whenever it changes.
    fn linearize(&self, x: &Self::State) -> (faer::Mat<f64>, SparseColMat<usize, f64>);

    /// `x ⊕ dx`: the step applied through each block's retraction, with
    /// parameter bounds enforced.
    fn retract(&self, x: &Self::State, dx: &DVector<f64>) -> Self::State;

    /// `‖x‖` over the parameter blocks the solve can move (used by a
    /// residual, with a free coordinate), summed in a fixed order.
    ///
    /// A held block is left out, as Ceres leaves constant blocks out of the
    /// program it minimizes: it is not part of `x`. Counted, a held block of
    /// large values (a camera's intrinsics in pixels) would set the scale
    /// the relative-step rule measures the free parameters' steps against,
    /// and stop a solve of a few small parameters (a laser plane under a
    /// held camera) before its first step.
    fn norm(&self, x: &Self::State) -> f64;
}

/// Why a solve returned no solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LmFailure {
    /// The cost at the initial parameters is not finite.
    NonFiniteCost,
    /// The residual or its Jacobian has a non-finite entry at the parameters
    /// of this (1-based) iteration: a residual is not differentiable there.
    NonFiniteLinearization {
        /// The iteration that linearized.
        iteration: usize,
    },
    /// The normal equations cannot be allocated.
    Failed,
    /// The run control's token was cancelled.
    Cancelled,
}

/// Outcome of a solve.
pub(crate) struct LmSolution<S> {
    /// Parameters at the last accepted step.
    pub state: S,
    /// The robust objective `Σᵢ ρᵢ(‖rᵢ‖²)` at `state`.
    pub cost: f64,
    /// Outer iterations executed.
    pub num_iters: usize,
    /// The stopping rule that ended the loop, or the iteration budget.
    pub termination: SolveTermination,
}

/// Minimize `engine.cost` from `x0`.
///
/// `opts.control` is polled before anything is evaluated (so a cancelled
/// solve fails even with nothing to solve), before every outer iteration and
/// before the solution is returned, one atomic load each; it is told each
/// finished iteration. Neither touches the arithmetic, so a solve with a
/// control is bit-identical to one without.
///
/// # Errors
///
/// When there is something to optimize: [`LmFailure::NonFiniteCost`] when the
/// initial cost is not finite, [`LmFailure::NonFiniteLinearization`] when a
/// linearization is (no step can be scaled or solved from it, so the loop
/// would stop at the parameters and call them a minimum), and
/// [`LmFailure::Failed`] when the normal equations cannot be allocated;
/// [`LmFailure::Cancelled`] when the control's token is cancelled.
pub(crate) fn levenberg_marquardt<E: LinearizationEngine>(
    engine: &E,
    x0: E::State,
    opts: &BackendSolveOptions,
) -> Result<LmSolution<E::State>, LmFailure> {
    let control = &opts.control;
    let max_iters = opts.max_iters;
    let verbosity = opts.verbosity;
    let min_abs_decrease = opts.min_abs_decrease.unwrap_or(DEFAULT_MIN_ABS_DECREASE);
    let min_rel_decrease = opts.min_rel_decrease.unwrap_or(DEFAULT_MIN_REL_DECREASE);
    let min_error = opts.min_error.unwrap_or(DEFAULT_MIN_ERROR);

    if control.is_cancelled() {
        return Err(LmFailure::Cancelled);
    }
    let mut x = x0;
    let mut current_cost = engine.cost(&x);
    if engine.free_dim() == 0 {
        return Ok(LmSolution {
            state: x,
            cost: current_cost,
            num_iters: 0,
            termination: SolveTermination::AllParametersFixed,
        });
    }
    if !current_cost.is_finite() {
        return Err(LmFailure::NonFiniteCost);
    }

    let mut scaling: Option<Vec<f64>> = None;
    let mut normal: Option<NormalEquations> = None;
    let mut damping = 1.0 / LM_INITIAL_TRUST_REGION_RADIUS;

    let mut num_iters = 0usize;
    // A stopping rule met on the last allowed iteration replaces this.
    let mut termination = SolveTermination::MaxIters;
    for outer_iter in 0..max_iters {
        if control.is_cancelled() {
            return Err(LmFailure::Cancelled);
        }
        num_iters = outer_iter + 1;
        let last_cost = current_cost;
        let (residuals, jac) = engine.linearize(&x);
        if !residuals
            .col(0)
            .iter()
            .chain(jac.val())
            .all(|v| v.is_finite())
        {
            return Err(LmFailure::NonFiniteLinearization {
                iteration: num_iters,
            });
        }

        let scaling = scaling.get_or_insert_with(|| jacobi_scaling(jac.as_ref()));
        let normal = match &mut normal {
            Some(normal) if normal.fits(jac.symbolic()) => normal,
            slot => slot.insert(NormalEquations::new(jac.symbolic()).ok_or(LmFailure::Failed)?),
        };
        normal.assemble(jac.as_ref(), &residuals, scaling);

        let x_norm = engine.norm(&x);
        let mut accepted = false;
        let mut step_negligible = false;
        for step_attempt in 0..LM_MAX_STEP_ATTEMPTS {
            let Some(lm_step) = normal.solve_damped(damping) else {
                damping = (damping * 2.0).max(LM_MIN_DAMPING);
                continue;
            };
            let dx = unscale(&lm_step, scaling);
            if !dx.iter().all(|v| v.is_finite()) {
                damping = (damping * 2.0).max(LM_MIN_DAMPING);
                continue;
            }
            if step_attempt == 0
                && dx.norm() <= LM_RELATIVE_STEP_TOLERANCE * (x_norm + LM_RELATIVE_STEP_TOLERANCE)
            {
                step_negligible = true;
                break;
            }

            let x_new = engine.retract(&x, &dx);
            let new_cost = engine.cost(&x_new);
            let actual_cost_change = current_cost - new_cost;
            let predicted_cost_change = normal.model_decrease(&lm_step);
            let rho = actual_cost_change / predicted_cost_change;

            if rho.is_finite() && rho > 0.0 && predicted_cost_change > 0.0 && new_cost.is_finite() {
                x = x_new;
                current_cost = new_cost;
                let tmp = 2.0 * rho - 1.0;
                damping *= (1.0_f64 / 3.0).max(1.0 - tmp * tmp * tmp);
                accepted = true;
                if verbosity > 1 {
                    println!(
                        "lm iter={outer_iter} attempt={step_attempt} cost={current_cost:.6e} rho={rho:.3e} damping={damping:.3e}"
                    );
                }
                break;
            }

            damping = (damping * 2.0).max(LM_MIN_DAMPING);
        }
        // The robust objective is Σρ; reports use the solve report's ½Σρ.
        control.report(Progress::iteration(
            num_iters,
            max_iters,
            0.5 * current_cost,
        ));

        if step_negligible {
            if verbosity > 0 {
                println!("lm stopped: relative step below tolerance");
            }
            termination = SolveTermination::RelativeStep;
            break;
        }
        if !accepted {
            // No step decreases the cost, however small: a minimum to
            // working precision.
            if verbosity > 0 {
                println!(
                    "lm stopped: no accepted step after {LM_MAX_STEP_ATTEMPTS} damping retries"
                );
            }
            termination = SolveTermination::NoAcceptedStep;
            break;
        }

        let abs_decrease = (last_cost - current_cost).abs();
        let stop = if current_cost < min_error {
            Some(SolveTermination::MinError)
        } else if abs_decrease < min_abs_decrease {
            Some(SolveTermination::MinAbsDecrease)
        } else if last_cost > 0.0 && abs_decrease / last_cost < min_rel_decrease {
            Some(SolveTermination::MinRelDecrease)
        } else {
            None
        };
        if let Some(stop) = stop {
            termination = stop;
            break;
        }
    }

    // A token cancelled while the last iteration was reported stops this
    // solve too: a cancelled solve never returns a solution.
    if control.is_cancelled() {
        return Err(LmFailure::Cancelled);
    }
    Ok(LmSolution {
        state: x,
        cost: current_cost,
        num_iters,
        termination,
    })
}

#[cfg(test)]
mod tests {
    //! Each stopping rule, on a residual `x − a` whose every quantity is
    //! known: each test sets the thresholds so that one rule, and only it,
    //! is met.

    use super::*;
    use faer::sparse::Triplet;

    /// `r(x) = x − target` in one coordinate; with `finite_only_at` set, the
    /// cost anywhere else is NaN, so no step is ever accepted.
    struct Shift {
        target: f64,
        finite_only_at: Option<f64>,
        dim: usize,
        /// The Jacobian's one entry, `1` but for a non-finite derivative.
        slope: f64,
        /// With this set, the slope is NaN anywhere but here.
        differentiable_only_at: Option<f64>,
    }

    impl Shift {
        fn to(target: f64) -> Self {
            Self {
                target,
                finite_only_at: None,
                dim: 1,
                slope: 1.0,
                differentiable_only_at: None,
            }
        }
    }

    impl LinearizationEngine for Shift {
        type State = DVector<f64>;

        fn free_dim(&self) -> usize {
            self.dim
        }
        fn cost(&self, x: &DVector<f64>) -> f64 {
            match self.finite_only_at {
                Some(at) if x[0] != at => f64::NAN,
                _ => (x[0] - self.target).powi(2),
            }
        }
        fn linearize(&self, x: &DVector<f64>) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
            let r = faer::Mat::from_fn(1, 1, |_, _| x[0] - self.target);
            let slope = match self.differentiable_only_at {
                Some(at) if x[0] != at => f64::NAN,
                _ => self.slope,
            };
            let j = SparseColMat::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, slope)])
                .expect("a 1×1 Jacobian");
            (r, j)
        }
        fn retract(&self, x: &DVector<f64>, dx: &DVector<f64>) -> DVector<f64> {
            x + dx
        }
        fn norm(&self, x: &DVector<f64>) -> f64 {
            x.norm()
        }
    }

    /// Options with every threshold off (`0`) unless set.
    fn options(max_iters: usize) -> BackendSolveOptions {
        BackendSolveOptions {
            max_iters,
            min_abs_decrease: Some(0.0),
            min_rel_decrease: Some(0.0),
            min_error: Some(0.0),
            ..BackendSolveOptions::default()
        }
    }

    fn stop(engine: &Shift, x0: f64, opts: &BackendSolveOptions) -> (SolveTermination, usize) {
        let solution = levenberg_marquardt(engine, DVector::from_element(1, x0), opts)
            .unwrap_or_else(|_| panic!("the solve fails"));
        (solution.termination, solution.num_iters)
    }

    #[test]
    fn each_threshold_names_itself() {
        let engine = Shift::to(1.0);
        // The first step lands within ~1e-4 of the target: cost ~1e-8.
        let min_error = BackendSolveOptions {
            min_error: Some(1e-6),
            ..options(10)
        };
        assert_eq!(
            stop(&engine, 0.0, &min_error),
            (SolveTermination::MinError, 1)
        );
        let min_abs = BackendSolveOptions {
            min_abs_decrease: Some(f64::INFINITY),
            ..options(10)
        };
        assert_eq!(
            stop(&engine, 0.0, &min_abs),
            (SolveTermination::MinAbsDecrease, 1)
        );
        // A decrease is at most the whole cost: below twice it always.
        let min_rel = BackendSolveOptions {
            min_rel_decrease: Some(2.0),
            ..options(10)
        };
        assert_eq!(
            stop(&engine, 0.0, &min_rel),
            (SolveTermination::MinRelDecrease, 1)
        );
        // The error rule is checked first when several are met.
        let all = BackendSolveOptions {
            min_error: Some(1e-6),
            min_abs_decrease: Some(f64::INFINITY),
            min_rel_decrease: Some(2.0),
            ..options(10)
        };
        assert_eq!(stop(&engine, 0.0, &all), (SolveTermination::MinError, 1));
    }

    #[test]
    fn a_zero_step_is_a_relative_step_stop() {
        // At the minimum the Gauss-Newton step is exactly zero.
        assert_eq!(
            stop(&Shift::to(1.0), 1.0, &options(10)),
            (SolveTermination::RelativeStep, 1)
        );
    }

    #[test]
    fn the_budget_runs_out_without_a_rule() {
        let engine = Shift::to(1.0);
        assert_eq!(
            stop(&engine, 0.0, &options(1)),
            (SolveTermination::MaxIters, 1)
        );
        assert_eq!(
            stop(&engine, 0.0, &options(0)),
            (SolveTermination::MaxIters, 0)
        );
    }

    #[test]
    fn a_rule_met_on_the_last_iteration_is_convergence() {
        let engine = Shift::to(1.0);
        let opts = BackendSolveOptions {
            min_error: Some(1e-6),
            ..options(1)
        };
        assert_eq!(stop(&engine, 0.0, &opts), (SolveTermination::MinError, 1));
    }

    #[test]
    fn no_finite_cost_off_the_start_is_no_accepted_step() {
        let engine = Shift {
            finite_only_at: Some(0.0),
            ..Shift::to(1.0)
        };
        assert_eq!(
            stop(&engine, 0.0, &options(10)),
            (SolveTermination::NoAcceptedStep, 1)
        );
    }

    #[test]
    fn nothing_free_is_all_parameters_fixed() {
        let engine = Shift {
            dim: 0,
            ..Shift::to(1.0)
        };
        assert_eq!(
            stop(&engine, 0.0, &options(10)),
            (SolveTermination::AllParametersFixed, 0)
        );
    }

    #[test]
    fn only_the_budget_is_not_convergence() {
        use SolveTermination::*;
        for t in [
            RelativeStep,
            MinError,
            MinAbsDecrease,
            MinRelDecrease,
            NoAcceptedStep,
            AllParametersFixed,
        ] {
            assert!(t.is_converged(), "{t:?}");
        }
        assert!(!MaxIters.is_converged());
    }

    #[test]
    fn a_linearization_that_is_not_finite_is_an_error() {
        for slope in [f64::NAN, f64::INFINITY] {
            let engine = Shift {
                slope,
                ..Shift::to(1.0)
            };
            let failure = levenberg_marquardt(
                &engine,
                DVector::from_element(1, 0.0),
                &BackendSolveOptions::default(),
            )
            .err();
            assert_eq!(
                failure,
                Some(LmFailure::NonFiniteLinearization { iteration: 1 }),
                "slope {slope}"
            );
        }
    }

    #[test]
    fn a_linearization_that_turns_non_finite_after_a_step_is_an_error() {
        // The first step is accepted; the second linearization is at a point
        // without a derivative.
        let engine = Shift {
            differentiable_only_at: Some(0.0),
            ..Shift::to(1.0)
        };
        let failure =
            levenberg_marquardt(&engine, DVector::from_element(1, 0.0), &options(10)).err();
        assert_eq!(
            failure,
            Some(LmFailure::NonFiniteLinearization { iteration: 2 })
        );
    }

    #[test]
    fn sharp_turn_after_many_good_steps_does_not_underflow_damping() {
        // Take ~65 good damped-free steps along a quadratic curve (driving
        // damping down to `LM_MIN_DAMPING`), then overshoot the finite-domain
        // wall so retries must restore damping from `LM_MIN_DAMPING` without
        // stalling at `0.0`.
        struct GentleThenTurn {
            threshold: f64,
        }
        impl LinearizationEngine for GentleThenTurn {
            type State = DVector<f64>;
            fn free_dim(&self) -> usize {
                1
            }
            fn cost(&self, x: &DVector<f64>) -> f64 {
                if x[0] > self.threshold {
                    f64::NAN
                } else {
                    (x[0] - (self.threshold - 1e-4)).powi(2)
                }
            }
            fn linearize(&self, x: &DVector<f64>) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
                let target = self.threshold - 1e-4;
                // First take ~65 small under-steps with unit scaling near J=1
                // and steep Jacobian 2.5, then switch to 0.5 near threshold so
                // the undamped Gauss-Newton step overshoots into the NaN wall.
                let slope = if x[0] < target - 0.01 { 2.5 } else { 0.5 };
                let r = faer::Mat::from_fn(1, 1, |_, _| x[0] - target);
                let j = SparseColMat::try_new_from_triplets(1, 1, &[Triplet::new(0, 0, slope)])
                    .unwrap();
                (r, j)
            }
            fn retract(&self, x: &DVector<f64>, dx: &DVector<f64>) -> DVector<f64> {
                x + dx
            }
            fn norm(&self, x: &DVector<f64>) -> f64 {
                x.norm()
            }
        }
        let mut opts = options(150);
        opts.min_abs_decrease = Some(0.0);
        opts.min_rel_decrease = Some(0.0);
        opts.min_error = Some(1e-18);
        let sol = levenberg_marquardt(
            &GentleThenTurn { threshold: 1.0 },
            DVector::from_element(1, 0.0),
            &opts,
        )
        .unwrap();
        assert_ne!(sol.termination, SolveTermination::NoAcceptedStep);
        assert!(sol.cost < 1e-16, "cost={}", sol.cost);
    }

    #[test]
    fn rejected_tiny_damped_step_does_not_trigger_relative_step() {
        // When every step from x0 = 1.0 lands in a NaN region, increasing
        // damping eventually makes ‖dx‖ ≤ tol; LM must report NoAcceptedStep
        // (not RelativeStep from a rejected tiny damped step).
        let engine = Shift {
            finite_only_at: Some(1.0),
            ..Shift::to(0.0)
        };
        assert_eq!(
            stop(&engine, 1.0, &options(10)),
            (SolveTermination::NoAcceptedStep, 1)
        );
    }
}
