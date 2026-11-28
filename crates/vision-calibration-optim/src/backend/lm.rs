//! The Levenberg–Marquardt loop shared by every backend.
//!
//! A backend supplies a [`LinearizationEngine`]: the robust objective, the
//! loss-corrected residual and Jacobian, and the retraction onto its
//! parameter manifolds. The loop itself — Jacobi column scaling, a clamped
//! Marquardt diagonal, the Ceres gain ratio, damping retries and the stopping
//! rules — is written once, so backends differ only in how they linearize.
//!
//! # When the loop stops
//!
//! A solve's result is reported with a covariance that assumes it is the
//! minimum, so the loop stops on how far it still is from the minimum, in
//! the units of that covariance, not on how much its last step gained. At
//! `x`, with gradient `g` and Gauss–Newton matrix `H` of the robust
//! objective `Σρ`, the undamped step to the minimum of the local model
//! lowers the model by `gᵀH⁻¹g`. With the residual variance
//! `σ̂² = Σρ / (m − n)` the solution's covariance is about `σ̂² H⁻¹`, so
//!
//! ```text
//! d² = gᵀH⁻¹g / σ̂²
//! ```
//!
//! is the squared Mahalanobis length of that step: the distance from the
//! minimum in standard deviations, which bounds every parameter's distance
//! in units of its own σ. The loop solves for the undamped step on the first
//! iteration and whenever the last accepted step moved the solution by at
//! most one σ (lowered `Σρ` by at most `σ̂²`); when `d ≤ sigma_tolerance` it
//! takes that step if it lowers the cost, and stops
//! ([`SolveTermination::SigmaStep`]).
//!
//! The test measures the undamped step, so damping that still shortens the
//! loop's own steps (it starts at `10⁻⁴` and falls threefold per good step)
//! does not hide a weak direction. Under a robust loss the backends' `H` is
//! iterative reweighting where `ρ″ < 0` (Ceres' clamped correction): larger
//! than the objective's curvature, so the step goes only part of the way
//! and `d` understates the distance. Along the step the gain ratio
//! `ρ = actual / predicted decrease` is `2 − a/b` for an objective of
//! curvature `a` and a model of curvature `b`, so the step leaves
//! `|ρ − 1| · d / √(2 − ρ)` of the distance; the loop stops only when that
//! is within the tolerance too, and otherwise iterates on.
//!
//! A solve can also make no headway: an accepted step that moves the
//! solution by less than a tenth of the tolerance (lowers `Σρ` by at most
//! `(0.001)² σ̂²`) while `d` is still larger ends it
//! ([`SolveTermination::StalledStep`]), not converged. Such a step says
//! nothing of the distance left: a long curved valley of a weakly
//! determined parameter (where the local model keeps placing the minimum
//! out of reach) stalls so, and so does a step the damping held short far
//! from the minimum, so a stall has no covariance; the solve's last iterate
//! still yields one as a diagnostic.
//!
//! A coordinate on a bound that the step would push through is held while
//! the step is solved: the minimum measured is the one the bounds allow.
//! With a coordinate held, the step is taken whenever it lowers the cost,
//! near or not: the loop's own steps push such a coordinate into its bound
//! and the clamp leaves the rest to crawl.

use crate::backend::normal_equations::{NormalEquations, jacobi_scaling, unscale};
use crate::backend::{BackendSolveOptions, SolveTermination};
use faer::sparse::SparseColMat;
use nalgebra::DVector;
use vision_calibration_core::{Progress, RunControl};

const LM_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;
const LM_MIN_DAMPING: f64 = 1e-12;
const LM_MAX_STEP_ATTEMPTS: usize = 32;
/// Stop once an accepted step is this small relative to the parameters
/// (`‖dx‖ ≤ ε(‖x‖ + ε)`, Ceres' `parameter_tolerance`, `‖x‖` over the free
/// blocks, [`LinearizationEngine::norm`]): it can no longer move them
/// appreciably.
const LM_RELATIVE_STEP_TOLERANCE: f64 = 1e-8;

/// The default of [`BackendSolveOptions::sigma_tolerance`]: a solve stops
/// within a hundredth of a standard deviation of its minimum.
pub(crate) const DEFAULT_SIGMA_TOLERANCE: f64 = 0.01;
/// A step that moves the solution by less than this fraction of
/// `sigma_tolerance` (in σ) while the step to the minimum is longer is a
/// stall ([`SolveTermination::StalledStep`], not convergence): progress too
/// slow to reach the minimum within any budget, along a long curved valley
/// or under heavy damping.
const STALL_FRACTION: f64 = 0.1;
/// The damping of the step [`SolveTermination::SigmaStep`] measures: it
/// leaves undamped every direction the covariance resolves (whose rank
/// tolerance is `1e-12` of the unit-diagonal matrix), and keeps a direction
/// without information factorable.
const SIGMA_STEP_DAMPING: f64 = LM_MIN_DAMPING;

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

    /// The free coordinates of `x` that sit on a bound, by Jacobian column,
    /// each with `true` for its upper bound; none without bounds.
    ///
    /// The distance to the minimum is measured with the coordinates a step
    /// would push through their bound held: the minimum is the one the
    /// bounds allow.
    fn on_bounds(&self, _x: &Self::State) -> Vec<(usize, bool)> {
        Vec::new()
    }
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
    // A threshold left as `None` is off.
    let min_abs_decrease = opts.min_abs_decrease.unwrap_or(0.0);
    let min_rel_decrease = opts.min_rel_decrease.unwrap_or(0.0);
    let min_error = opts.min_error.unwrap_or(0.0);
    let sigma_tolerance = opts.sigma_tolerance;

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
    // How much the last accepted step lowered the cost; `None` before the
    // first.
    let mut last_decrease: Option<f64> = None;
    for outer_iter in 0..max_iters {
        if control.is_cancelled() {
            return Err(LmFailure::Cancelled);
        }
        num_iters = outer_iter + 1;
        let last_cost = current_cost;
        // Whether this iteration measured the distance to the minimum and
        // found it beyond the tolerance: a stall is judged only then.
        let mut measured_far = false;
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

        // The distance to the minimum, measured on the first iteration and
        // once the last step moved the solution by at most one σ.
        if let Some(variance) =
            residual_variance(current_cost, residuals.nrows(), engine.free_dim())
            && sigma_tolerance > 0.0
            && last_decrease.is_none_or(|decrease| decrease <= variance)
        {
            let finish = gauss_newton_finish(
                engine,
                &x,
                current_cost,
                normal,
                scaling,
                variance,
                sigma_tolerance,
            );
            match finish {
                Finish::Far => measured_far = true,
                Finish::AtMinimum => {
                    report(control, num_iters, max_iters, current_cost);
                    termination = SolveTermination::SigmaStep;
                    break;
                }
                Finish::Moved { state, cost, stop } => {
                    last_decrease = Some(current_cost - cost);
                    x = state;
                    current_cost = cost;
                    report(control, num_iters, max_iters, current_cost);
                    if stop {
                        termination = SolveTermination::SigmaStep;
                        break;
                    }
                    // The step went part of the way (a robust loss's
                    // reweighted curvature): linearize where it ended.
                    continue;
                }
            }
        }

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
        report(control, num_iters, max_iters, current_cost);

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
        last_decrease = Some(last_cost - current_cost);
        // A step that moved the solution by under a tenth of the tolerance
        // in σ, where the step to the minimum measured at this iteration is
        // longer than it. Unmeasured (the last step moved the solution more
        // than a σ), a small step is left to the next iteration's measure:
        // a step clamped on a bound is one, and the measure holds the bound.
        let stalled = measured_far
            && residual_variance(current_cost, residuals.nrows(), engine.free_dim()).is_some_and(
                |variance| {
                    let stall = STALL_FRACTION * sigma_tolerance;
                    last_cost - current_cost <= stall * stall * variance
                },
            );
        let stop = if current_cost < min_error {
            Some(SolveTermination::MinError)
        } else if stalled {
            Some(SolveTermination::StalledStep)
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

/// Tell `control` an iteration finished at `cost` (the robust objective
/// `Σρ`; reports use the solve report's `½Σρ`).
fn report(control: &RunControl, iteration: usize, max_iters: usize, cost: f64) {
    control.report(Progress::iteration(iteration, max_iters, 0.5 * cost));
}

/// The residual variance `σ̂² = Σρ / (m − n)` of `m` residuals and `n` free
/// parameters; `None` without redundancy or at a zero cost, where no
/// distance can be measured in standard deviations.
fn residual_variance(cost: f64, rows: usize, free: usize) -> Option<f64> {
    let dof = rows.checked_sub(free).filter(|&dof| dof > 0)?;
    let variance = cost / dof as f64;
    (variance > 0.0 && variance.is_finite()).then_some(variance)
}

/// What [`gauss_newton_finish`] found.
enum Finish<S> {
    /// More than the tolerance from the minimum, or not measurable here:
    /// the loop takes its own step.
    Far,
    /// Within the tolerance, and the step to the minimum does not lower the
    /// cost: the solution stays.
    AtMinimum,
    /// The step lowered the cost to `cost` at `state`; `stop` when it was
    /// within the tolerance and what it leaves is within it too.
    Moved { state: S, cost: f64, stop: bool },
}

/// The undamped Gauss–Newton step at `x`, taken if it lowers the cost when
/// it is within `tolerance` standard deviations of the minimum of the local
/// model (the module docs give the measure), or when it holds coordinates on
/// their bounds (a projected step, which the loop's clamped steps are not).
///
/// `normal` holds the normal equations at `x` under the column `scaling`;
/// `variance` is `σ̂²` there.
fn gauss_newton_finish<E: LinearizationEngine>(
    engine: &E,
    x: &E::State,
    cost: f64,
    normal: &mut NormalEquations,
    scaling: &[f64],
    variance: f64,
    tolerance: f64,
) -> Finish<E::State> {
    let Some(mut step) = normal.solve_damped(SIGMA_STEP_DAMPING) else {
        return Finish::Far;
    };
    // A coordinate on its bound that the step would push through is held:
    // the minimum is the one the bounds allow (an active set, refined while
    // holding one frees the step to push another).
    let on_bounds = engine.on_bounds(x);
    let mut holding = false;
    if !on_bounds.is_empty() {
        let mut held = vec![false; scaling.len()];
        for _ in 0..=on_bounds.len() {
            let mut changed = false;
            for &(column, upper) in &on_bounds {
                let outward = if upper {
                    step[(column, 0)] > 0.0
                } else {
                    step[(column, 0)] < 0.0
                };
                if outward && !held[column] {
                    held[column] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            let Some(held_step) = normal.solve_damped_holding(SIGMA_STEP_DAMPING, &held) else {
                return Finish::Far;
            };
            step = held_step;
            holding = true;
        }
    }
    let predicted = normal.model_decrease(&step);
    let near = predicted <= tolerance * tolerance * variance;
    // Far from the minimum the step is taken only with coordinates held: the
    // loop's own steps push them into their bounds, and the clamp leaves the
    // rest to crawl (a projected Gauss-Newton step).
    if !(near || holding && predicted.is_finite()) {
        return Finish::Far;
    }
    if predicted <= 0.0 {
        // No gradient left to follow.
        return Finish::AtMinimum;
    }
    let dx = unscale(&step, scaling);
    if !dx.iter().all(|v| v.is_finite()) {
        return Finish::Far;
    }
    let state = engine.retract(x, &dx);
    let new_cost = engine.cost(&state);
    if new_cost.partial_cmp(&cost) != Some(std::cmp::Ordering::Less) {
        // A step this short that does not lower the cost: the minimum to
        // working precision.
        return if near { Finish::AtMinimum } else { Finish::Far };
    }
    // Along the step the objective has curvature `a`, the model `b`, and
    // the gain ratio is `2 − a/b`: the step covered `a/b` of a distance
    // `d / √(a/b)` and leaves `|1 − a/b|` of it.
    //
    // A gain of 2 or more means `a ≤ 0` along the step: not a quadratic
    // bowl, which at a distance within the tolerance of the minimum only
    // rounding makes (a predicted decrease of a few ulps of the cost). The
    // step was within the tolerance, so it leaves nothing to measure.
    let gain = (cost - new_cost) / predicted;
    let distance = (predicted / variance).sqrt();
    let remaining = if gain < 2.0 {
        (gain - 1.0).abs() * distance / (2.0 - gain).sqrt()
    } else {
        0.0
    };
    Finish::Moved {
        state,
        cost: new_cost,
        stop: near && remaining <= tolerance,
    }
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
    fn only_the_budget_and_a_stall_are_not_convergence() {
        use SolveTermination::*;
        for t in [
            RelativeStep,
            MinError,
            MinAbsDecrease,
            MinRelDecrease,
            NoAcceptedStep,
            AllParametersFixed,
            SigmaStep,
        ] {
            assert!(t.is_converged(), "{t:?}");
        }
        assert!(!MaxIters.is_converged());
        assert!(!StalledStep.is_converged());
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

    /// `rᵢ = (x − tᵢ)/c` with the Jacobian `c`, `Σr²` with `c = 1`: the
    /// mean of `targets`, with the gradient of the objective exact and the
    /// model's curvature `c²` times the objective's, as a robust loss's
    /// reweighting has it where `ρ″ < 0` (`c = 1` is plain least squares).
    struct Mean {
        targets: Vec<f64>,
        curvature: f64,
    }

    impl LinearizationEngine for Mean {
        type State = DVector<f64>;

        fn free_dim(&self) -> usize {
            1
        }
        fn cost(&self, x: &DVector<f64>) -> f64 {
            self.targets.iter().map(|t| (x[0] - t).powi(2)).sum()
        }
        fn linearize(&self, x: &DVector<f64>) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
            let c = self.curvature.sqrt();
            let m = self.targets.len();
            let r = faer::Mat::from_fn(m, 1, |i, _| (x[0] - self.targets[i]) / c);
            let entries: Vec<_> = (0..m).map(|i| Triplet::new(i, 0, c)).collect();
            let j = SparseColMat::try_new_from_triplets(m, 1, &entries).expect("an m×1 Jacobian");
            (r, j)
        }
        fn retract(&self, x: &DVector<f64>, dx: &DVector<f64>) -> DVector<f64> {
            x + dx
        }
        fn norm(&self, x: &DVector<f64>) -> f64 {
            x.norm()
        }
    }

    impl Mean {
        /// 200 targets of spread about 1 around 3.
        fn new(curvature: f64) -> Self {
            let targets = (0..200)
                .map(|i| 3.0 + ((i as f64) * 0.618_033_988_75).fract() * 3.4 - 1.7)
                .collect();
            Self { targets, curvature }
        }
        fn minimum(&self) -> f64 {
            self.targets.iter().sum::<f64>() / self.targets.len() as f64
        }
        /// The distance of `x` from the minimum in standard deviations of
        /// the mean.
        fn sigmas(&self, x: f64) -> f64 {
            let m = self.targets.len() as f64;
            let variance = self.cost(&DVector::from_element(1, self.minimum())) / (m - 1.0);
            (x - self.minimum()).abs() / (variance / m).sqrt()
        }
    }

    /// The default rules only: the σ rule on, the decrease rules off.
    fn defaults(max_iters: usize) -> BackendSolveOptions {
        BackendSolveOptions {
            max_iters,
            ..BackendSolveOptions::default()
        }
    }

    #[test]
    fn a_solve_stops_within_a_hundredth_of_a_sigma() {
        let engine = Mean::new(1.0);
        let solution =
            levenberg_marquardt(&engine, DVector::from_element(1, 40.0), &defaults(100)).unwrap();
        assert_eq!(solution.termination, SolveTermination::SigmaStep);
        let off = engine.sigmas(solution.state[0]);
        assert!(off <= 0.01, "{off} σ off");
    }

    #[test]
    fn a_start_at_the_minimum_stops_on_the_first_iteration() {
        let engine = Mean::new(1.0);
        let start = DVector::from_element(1, engine.minimum());
        let solution = levenberg_marquardt(&engine, start, &defaults(100)).unwrap();
        assert_eq!(
            (solution.termination, solution.num_iters),
            (SolveTermination::SigmaStep, 1)
        );
    }

    #[test]
    fn a_reweighted_curvature_iterates_until_the_remainder_is_small() {
        // The model's curvature five times the objective's: a Gauss–Newton
        // step covers a fifth of the way, and the model sees √(1/5) of the
        // distance. From 0.02σ it sees 0.009σ, inside the tolerance, but the
        // step leaves 0.016σ, which its gain ratio (1.8) tells: the loop
        // goes on. From 0.01σ the step leaves 0.008σ: one iteration.
        for (start_sigmas, one_iteration) in [(0.01, true), (0.02, false)] {
            let engine = Mean::new(5.0);
            let m = engine.targets.len() as f64;
            let variance = engine.cost(&DVector::from_element(1, engine.minimum())) / (m - 1.0);
            let start = engine.minimum() + start_sigmas * (variance / m).sqrt();
            let solution =
                levenberg_marquardt(&engine, DVector::from_element(1, start), &defaults(100))
                    .unwrap();
            assert_eq!(solution.termination, SolveTermination::SigmaStep);
            assert_eq!(
                solution.num_iters == 1,
                one_iteration,
                "from {start_sigmas} σ: {} iterations",
                solution.num_iters
            );
            let off = engine.sigmas(solution.state[0]);
            assert!(off <= 0.01, "from {start_sigmas} σ: {off} σ off");
        }
    }

    #[test]
    fn a_model_that_overstates_the_distance_ends_on_a_stall() {
        // The model's curvature a thousandth of the objective's: it places
        // the minimum √1000 times further off than it is, the damping keeps
        // the steps short, and the loop stalls before the model's distance
        // falls within the tolerance.
        let engine = Mean::new(1e-3);
        let solution =
            levenberg_marquardt(&engine, DVector::from_element(1, 4.0), &defaults(500)).unwrap();
        assert_eq!(solution.termination, SolveTermination::StalledStep);
        assert!(!solution.termination.is_converged());
        let off = engine.sigmas(solution.state[0]);
        assert!(off <= 0.01, "{off} σ off");
    }

    /// Least squares over two nearly collinear columns,
    /// `rᵢ = x₀ uᵢ + x₁ (uᵢ + ε vᵢ) − yᵢ`: a strong direction and a weak one
    /// (eigenvalue about `ε²` of the scaled normal matrix).
    struct Collinear {
        u: Vec<f64>,
        v: Vec<f64>,
        y: Vec<f64>,
        eps: f64,
    }

    impl Collinear {
        fn new(eps: f64) -> Self {
            let m = 200;
            let u: Vec<f64> = (0..m).map(|i| (0.37 * i as f64).cos() + 1.5).collect();
            let v: Vec<f64> = (0..m).map(|i| (0.91 * i as f64).sin()).collect();
            let y = (0..m)
                .map(|i| u[i] + 2.0 * (u[i] + eps * v[i]) + 0.1 * (1.7 * i as f64).sin())
                .collect();
            Self { u, v, y, eps }
        }
        fn columns(&self) -> nalgebra::DMatrix<f64> {
            let m = self.u.len();
            nalgebra::DMatrix::from_fn(m, 2, |i, j| {
                if j == 0 {
                    self.u[i]
                } else {
                    self.u[i] + self.eps * self.v[i]
                }
            })
        }
        /// The minimum, the residual variance there, and the normal matrix.
        fn minimum(&self) -> (DVector<f64>, f64, nalgebra::DMatrix<f64>) {
            let a = self.columns();
            let h = a.transpose() * &a;
            let y = DVector::from_vec(self.y.clone());
            let x = h.clone().cholesky().unwrap().solve(&(a.transpose() * &y));
            let variance = (&a * &x - y).norm_squared() / (self.y.len() - 2) as f64;
            (x, variance, h)
        }
        /// The distance of `x` from the minimum, in standard deviations
        /// (Mahalanobis).
        fn sigmas(&self, x: &DVector<f64>) -> f64 {
            let (xs, variance, h) = self.minimum();
            let e = x - xs;
            ((e.transpose() * h * e)[(0, 0)] / variance).sqrt()
        }
    }

    impl LinearizationEngine for Collinear {
        type State = DVector<f64>;

        fn free_dim(&self) -> usize {
            2
        }
        fn cost(&self, x: &DVector<f64>) -> f64 {
            (self.columns() * x - DVector::from_vec(self.y.clone())).norm_squared()
        }
        fn linearize(&self, x: &DVector<f64>) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
            let a = self.columns();
            let r = &a * x - DVector::from_vec(self.y.clone());
            let m = self.u.len();
            let entries: Vec<_> = (0..m)
                .flat_map(|i| [Triplet::new(i, 0, a[(i, 0)]), Triplet::new(i, 1, a[(i, 1)])])
                .collect();
            let j = SparseColMat::try_new_from_triplets(m, 2, &entries).expect("an m×2 Jacobian");
            (faer::Mat::from_fn(m, 1, |i, _| r[i]), j)
        }
        fn retract(&self, x: &DVector<f64>, dx: &DVector<f64>) -> DVector<f64> {
            x + dx
        }
        fn norm(&self, x: &DVector<f64>) -> f64 {
            x.norm()
        }
    }

    #[test]
    fn a_step_the_damping_held_short_is_not_convergence() {
        // At the strong direction's minimum, a σ off along the weak one
        // (eigenvalue ~1e-10 of the scaled matrix, against the starting
        // damping of 1e-4): the first steps cover a millionth of the way and
        // lower the cost by about 1e-6 σ̂², a stall. A stall reports no
        // convergence; the solve goes on to its minimum or says it did not.
        let engine = Collinear::new(1e-5);
        let (xs, variance, h) = engine.minimum();
        let eig = h.symmetric_eigen();
        let weak = eig.eigenvalues.imin();
        let w = eig.eigenvectors.column(weak).into_owned();
        let start = &xs + w * (variance / eig.eigenvalues[weak]).sqrt();
        assert!((engine.sigmas(&start) - 1.0).abs() < 1e-6);
        let solution = levenberg_marquardt(&engine, start, &defaults(100)).unwrap();
        let off = engine.sigmas(&solution.state);
        assert!(
            !solution.termination.is_converged() || off <= 0.01,
            "{:?} after {} iterations, {off:.3} σ off",
            solution.termination,
            solution.num_iters
        );
    }

    #[test]
    fn a_rounding_gain_at_the_minimum_ends_the_solve() {
        // Started at the minimum: the Gauss-Newton step predicts a decrease
        // of a few ulps, and the cost (offset by a constant anywhere off the
        // exact minimum) falls by far more, a gain ratio far above 2. The
        // solve stops on its first iteration (the remaining-distance estimate
        // once read infinite there and iterated on).
        struct Offset(Mean);
        impl LinearizationEngine for Offset {
            type State = DVector<f64>;
            fn free_dim(&self) -> usize {
                1
            }
            fn cost(&self, x: &DVector<f64>) -> f64 {
                let at = self.0.minimum();
                self.0.cost(x) - if (x[0] - at).abs() < 1e-10 { 1e-9 } else { 0.0 }
            }
            fn linearize(&self, x: &DVector<f64>) -> (faer::Mat<f64>, SparseColMat<usize, f64>) {
                self.0.linearize(x)
            }
            fn retract(&self, x: &DVector<f64>, dx: &DVector<f64>) -> DVector<f64> {
                x + dx
            }
            fn norm(&self, x: &DVector<f64>) -> f64 {
                x.norm()
            }
        }
        let engine = Offset(Mean::new(1.0));
        // A nanometre off: a predicted decrease of ~1e-16, positive.
        let start = DVector::from_element(1, engine.0.minimum() + 1e-9);
        let solution = levenberg_marquardt(&engine, start, &defaults(100)).unwrap();
        assert_eq!(solution.termination, SolveTermination::SigmaStep);
        assert_eq!(solution.num_iters, 1);
    }

    #[test]
    fn without_redundancy_the_sigma_rule_is_off() {
        // One residual, one parameter: no σ to measure in. The other rules
        // end the solve, at the exact minimum.
        let solution = levenberg_marquardt(
            &Shift::to(1.0),
            DVector::from_element(1, 0.0),
            &defaults(100),
        )
        .unwrap();
        assert_ne!(solution.termination, SolveTermination::SigmaStep);
        assert!(solution.termination.is_converged());
        assert!((solution.state[0] - 1.0).abs() < 1e-4);
    }
}
