//! The Levenberg–Marquardt loop shared by every backend.
//!
//! A backend supplies a [`LinearizationEngine`]: the robust objective, the
//! loss-corrected residual and Jacobian, and the retraction onto its
//! parameter manifolds. The loop itself — Jacobi column scaling, a clamped
//! Marquardt diagonal, the Ceres gain ratio, damping retries and the stopping
//! rules — is written once, so backends differ only in how they linearize.

use crate::backend::{BackendSolveOptions, LinearSolverKind};
use faer::sparse::{SparseColMat, Triplet};
use faer_ext::IntoNalgebra;
use nalgebra::DVector;
use std::ops::Mul;
use tiny_solver::linear::sparse::SparseLinearSolver;
use tiny_solver::linear::{SparseCholeskySolver, SparseQRSolver};

const LM_MIN_DIAGONAL: f64 = 1e-6;
const LM_MAX_DIAGONAL: f64 = 1e32;
const LM_INITIAL_TRUST_REGION_RADIUS: f64 = 1e4;
const LM_MAX_STEP_ATTEMPTS: usize = 32;
/// Stop once a step is this small relative to the parameters
/// (`‖dx‖ ≤ ε(‖x‖ + ε)`, Ceres' `parameter_tolerance`): it can no longer
/// change them, accepted or not.
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

    /// Number of Jacobian columns (the free tangent dimension).
    fn dim(&self) -> usize;

    /// The robust objective `Σᵢ ρᵢ(‖rᵢ‖²)` at `x`.
    fn cost(&self, x: &Self::State) -> f64;

    /// Loss-corrected residual (one column) and Jacobian at `x`.
    fn linearize(&self, x: &Self::State) -> (faer::Mat<f64>, SparseColMat<usize, f64>);

    /// `x ⊕ dx`: the step applied through each block's retraction, with
    /// parameter bounds enforced.
    fn retract(&self, x: &Self::State, dx: &DVector<f64>) -> Self::State;

    /// `‖x‖` over all parameters, summed in a fixed order.
    fn norm(&self, x: &Self::State) -> f64;
}

/// Outcome of a solve.
pub(crate) struct LmSolution<S> {
    /// Parameters at the last accepted step.
    pub state: S,
    /// The robust objective `Σᵢ ρᵢ(‖rᵢ‖²)` at `state`.
    pub cost: f64,
    /// Outer iterations executed.
    pub num_iters: usize,
}

/// Minimize `engine.cost` from `x0`.
///
/// Returns `None` when there is something to optimize but the initial cost is
/// not finite.
pub(crate) fn levenberg_marquardt<E: LinearizationEngine>(
    engine: &E,
    x0: E::State,
    opts: &BackendSolveOptions,
) -> Option<LmSolution<E::State>> {
    let max_iters = opts.max_iters;
    let verbosity = opts.verbosity;
    let min_abs_decrease = opts.min_abs_decrease.unwrap_or(DEFAULT_MIN_ABS_DECREASE);
    let min_rel_decrease = opts.min_rel_decrease.unwrap_or(DEFAULT_MIN_REL_DECREASE);
    let min_error = opts.min_error.unwrap_or(DEFAULT_MIN_ERROR);

    let mut x = x0;
    let mut current_cost = engine.cost(&x);
    let dim = engine.dim();
    if dim == 0 {
        return Some(LmSolution {
            state: x,
            cost: current_cost,
            num_iters: 0,
        });
    }
    if !current_cost.is_finite() {
        return None;
    }

    let mut linear_solver = make_linear_solver(
        opts.linear_solver
            .unwrap_or(LinearSolverKind::SparseCholesky),
    );
    let mut jacobi_scaling_diagonal = None;
    let mut damping = 1.0 / LM_INITIAL_TRUST_REGION_RADIUS;

    let mut num_iters = 0usize;
    for outer_iter in 0..max_iters {
        num_iters = outer_iter + 1;
        let last_cost = current_cost;
        let (residuals, mut jac) = engine.linearize(&x);

        if jacobi_scaling_diagonal.is_none() {
            jacobi_scaling_diagonal = Some(build_jacobi_scaling(&jac));
        }
        let scaling = jacobi_scaling_diagonal
            .as_ref()
            .expect("scaling initialized");
        jac = jac * scaling;

        let jtj = jac
            .as_ref()
            .transpose()
            .to_col_major()
            .unwrap()
            .mul(jac.as_ref());
        let jtr = jac.as_ref().transpose().mul(-&residuals);

        let x_norm = engine.norm(&x);
        let mut accepted = false;
        let mut step_negligible = false;
        for step_attempt in 0..LM_MAX_STEP_ATTEMPTS {
            let mut jtj_regularized = jtj.clone();
            for i in 0..dim {
                let diag = jtj[(i, i)].clamp(LM_MIN_DIAGONAL, LM_MAX_DIAGONAL);
                jtj_regularized[(i, i)] += damping * diag;
            }

            let Some(lm_step) = linear_solver.solve_jtj(&jtr, &jtj_regularized) else {
                damping *= 2.0;
                continue;
            };
            let dx = scaling * &lm_step;
            let dx_na = dx.as_ref().into_nalgebra().column(0).clone_owned();
            if !dx_na.iter().all(|v| v.is_finite()) {
                damping *= 2.0;
                continue;
            }
            if dx_na.norm() <= LM_RELATIVE_STEP_TOLERANCE * (x_norm + LM_RELATIVE_STEP_TOLERANCE) {
                step_negligible = true;
                break;
            }

            let x_new = engine.retract(&x, &dx_na);
            let new_cost = engine.cost(&x_new);
            let actual_cost_change = current_cost - new_cost;
            let linear_cost_change: faer::Mat<f64> =
                lm_step.transpose().mul(2.0 * &jtr - &jtj * &lm_step);
            let predicted_cost_change = linear_cost_change[(0, 0)];
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

            damping *= 2.0;
        }

        if step_negligible {
            if verbosity > 0 {
                println!("lm stopped: relative step below tolerance");
            }
            break;
        }
        if !accepted {
            if verbosity > 0 {
                println!(
                    "lm stopped: no accepted step after {LM_MAX_STEP_ATTEMPTS} damping retries"
                );
            }
            break;
        }

        if current_cost < min_error {
            break;
        }
        let abs_decrease = (last_cost - current_cost).abs();
        if abs_decrease < min_abs_decrease {
            break;
        }
        if last_cost > 0.0 && abs_decrease / last_cost < min_rel_decrease {
            break;
        }
    }

    Some(LmSolution {
        state: x,
        cost: current_cost,
        num_iters,
    })
}

fn make_linear_solver(kind: LinearSolverKind) -> Box<dyn SparseLinearSolver> {
    match kind {
        LinearSolverKind::SparseCholesky => Box::new(SparseCholeskySolver::new()),
        LinearSolverKind::SparseQR => Box::new(SparseQRSolver::new()),
    }
}

/// Column scaling `1 / (1 + ‖J_c‖)`, fixed from the first Jacobian.
fn build_jacobi_scaling(jac: &SparseColMat<usize, f64>) -> SparseColMat<usize, f64> {
    let cols = jac.shape().1;
    let jacobi_scaling_vec: Vec<Triplet<usize, usize, f64>> = (0..cols)
        .map(|c| {
            let v = jac.val_of_col(c).iter().map(|&i| i * i).sum::<f64>().sqrt();
            Triplet::new(c, c, 1.0 / (1.0 + v))
        })
        .collect();

    SparseColMat::<usize, f64>::try_new_from_triplets(cols, cols, &jacobi_scaling_vec).unwrap()
}
