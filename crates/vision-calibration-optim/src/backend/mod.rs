//! Backend adapters that compile the IR into solver-specific problems.
//!
//! Backends are responsible for translating the IR into solver-native graphs,
//! applying manifolds and constraints, and returning a solved parameter map.

pub(crate) mod covariance;
#[cfg(test)]
mod covariance_tests;
mod factrs_backend;
mod kernels;
mod lm;
mod normal_equations;
#[cfg(test)]
mod parity_tests;
mod s2;
mod tiny_solver_backend;
mod tiny_solver_manifolds;

use crate::Error;
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use vision_calibration_core::RunControl;

use crate::ir::ProblemIR;

use factrs_backend::FactrsBackend;
use tiny_solver_backend::TinySolverBackend;

/// Which engine linearizes the problem.
///
/// Both run the same Levenberg–Marquardt loop over the same residual
/// kernels and reach the same minimizer; they differ in how they build the
/// Jacobian and how they fold in a robust loss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SolverBackend {
    /// `tiny-solver`: forward-mode dual numbers on the stack, each residual
    /// block differentiated in passes of four directions; a robust loss
    /// enters through the Triggs correction (second-order exact).
    #[default]
    TinySolver,
    /// `factrs`: static-size forward-mode dual numbers; a robust loss enters
    /// through iterative reweighting (each block scaled by `√ρ′`).
    Factrs,
}

/// Backend-agnostic solver options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSolveOptions {
    /// Linearization engine.
    #[serde(default)]
    pub backend: SolverBackend,
    /// Maximum number of iterations for the optimizer.
    pub max_iters: usize,
    /// Verbosity level (backend-specific).
    pub verbosity: usize,
    /// Absolute error decrease threshold for early termination.
    pub min_abs_decrease: Option<f64>,
    /// Relative error decrease threshold for early termination.
    pub min_rel_decrease: Option<f64>,
    /// Error threshold for early termination.
    pub min_error: Option<f64>,
    /// Progress reports and cancellation: polled before every iteration,
    /// told each finished one. Not configuration, so never serialized.
    #[serde(skip)]
    pub control: RunControl,
}

impl Default for BackendSolveOptions {
    fn default() -> Self {
        Self {
            backend: SolverBackend::default(),
            max_iters: 100,
            verbosity: 0,
            min_abs_decrease: Some(1e-5),
            min_rel_decrease: Some(1e-5),
            min_error: Some(1e-10),
            control: RunControl::default(),
        }
    }
}

/// Why a Levenberg–Marquardt solve stopped.
///
/// Every reason but [`MaxIters`](Self::MaxIters) is a stopping rule
/// ([`is_converged`](Self::is_converged)): the loop could not improve the
/// solution further, and only then is its covariance computed. A cancelled
/// solve, or one whose cost is not finite, returns an error instead of a
/// solution. The cost the thresholds are compared with is the robust
/// objective `Σᵢ ρᵢ(‖rᵢ‖²)`, twice the report's
/// [`final_cost`](SolveReport::final_cost).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SolveTermination {
    /// The step became negligible next to the parameters,
    /// `‖dx‖ ≤ 10⁻⁸ (‖x‖ + 10⁻⁸)`: it can no longer change them.
    RelativeStep,
    /// The cost fell below its floor, `min_error` (10⁻¹⁰).
    MinError,
    /// An accepted step lowered the cost by less than `min_abs_decrease`
    /// (10⁻⁵).
    MinAbsDecrease,
    /// An accepted step lowered the cost by less than `min_rel_decrease`
    /// (10⁻⁵) of the cost before it.
    MinRelDecrease,
    /// No step lowered the cost, with the damping doubled on each of 32
    /// tries: a minimum to working precision, or a parameter held on its
    /// bound (the covariance then names it).
    NoAcceptedStep,
    /// The iteration budget, `max_iters`, ran out before a stopping rule
    /// was met; the solution has no covariance.
    MaxIters,
    /// Every parameter is fixed: no iteration ran.
    AllParametersFixed,
}

impl SolveTermination {
    /// Whether a stopping rule ended the solve, rather than its iteration
    /// budget: every reason but [`MaxIters`](Self::MaxIters). The
    /// covariance of a solution is computed only then.
    pub fn is_converged(self) -> bool {
        !matches!(self, Self::MaxIters)
    }
}

/// Summary of backend solve outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[non_exhaustive]
pub struct SolveReport {
    /// The objective at the solution: `½ Σᵢ ρᵢ(‖rᵢ‖²)` over the residual
    /// blocks, where `ρᵢ` is the block's robust loss (`ρ(s) = s` without
    /// one) — so `½‖r‖²` for a plain least-squares problem.
    pub final_cost: f64,
    /// Number of outer solver iterations executed.
    #[serde(default)]
    pub num_iters: usize,
    /// Why the solve stopped. `None` when that is unknown: a report written
    /// before the reason was recorded, or one that no solve produced (a
    /// result taken from a scored initial guess).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub termination: Option<SolveTermination>,
}
impl SolveReport {
    /// A [`SolveReport`] with the given fields and no
    /// [`termination`](Self::termination).
    pub fn new(final_cost: f64, num_iters: usize) -> Self {
        Self {
            final_cost,
            num_iters,
            termination: None,
        }
    }
}

/// Solver output from a backend.
///
/// The `params` map uses the IR parameter block names.
#[derive(Debug, Clone)]
pub(crate) struct BackendSolution {
    /// Optimized parameter vectors keyed by block name.
    pub params: HashMap<String, DVector<f64>>,
    /// Final robustified cost if supported by the backend.
    pub solve_report: SolveReport,
    /// Noise level of each residual family at the solution, whether or
    /// not the covariance is available.
    pub noise: Vec<covariance::FamilyNoise>,
    /// Covariance of the free parameters at the solution.
    pub uncertainty: covariance::ParamUncertainty,
}

/// Backend interface implemented by solver adapters.
pub trait OptimBackend {
    /// Solve a compiled IR with the provided initial parameters.
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution, Error>;
}

/// A backend's [`lm::LinearizationEngine`], with what the shared solve
/// ([`solve_on`]) needs beyond the LM loop.
trait SolveEngine: lm::LinearizationEngine {
    /// The backend's name, for the error message.
    const NAME: &'static str;

    /// The IR parameter blocks at `x`.
    fn read_back(&self, ir: &ProblemIR, x: &Self::State) -> HashMap<String, DVector<f64>>;

    /// The engine's own linearization at `x`, with each IR block's columns,
    /// for an engine whose problem has no robust loss; `None` when every
    /// coordinate is fixed.
    fn raw_at(&self, ir: &ProblemIR, x: &Self::State) -> Option<covariance::RawLinearization>;

    /// The loss-free linearization of `ir` at `params`, from a fresh compile
    /// on this backend; `None` when every coordinate is fixed.
    fn raw_linearization(
        ir: &ProblemIR,
        params: &HashMap<String, DVector<f64>>,
    ) -> Result<Option<covariance::RawLinearization>, Error>;
}

/// How much post-solve linearization/covariance work `solve_on` performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CovarianceMode {
    None,
    FirstOrder,
    Full,
}

/// Solve `ir` on `engine` from `x0` with the shared LM, then read the
/// solution back and compute its noise/covariance according to `cov_mode`.
fn solve_on<E: SolveEngine>(
    ir: &ProblemIR,
    engine: &E,
    x0: E::State,
    opts: &BackendSolveOptions,
    cov_mode: CovarianceMode,
) -> Result<BackendSolution, Error> {
    let lm::LmSolution {
        state,
        cost,
        num_iters,
        termination,
    } = lm::levenberg_marquardt(engine, x0, opts).map_err(|failure| match failure {
        lm::LmFailure::NonFiniteCost => Error::numerical(format!(
            "{}: the cost at the initial parameters is not finite",
            E::NAME
        )),
        lm::LmFailure::NonFiniteLinearization { iteration } => Error::numerical(format!(
            "{}: a residual or its Jacobian is not finite at the parameters of iteration \
             {iteration}; a residual is not differentiable there",
            E::NAME
        )),
        lm::LmFailure::Failed => Error::numerical(format!(
            "{}: the normal equations cannot be allocated",
            E::NAME
        )),
        lm::LmFailure::Cancelled => Error::Cancelled,
    })?;

    let params = engine.read_back(ir, &state);
    if cov_mode == CovarianceMode::None {
        return Ok(BackendSolution {
            noise: Vec::new(),
            uncertainty: covariance::ParamUncertainty::Unavailable(
                covariance::UnavailableReason::NotComputed,
            ),
            params,
            solve_report: SolveReport {
                final_cost: 0.5 * cost,
                num_iters,
                termination: Some(termination),
            },
        });
    }

    // Without a robust loss the solve's own linearization is the raw one.
    let raw = if ir.has_robust_loss() {
        E::raw_linearization(ir, &params)?
    } else {
        engine.raw_at(ir, &state)
    };
    if let Some(raw) = &raw
        && !raw
            .residual
            .col(0)
            .iter()
            .chain(raw.jacobian.val())
            .all(|v| v.is_finite())
    {
        return Err(Error::numerical(format!(
            "{}: a residual or its Jacobian is not finite at the solution; a residual is \
             not differentiable there",
            E::NAME
        )));
    }
    let second_order = cov_mode == CovarianceMode::Full;
    Ok(BackendSolution {
        noise: covariance::noise(ir, raw.as_ref()),
        uncertainty: covariance::estimate(
            ir,
            &params,
            raw,
            termination.is_converged(),
            second_order,
        ),
        params,
        solve_report: SolveReport {
            final_cost: 0.5 * cost,
            num_iters,
            termination: Some(termination),
        },
    })
}

/// The loss-free residual and Jacobian of `ir` at `params`, with each IR
/// block's columns; `None` when every coordinate is fixed. The tiny-solver
/// engine linearizes every factor kind, whichever backend solved.
pub(crate) fn raw_linearization(
    ir: &ProblemIR,
    params: &HashMap<String, DVector<f64>>,
) -> Result<Option<covariance::RawLinearization>, Error> {
    TinySolverBackend.raw_linearization(ir, params)
}

/// The loss-free residuals of `ir` at `params` moved by tangent steps. The
/// tiny-solver factors evaluate every factor kind, whichever backend solved;
/// a step is given per [`covariance::Coordinate`], whose meaning both
/// backends share.
pub(crate) fn residual_probe<'ir>(
    ir: &'ir ProblemIR,
    params: &HashMap<String, DVector<f64>>,
) -> Result<tiny_solver_backend::ResidualProbe<'ir>, Error> {
    tiny_solver_backend::ResidualProbe::new(ir, params)
}

/// [`solve`] without the covariance's second-order prediction, for a solve
/// whose covariance only steers the next one; [`add_second_order`] adds it
/// to the one kept.
pub(crate) fn solve_first_order(
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Result<BackendSolution, Error> {
    match opts.backend {
        SolverBackend::TinySolver => {
            TinySolverBackend.solve_with(ir, initial, opts, CovarianceMode::FirstOrder)
        }
        SolverBackend::Factrs => {
            FactrsBackend.solve_with(ir, initial, opts, CovarianceMode::FirstOrder)
        }
    }
}

/// [`solve`] skipping linearization and covariance completely, for intermediate
/// warm-start stages whose uncertainty is discarded.
pub(crate) fn solve_without_covariance(
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Result<BackendSolution, Error> {
    match opts.backend {
        SolverBackend::TinySolver => {
            TinySolverBackend.solve_with(ir, initial, opts, CovarianceMode::None)
        }
        SolverBackend::Factrs => FactrsBackend.solve_with(ir, initial, opts, CovarianceMode::None),
    }
}

/// Adds the second-order prediction to a [`solve_first_order`] solution of
/// `ir`, from a fresh loss-free linearization at its parameters; the
/// covariance itself is kept as solved.
pub(crate) fn add_second_order(
    ir: &ProblemIR,
    solution: &mut BackendSolution,
) -> Result<(), Error> {
    let covariance::ParamUncertainty::Available(estimate) = &mut solution.uncertainty else {
        return Ok(());
    };
    let raw = raw_linearization(ir, &solution.params)?;
    if let covariance::ParamUncertainty::Available(full) =
        covariance::estimate(ir, &solution.params, raw, true, true)
    {
        estimate.second_order = full.second_order;
    }
    Ok(())
}

/// Solve `ir` from `initial` with the backend `opts.backend` selects.
///
/// This is the backend-agnostic entry point every problem uses.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for an IR the backend cannot compile and
/// [`Error::Numerical`] if the solve fails.
pub(crate) fn solve(
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Result<BackendSolution, Error> {
    match opts.backend {
        SolverBackend::TinySolver => TinySolverBackend.solve(ir, initial, opts),
        SolverBackend::Factrs => FactrsBackend.solve(ir, initial, opts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_json_round_trips() {
        let opts = BackendSolveOptions {
            backend: SolverBackend::Factrs,
            max_iters: 7,
            ..BackendSolveOptions::default()
        };
        let json = serde_json::to_string(&opts).unwrap();
        let back: BackendSolveOptions = serde_json::from_str(&json).unwrap();
        assert_eq!(back.backend, SolverBackend::Factrs);
        assert_eq!(back.max_iters, 7);
        assert_eq!(back.min_rel_decrease, opts.min_rel_decrease);
    }

    #[test]
    fn options_json_with_a_linear_solver_still_loads() {
        let json = r#"{"max_iters": 5, "verbosity": 0, "linear_solver": "SparseQR",
            "min_abs_decrease": null, "min_rel_decrease": null, "min_error": null}"#;
        let opts: BackendSolveOptions = serde_json::from_str(json).unwrap();
        assert_eq!(opts.max_iters, 5);
        assert_eq!(opts.backend, SolverBackend::TinySolver);
    }
}
