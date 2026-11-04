//! Backend adapters that compile the IR into solver-specific problems.
//!
//! Backends are responsible for translating the IR into solver-native graphs,
//! applying manifolds and constraints, and returning a solved parameter map.

pub(crate) mod covariance;
#[cfg(test)]
mod covariance_tests;
mod factrs_backend;
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
pub enum SolverBackend {
    /// `tiny-solver`: dynamic-size forward-mode dual numbers; a robust loss
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

/// Summary of backend solve outcome.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
pub struct SolveReport {
    /// The objective at the solution: `½ Σᵢ ρᵢ(‖rᵢ‖²)` over the residual
    /// blocks, where `ρᵢ` is the block's robust loss (`ρ(s) = s` without
    /// one) — so `½‖r‖²` for a plain least-squares problem.
    pub final_cost: f64,
    /// Number of outer solver iterations executed.
    #[serde(default)]
    pub num_iters: usize,
}

/// Solver output from a backend.
///
/// The `params` map uses the IR parameter block names.
#[derive(Debug, Clone)]
pub struct BackendSolution {
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
        SolverBackend::TinySolver => TinySolverBackend.solve_with(ir, initial, opts, false),
        SolverBackend::Factrs => FactrsBackend.solve_with(ir, initial, opts, false),
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
