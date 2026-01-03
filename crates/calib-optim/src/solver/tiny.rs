//! Thin wrapper around tiny-solver for consistent option handling.

use anyhow::{anyhow, Result};
use nalgebra::DVector;
use std::collections::HashMap;
use tiny_solver::linear::sparse::LinearSolverType;
use tiny_solver::optimizer::{Optimizer, OptimizerOptions};
use tiny_solver::problem::Problem;
use tiny_solver::LevenbergMarquardtOptimizer;

/// User-facing solver options mapped onto tiny-solver's optimizer settings.
#[derive(Clone)]
pub struct TinySolveOptions {
    pub max_iters: usize,
    pub verbosity: usize,
    pub linear_solver: Option<LinearSolverType>,
    pub min_abs_decrease: Option<f64>,
    pub min_rel_decrease: Option<f64>,
    pub min_error: Option<f64>,
}

impl Default for TinySolveOptions {
    fn default() -> Self {
        let defaults = OptimizerOptions::default();
        Self {
            max_iters: defaults.max_iteration,
            verbosity: defaults.verbosity_level,
            linear_solver: Some(defaults.linear_solver_type),
            min_abs_decrease: Some(defaults.min_abs_error_decrease_threshold),
            min_rel_decrease: Some(defaults.min_rel_error_decrease_threshold),
            min_error: Some(defaults.min_error_threshold),
        }
    }
}

impl TinySolveOptions {
    fn to_optimizer_options(&self) -> OptimizerOptions {
        let mut opts = OptimizerOptions::default();
        opts.max_iteration = self.max_iters;
        opts.verbosity_level = self.verbosity;
        if let Some(solver) = self.linear_solver.clone() {
            opts.linear_solver_type = solver;
        }
        if let Some(v) = self.min_abs_decrease {
            opts.min_abs_error_decrease_threshold = v;
        }
        if let Some(v) = self.min_rel_decrease {
            opts.min_rel_error_decrease_threshold = v;
        }
        if let Some(v) = self.min_error {
            opts.min_error_threshold = v;
        }
        opts
    }
}

/// Solve a tiny-solver problem with the given initial values and options.
pub fn solve(
    problem: &Problem,
    initial: HashMap<String, DVector<f64>>,
    opts: &TinySolveOptions,
) -> Result<HashMap<String, DVector<f64>>> {
    let optimizer = LevenbergMarquardtOptimizer::default();
    let options = opts.to_optimizer_options();
    optimizer
        .optimize(problem, &initial, Some(options))
        .ok_or_else(|| anyhow!("tiny-solver failed to converge"))
}
