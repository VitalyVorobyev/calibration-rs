//! Backend adapters that compile the IR into solver-specific problems.

mod tiny_solver_backend;

use anyhow::{anyhow, Result};
use nalgebra::DVector;
use std::collections::HashMap;

use crate::ir::ProblemIR;

pub use tiny_solver_backend::TinySolverBackend;

/// Backend-agnostic solver options.
#[derive(Debug, Clone)]
pub struct BackendSolveOptions {
    pub max_iters: usize,
    pub verbosity: usize,
    pub linear_solver: Option<LinearSolverKind>,
    pub min_abs_decrease: Option<f64>,
    pub min_rel_decrease: Option<f64>,
    pub min_error: Option<f64>,
}

impl Default for BackendSolveOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            verbosity: 0,
            linear_solver: Some(LinearSolverKind::SparseCholesky),
            min_abs_decrease: Some(1e-5),
            min_rel_decrease: Some(1e-5),
            min_error: Some(1e-10),
        }
    }
}

/// Linear solver selection (backend-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverKind {
    SparseCholesky,
    SparseQR,
}

/// Solver output from a backend.
#[derive(Debug, Clone)]
pub struct BackendSolution {
    pub params: HashMap<String, DVector<f64>>,
    pub final_cost: f64,
}

/// Backend interface implemented by solver adapters.
pub trait OptimBackend {
    fn solve(
        &self,
        ir: &ProblemIR,
        initial: &HashMap<String, DVector<f64>>,
        opts: &BackendSolveOptions,
    ) -> Result<BackendSolution>;
}

/// Supported solver backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    TinySolver,
    Ceres,
}

/// Solve a problem using the selected backend.
pub fn solve_with_backend(
    backend: BackendKind,
    ir: &ProblemIR,
    initial: &HashMap<String, DVector<f64>>,
    opts: &BackendSolveOptions,
) -> Result<BackendSolution> {
    match backend {
        BackendKind::TinySolver => TinySolverBackend.solve(ir, initial, opts),
        BackendKind::Ceres => Err(anyhow!("Ceres backend not implemented")),
    }
}
