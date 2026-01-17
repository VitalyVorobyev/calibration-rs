//! Backend adapters that compile the IR into solver-specific problems.
//!
//! Backends are responsible for translating the IR into solver-native graphs,
//! applying manifolds and constraints, and returning a solved parameter map.

mod tiny_solver_backend;
mod tiny_solver_manifolds;

use anyhow::{anyhow, Result};
use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::ir::ProblemIR;

pub use tiny_solver_backend::TinySolverBackend;

/// Backend-agnostic solver options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSolveOptions {
    /// Maximum number of iterations for the optimizer.
    pub max_iters: usize,
    /// Verbosity level (backend-specific).
    pub verbosity: usize,
    /// Optional linear solver selection.
    pub linear_solver: Option<LinearSolverKind>,
    /// Absolute error decrease threshold for early termination.
    pub min_abs_decrease: Option<f64>,
    /// Relative error decrease threshold for early termination.
    pub min_rel_decrease: Option<f64>,
    /// Error threshold for early termination.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinearSolverKind {
    /// Sparse Cholesky decomposition.
    SparseCholesky,
    /// Sparse QR decomposition.
    SparseQR,
}

/// Solver output from a backend.
///
/// The `params` map uses the IR parameter block names.
#[derive(Debug, Clone)]
pub struct BackendSolution {
    /// Optimized parameter vectors keyed by block name.
    pub params: HashMap<String, DVector<f64>>,
    /// Final robustified cost if supported by the backend.
    pub final_cost: f64,
}

/// Backend interface implemented by solver adapters.
pub trait OptimBackend {
    /// Solve a compiled IR with the provided initial parameters.
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
    /// tiny-solver Levenberg-Marquardt backend.
    TinySolver,
    /// Placeholder for a Ceres backend.
    Ceres,
}

/// Solve a problem using the selected backend.
///
/// This is the main backend-agnostic entry point used by problems.
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
