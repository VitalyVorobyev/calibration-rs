use calib_core::Real;
use nalgebra::{DMatrix, DVector};

/// Generic non-linear least squares problem with dense parameter/residual vectors.
pub trait NllsProblem {
    fn residuals(&self, x: &DVector<Real>) -> DVector<Real>;
    fn jacobian(&self, x: &DVector<Real>) -> DMatrix<Real>;
}

#[derive(Debug, Clone, Copy)]
pub struct SolveOptions {
    /// Maximum number of solver iterations before termination.
    ///
    /// Backends may interpret this as a function-evaluation cap; the LM backend
    /// follows the MINPACK convention `max_iters * (n + 1)`.
    pub max_iters: usize,
    /// Relative tolerance on the objective (cost) reduction.
    pub ftol: Real,
    /// Orthogonality/gradient tolerance.
    pub gtol: Real,
    /// Relative tolerance on parameter updates.
    pub xtol: Real,
    /// Enable verbose solver logging if supported by the backend.
    pub verbose: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            ftol: 1e-10,
            gtol: 1e-10,
            xtol: 1e-10,
            verbose: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SolveReport {
    pub iterations: usize,
    pub final_cost: Real,
    pub converged: bool,
}

pub trait NllsSolverBackend {
    fn solve<P: NllsProblem>(
        &self,
        problem: &P,
        x0: DVector<Real>,
        opts: &SolveOptions,
    ) -> (DVector<Real>, SolveReport);
}
