use calib_core::Real;
use nalgebra::{DMatrix, DVector};

/// Generic non-linear least squares problem with dense parameter/residual vectors.
pub trait NllsProblem {
    fn residuals(&self, x: &DVector<Real>) -> DVector<Real>;
    fn jacobian(&self, x: &DVector<Real>) -> DMatrix<Real>;
}

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub max_iters: usize,
    pub ftol: Real,
    pub gtol: Real,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            max_iters: 50,
            ftol: 1e-12,
            gtol: 1e-12,
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
