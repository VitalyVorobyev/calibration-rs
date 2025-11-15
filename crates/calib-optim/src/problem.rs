use calib_core::Real;
use nalgebra::{DMatrix, DVector};

pub trait NllsProblem {
    type Param = DVector<Real>;
    type Residual = DVector<Real>;

    fn residuals(&self, x: &Self::Param) -> Self::Residual;
    fn jacobian(&self, x: &Self::Param) -> DMatrix<Real>;
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
        x0: P::Param,
        opts: &SolveOptions,
    ) -> (P::Param, SolveReport);
}
