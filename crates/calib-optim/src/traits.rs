use nalgebra::DMatrix;

/// Generic non-linear least squares problem.
pub trait NllsProblem {
    type Param;     // typically DVector<f64> or small fixed vector
    type Residual;  // DVector<f64>

    fn residuals(&self, x: &Self::Param) -> Self::Residual;
    fn jacobian(&self, x: &Self::Param) -> nalgebra::DMatrix<f64>;
}

pub struct SolveOptions {
    pub max_iters: usize,
    pub tol: f64,
    // later: damping, verbose, etc.
}

pub struct SolveReport {
    pub iterations: usize,
    pub final_cost: f64,
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
