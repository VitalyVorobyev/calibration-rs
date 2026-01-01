use calib_core::Real;
use nalgebra::{DMatrix, DVector};

/// Generic non-linear least squares problem with dense parameter/residual vectors.
///
/// The default implementations apply robust IRLS row scaling without differentiating
/// the weights: residuals and Jacobian rows are scaled by `sqrt(w_i)` computed from
/// unweighted residuals.
pub trait NllsProblem {
    /// Number of parameters in the optimization vector.
    fn num_params(&self) -> usize;
    /// Number of residual rows in the problem.
    fn num_residuals(&self) -> usize;

    /// Unweighted residuals for the current parameters.
    fn residuals_unweighted(&self, x: &DVector<Real>) -> DVector<Real>;
    /// Unweighted Jacobian for the current parameters.
    fn jacobian_unweighted(&self, x: &DVector<Real>) -> DMatrix<Real>;

    /// Per-row IRLS scales (sqrt(weights)) computed from unweighted residuals.
    fn robust_row_scales(&self, r_unweighted: &DVector<Real>) -> DVector<Real> {
        DVector::from_element(r_unweighted.len(), 1.0)
    }

    /// Weighted residuals used by the solver.
    fn residuals(&self, x: &DVector<Real>) -> DVector<Real> {
        let mut r = self.residuals_unweighted(x);
        let scales = self.robust_row_scales(&r);
        debug_assert_eq!(scales.len(), r.len());
        r.component_mul_assign(&scales);
        r
    }

    /// Weighted Jacobian used by the solver.
    fn jacobian(&self, x: &DVector<Real>) -> DMatrix<Real> {
        let r_unweighted = self.residuals_unweighted(x);
        let scales = self.robust_row_scales(&r_unweighted);
        let mut j = self.jacobian_unweighted(x);
        debug_assert_eq!(scales.len(), j.nrows());
        for (mut row, scale) in j.row_iter_mut().zip(scales.iter()) {
            if *scale != 1.0 {
                row.scale_mut(*scale);
            }
        }
        j
    }
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
            max_iters: 200,
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
