//! Least-squares helpers specific to the linear calibration solvers. The
//! shared numeric primitives (Hartley normalization, polynomial roots, null
//! spaces, SVD reshaping, the projection onto SO(3)) are in
//! [`vision_calibration_core::linalg`].
//!
//! - [`ridge_lstsq`] — ridge-regularized normal-equation least squares.

use crate::Error;
use nalgebra::{DMatrix, DVector};
use vision_calibration_core::Real;

/// Solve the overdetermined least-squares system `A x ≈ b` via the
/// ridge-regularized normal equations `x = (AᵀA + λI)⁻¹ Aᵀ b`.
///
/// Like [`null_space`](vision_calibration_core::linalg::null_space), this avoids `A.svd(...)` so it cannot hang on nalgebra's
/// unbounded QR iteration: `AᵀA` is `k×k` (`k = A.ncols()`, small) and is solved
/// by LU. `λ ≥ 0` is a Tikhonov ridge that regularizes a near-singular design
/// the way small-singular-value truncation would in an SVD solve; pass `0.0` for
/// the plain normal equations. A caller typically scales `λ` to the matrix
/// (e.g. `1e-9 * max_diag`) so it is negligible when the design is well-posed.
///
/// # Errors
///
/// Returns [`Error::Singular`] if the (regularized) normal system is non-finite
/// or not solvable.
pub fn ridge_lstsq(
    a: &DMatrix<Real>,
    b: &DVector<Real>,
    lambda: Real,
) -> Result<DVector<Real>, Error> {
    let at = a.transpose();
    let mut ata = &at * a;
    let atb = &at * b;
    if lambda != 0.0 {
        for i in 0..ata.nrows() {
            ata[(i, i)] += lambda;
        }
    }
    if ata.iter().any(|v| !v.is_finite()) || atb.iter().any(|v| !v.is_finite()) {
        return Err(Error::Singular);
    }
    ata.lu().solve(&atb).ok_or(Error::Singular)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ridge_lstsq --------------------------------------------------------

    /// Plain (`λ = 0`) normal equations recover the exact solution of a
    /// consistent overdetermined system.
    #[test]
    fn ridge_lstsq_recovers_exact_solution() {
        // x = [3, -1]. Rows b_i = a_i · x.
        let x_true = DVector::from_vec(vec![3.0, -1.0]);
        let mut a = DMatrix::<Real>::zeros(10, 2);
        let mut b = DVector::<Real>::zeros(10);
        for i in 0..10 {
            a[(i, 0)] = (i as Real) * 0.5 + 1.0;
            a[(i, 1)] = (i as Real).sin();
            b[i] = a[(i, 0)] * x_true[0] + a[(i, 1)] * x_true[1];
        }
        let x = ridge_lstsq(&a, &b, 0.0).unwrap();
        assert!((&x - &x_true).norm() < 1e-9, "lstsq mismatch: {x:?}");
    }

    /// A non-finite design must yield [`Error::Singular`].
    #[test]
    fn ridge_lstsq_rejects_non_finite() {
        let mut a = DMatrix::<Real>::zeros(6, 2);
        a[(0, 0)] = Real::INFINITY;
        let b = DVector::<Real>::zeros(6);
        assert!(matches!(ridge_lstsq(&a, &b, 1e-6), Err(Error::Singular)));
    }
}
