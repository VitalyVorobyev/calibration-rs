//! Mathematical utilities for linear calibration algorithms.
//!
//! The low-level numeric primitives shared with the multiple-view-geometry
//! solvers — Hartley normalization, polynomial root-finding, homogeneous
//! null-space extraction, and SVD/vector reshaping — now live in
//! [`vision_calibration_core::linalg`] and are **re-exported here** so existing
//! `crate::math::…` call sites keep working. This crate is the canonical home
//! only for the helpers below that are calibration-specific:
//!
//! - [`ridge_lstsq`] — ridge-regularized normal-equation least squares.
//! - [`project_to_so3`] — polar projection of a 3×3 matrix onto SO(3).
//!
//! # Example
//!
//! ```
//! use vision_calibration_linear::math::normalize_points_2d;
//! use vision_calibration_core::Pt2;
//!
//! let points = vec![
//!     Pt2::new(100.0, 200.0),
//!     Pt2::new(150.0, 250.0),
//!     Pt2::new(120.0, 220.0),
//! ];
//!
//! let (normalized, _transform) = normalize_points_2d(&points).unwrap();
//! // normalized points have mean at origin, mean distance = sqrt(2)
//! assert_eq!(normalized.len(), 3);
//! ```

use crate::Error;
use nalgebra::{DMatrix, DVector};
use vision_calibration_core::{Mat3, Real};

pub use vision_calibration_core::linalg::{
    NullSpaceSolution, mat3_from_svd_row, mat3_from_vec, mat34_from_svd_row, mat34_from_vec,
    normalize_points_2d, normalize_points_3d, null_space, solve_cubic_real, solve_quadratic_real,
    solve_quartic_real,
};

/// Solve the overdetermined least-squares system `A x ≈ b` via the
/// ridge-regularized normal equations `x = (AᵀA + λI)⁻¹ Aᵀ b`.
///
/// Like [`null_space`], this avoids `A.svd(...)` so it cannot hang on nalgebra's
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

/// Project a 3×3 matrix onto the rotation group SO(3) via polar decomposition.
///
/// Returns the nearest (Frobenius) rotation `R = U Vᵀ` from the SVD `M = U Σ Vᵀ`,
/// with the sign of `Vᵀ`'s last row flipped if needed to force `det(R) = +1`
/// (avoiding a reflection). The SVD here is a fixed 3×3 — it cannot trigger the
/// dense-matrix hang that [`null_space`] guards against — so a direct SVD is
/// used. Centralizes the polar-projection idiom shared by the PnP, hand-eye, and
/// pose solvers.
///
/// # Errors
///
/// Returns [`Error::Singular`] if the SVD factors are unavailable (a non-finite
/// input matrix).
pub fn project_to_so3(m: &Mat3) -> Result<Mat3, Error> {
    let svd = m.svd(true, true);
    let u = svd.u.ok_or(Error::Singular)?;
    let v_t = svd.v_t.ok_or(Error::Singular)?;
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        let mut u_fixed = u;
        let last = u_fixed.ncols() - 1;
        for row in 0..u_fixed.nrows() {
            u_fixed[(row, last)] = -u_fixed[(row, last)];
        }
        r = u_fixed * v_t;
    }
    Ok(r)
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

    // ---- project_to_so3 -----------------------------------------------------

    /// A slightly perturbed rotation projects back to a proper rotation
    /// (orthonormal, det = +1) close to the original.
    #[test]
    fn project_to_so3_recovers_rotation() {
        use nalgebra::Rotation3;
        let r = *Rotation3::from_euler_angles(0.2, -0.35, 0.1).matrix();
        let mut noisy = r;
        noisy[(0, 1)] += 0.02;
        noisy[(2, 0)] -= 0.015;
        let proj = project_to_so3(&noisy).unwrap();
        assert!((proj.determinant() - 1.0).abs() < 1e-9, "det != 1");
        assert!(
            (proj * proj.transpose() - Mat3::identity()).norm() < 1e-9,
            "not orthonormal"
        );
        assert!((proj - r).norm() < 0.05, "projection drifted from input");
    }

    /// A reflection (det < 0) is corrected to a proper rotation (det = +1).
    #[test]
    fn project_to_so3_fixes_reflection() {
        // diag(1, 1, -1) is an orthogonal reflection with det = -1.
        let refl = Mat3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0);
        let proj = project_to_so3(&refl).unwrap();
        assert!((proj.determinant() - 1.0).abs() < 1e-9, "det != 1");
    }
}
