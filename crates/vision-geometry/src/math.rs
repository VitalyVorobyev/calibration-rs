//! Mathematical utilities for geometric solvers.
//!
//! The low-level numeric primitives — Hartley normalization, polynomial
//! root-finding, homogeneous null-space extraction, and SVD/vector reshaping —
//! live in [`vision_calibration_core::linalg`] (shared with the calibration
//! solvers) and are **re-exported here** so the in-crate `crate::math::…` call
//! sites are unchanged. This module is the canonical home only for the
//! DLT-specific rank guard `dlt_rank_ok`.
//!
//! # References
//!
//! Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.

use vision_calibration_core::Real;

pub use vision_calibration_core::linalg::{
    MathError, NullSpaceSolution, mat3_from_svd_row, mat3_from_vec, mat34_from_svd_row,
    mat34_from_vec, normalize_points_2d, normalize_points_3d, null_space, solve_cubic_real,
    solve_quadratic_real, solve_quartic_real,
};

/// Check that a homogeneous DLT system has a well-defined null space.
///
/// Returns `true` when the design matrix (whose descending singular values are
/// `singular_values`) has a null space of exactly `expected_null_dim` dimensions
/// — i.e. the singular value immediately *above* the null block is meaningfully
/// positive relative to the largest singular value.
///
/// A rank-deficient system (collinear / coincident / degenerate input) causes that
/// boundary singular value to collapse toward zero.
///
/// # Parameters
///
/// - `singular_values`: descending singular values of the design matrix.
/// - `cols`: number of columns in the design matrix (= number of unknowns).
/// - `expected_null_dim`: expected null-space dimension (1 for most DLT systems,
///   2 for the 7-point fundamental solver).
/// - `rel_tol`: minimum acceptable ratio `sv[boundary] / sv[0]`.  A value of
///   `1e-7` is conservative — it only catches near-exact rank deficiency and
///   will not reject valid-but-noisy data.
///
/// # Returns
///
/// `false` when:
/// - `singular_values` is empty, or `sv[0] <= Real::EPSILON` (all-zero matrix),
/// - the boundary index `cols - expected_null_dim - 1` is out of range, or
/// - `sv[boundary] / sv[0] < rel_tol`.
pub(crate) fn dlt_rank_ok(
    singular_values: &[Real],
    cols: usize,
    expected_null_dim: usize,
    rel_tol: Real,
) -> bool {
    if singular_values.is_empty() {
        return false;
    }
    let sv0 = singular_values[0];
    if sv0 <= Real::EPSILON {
        return false;
    }
    // The boundary index is one step above the null block.
    let Some(idx) = cols.checked_sub(expected_null_dim + 1) else {
        return false;
    };
    let Some(&sv_boundary) = singular_values.get(idx) else {
        return false;
    };
    sv_boundary / sv0 >= rel_tol
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clearly full-rank 9-column spectrum (9 unknowns, expected 1-D null
    /// space): sv[7] / sv[0] >> 1e-7 → true.
    #[test]
    fn dlt_rank_ok_full_rank_spectrum() {
        // Simulate a well-conditioned 9-column system: sv[8] ≈ 0 (the null
        // space), all others comfortably above zero.
        let sv: Vec<Real> = vec![10.0, 9.0, 8.5, 7.0, 6.0, 5.5, 4.0, 3.5, 1e-12];
        // cols=9, expected_null_dim=1 → boundary index = 9-1-1 = 7.
        // sv[7] / sv[0] = 3.5 / 10.0 = 0.35 >> 1e-7.
        assert!(
            dlt_rank_ok(&sv, 9, 1, 1e-7),
            "full-rank spectrum must return true"
        );
    }

    /// A degenerate spectrum where the boundary value has collapsed: the
    /// null space is effectively 2-D → false.
    #[test]
    fn dlt_rank_ok_collapsed_boundary_value() {
        // sv[7] and sv[8] are both near-zero: rank is 7, not 8.
        let sv: Vec<Real> = vec![10.0, 9.0, 8.5, 7.0, 6.0, 5.5, 4.0, 1e-14, 1e-15];
        // boundary index = 7 → sv[7] / sv[0] = 1e-14 / 10.0 = 1e-15 < 1e-7.
        assert!(
            !dlt_rank_ok(&sv, 9, 1, 1e-7),
            "collapsed boundary value must return false"
        );
    }

    /// Empty singular values → false.
    #[test]
    fn dlt_rank_ok_empty_sv() {
        assert!(!dlt_rank_ok(&[], 9, 1, 1e-7));
    }

    /// sv[0] ≈ 0 (all-zero matrix) → false.
    #[test]
    fn dlt_rank_ok_zero_matrix() {
        let sv: Vec<Real> = vec![0.0; 9];
        assert!(!dlt_rank_ok(&sv, 9, 1, 1e-7));
    }
}
