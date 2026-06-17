//! Low-level linear-algebra primitives shared across the workspace.
//!
//! These are foundational numeric building blocks — Hartley normalization,
//! polynomial root-finding, homogeneous null-space extraction, and SVD/vector
//! reshaping — used by both the calibration solvers (`vision-calibration-linear`)
//! and the multiple-view-geometry solvers (`vision-geometry`). They live here, in
//! the shared foundation crate, so neither downstream crate has to depend on the
//! other to reach them.
//!
//! # Hartley normalization
//!
//! Normalizing points before DLT-style algorithms improves numerical stability
//! by centering the data and scaling to unit mean distance. This is critical for
//! accurate homography, fundamental-matrix, and camera-matrix estimation.
//!
//! # References
//!
//! Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed.

use crate::{Mat3, Mat4, Pt2, Pt3, Real};
use nalgebra::{DMatrix, DVector, Matrix3x4, Schur};

/// Errors returned by the fallible primitives in this module.
///
/// Typed (rather than stringly `anyhow`) so downstream crates that re-export
/// these primitives can map the failure into their own error enums without
/// losing the variant.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MathError {
    /// A matrix inversion or decomposition produced a degenerate result
    /// (a non-finite / NaN design, or no usable decomposition).
    #[error("singular matrix or degenerate configuration")]
    Singular,
}

/// Hartley normalization for 2D points.
///
/// Centers points at the origin and scales so that the mean distance from
/// the origin is `√2`. This conditioning improves numerical stability in
/// DLT-style algorithms.
///
/// Returns `Some((normalized_points, transform_matrix))` where `T` is a 3×3
/// matrix such that `p_norm = T * p_homogeneous`, or `None` if input is empty
/// or all points coincide.
pub fn normalize_points_2d(points: &[Pt2]) -> Option<(Vec<Pt2>, Mat3)> {
    if points.is_empty() {
        return None;
    }

    let n = points.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in points {
        cx += p.x;
        cy += p.y;
    }
    cx /= n;
    cy /= n;

    let mut mean_dist = 0.0;
    for p in points {
        let dx = p.x - cx;
        let dy = p.y - cy;
        mean_dist += (dx * dx + dy * dy).sqrt();
    }
    mean_dist /= n;

    if mean_dist <= f64::EPSILON {
        return None;
    }

    let scale = (2.0_f64).sqrt() / mean_dist;
    let t = Mat3::new(
        scale,
        0.0,
        -scale * cx,
        0.0,
        scale,
        -scale * cy,
        0.0,
        0.0,
        1.0,
    );

    let norm = points
        .iter()
        .map(|p| Pt2::new((p.x - cx) * scale, (p.y - cy) * scale))
        .collect();

    Some((norm, t))
}

/// Hartley normalization for 3D points.
///
/// Centers points at the origin and scales so that the mean distance from
/// the origin is `√3`. This is the 3D analog of [`normalize_points_2d`].
///
/// Returns `Some((normalized_points, transform_matrix))` where `T` is a 4×4
/// matrix, or `None` if input is empty or all points coincide.
pub fn normalize_points_3d(points: &[Pt3]) -> Option<(Vec<Pt3>, Mat4)> {
    if points.is_empty() {
        return None;
    }

    let n = points.len() as Real;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in points {
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    cx /= n;
    cy /= n;
    cz /= n;

    let mut mean_dist = 0.0;
    for p in points {
        let dx = p.x - cx;
        let dy = p.y - cy;
        let dz = p.z - cz;
        mean_dist += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    mean_dist /= n;

    if mean_dist <= Real::EPSILON {
        return None;
    }

    let scale = (3.0_f64).sqrt() / mean_dist;
    let t = Mat4::new(
        scale,
        0.0,
        0.0,
        -scale * cx,
        0.0,
        scale,
        0.0,
        -scale * cy,
        0.0,
        0.0,
        scale,
        -scale * cz,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let norm = points
        .iter()
        .map(|p| Pt3::new((p.x - cx) * scale, (p.y - cy) * scale, (p.z - cz) * scale))
        .collect();

    Some((norm, t))
}

/// Solve quadratic equation `ax² + bx + c = 0` for real roots.
///
/// Returns sorted, deduplicated real roots.
pub fn solve_quadratic_real(a: Real, b: Real, c: Real) -> Vec<Real> {
    let eps = 1e-12;
    if a.abs() < eps {
        if b.abs() < eps {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc.abs() < eps {
        return vec![-b / (2.0 * a)];
    }
    if disc < 0.0 {
        return Vec::new();
    }
    let sqrt_disc = disc.sqrt();
    let r1 = (-b + sqrt_disc) / (2.0 * a);
    let r2 = (-b - sqrt_disc) / (2.0 * a);
    if (r1 - r2).abs() < 1e-8 {
        vec![r1]
    } else {
        let mut roots = vec![r1, r2];
        roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        roots
    }
}

/// Solve cubic equation `ax³ + bx² + cx + d = 0` for real roots.
///
/// Uses Cardano's formula with depressed cubic transformation.
/// Returns sorted, deduplicated real roots.
pub fn solve_cubic_real(a: Real, b: Real, c: Real, d: Real) -> Vec<Real> {
    let eps = 1e-12;
    if a.abs() < eps {
        return solve_quadratic_real(b, c, d);
    }

    let a_inv = 1.0 / a;
    let b = b * a_inv;
    let c = c * a_inv;
    let d = d * a_inv;

    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;

    let disc = (q * 0.5) * (q * 0.5) + (p / 3.0) * (p / 3.0) * (p / 3.0);
    let shift = b / 3.0;

    let mut roots = Vec::new();
    if disc > eps {
        let sqrt_disc = disc.sqrt();
        let u = (-q * 0.5 + sqrt_disc).signum() * (-q * 0.5 + sqrt_disc).abs().powf(1.0 / 3.0);
        let v = (-q * 0.5 - sqrt_disc).signum() * (-q * 0.5 - sqrt_disc).abs().powf(1.0 / 3.0);
        roots.push(u + v - shift);
    } else if disc.abs() <= eps {
        let u = (-q * 0.5).signum() * (-q * 0.5).abs().powf(1.0 / 3.0);
        roots.push(2.0 * u - shift);
        roots.push(-u - shift);
    } else {
        let r = (-p / 3.0).sqrt();
        let phi = ((-q * 0.5) / (r * r * r)).clamp(-1.0, 1.0).acos();
        let two_r = 2.0 * r;
        roots.push(two_r * (phi / 3.0).cos() - shift);
        roots.push(two_r * ((phi + 2.0 * std::f64::consts::PI) / 3.0).cos() - shift);
        roots.push(two_r * ((phi + 4.0 * std::f64::consts::PI) / 3.0).cos() - shift);
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    roots
}

/// Solve quartic equation `ax⁴ + bx³ + cx² + dx + e = 0` for real roots.
///
/// Uses companion matrix eigenvalue method via Schur decomposition.
/// Returns sorted, deduplicated real roots.
pub fn solve_quartic_real(a: Real, b: Real, c: Real, d: Real, e: Real) -> Vec<Real> {
    let eps = 1e-12;
    if a.abs() < eps {
        return solve_cubic_real(b, c, d, e);
    }

    let mut comp = DMatrix::<Real>::zeros(4, 4);
    comp[(0, 0)] = -b / a;
    comp[(0, 1)] = -c / a;
    comp[(0, 2)] = -d / a;
    comp[(0, 3)] = -e / a;
    comp[(1, 0)] = 1.0;
    comp[(2, 1)] = 1.0;
    comp[(3, 2)] = 1.0;

    let schur = Schur::new(comp);
    let eigvals = schur.complex_eigenvalues();

    let mut roots = Vec::new();
    for val in eigvals.iter() {
        if val.im.abs() < 1e-8 {
            roots.push(val.re);
        }
    }

    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    roots
}

/// Solution of a homogeneous least-squares ([`null_space`]) solve.
#[derive(Debug, Clone)]
pub struct NullSpaceSolution {
    /// Unit right-singular vector of `A` with the smallest singular value — the
    /// minimizer of `‖A x‖` subject to `‖x‖ = 1`.
    pub vector: DVector<Real>,
    /// Singular values of `A` in descending order (`σ_i = √λ_i` of `AᵀA`).
    /// Useful for rank / conditioning guards.
    pub singular_values: Vec<Real>,
}

/// Solve the homogeneous least-squares problem `A x = 0` for the unit vector
/// `x` that minimizes `‖A x‖` — the right-singular vector of `A` with the
/// smallest singular value.
///
/// Computed as the smallest-eigenvalue eigenvector of the normal matrix `AᵀA`
/// via a **symmetric eigendecomposition**, deliberately *not* `A.svd(...)`.
/// nalgebra's Golub-Kahan SVD runs an unbounded QR iteration that can fail to
/// converge — hanging for minutes — on real dense design matrices, and the
/// non-convergence is in the QR sweep itself, so it persists even with
/// `compute_u = false`. `AᵀA` is always `k×k` where `k = A.ncols()` (≤ 12 for
/// every DLT system in this workspace) regardless of the row count, and a
/// symmetric eigendecomposition of such a small matrix converges in a handful
/// of sweeps. With Hartley-normalized inputs the squared conditioning is
/// harmless.
///
/// The returned [`NullSpaceSolution`] also carries the descending singular
/// values (derived from the `AᵀA` eigenvalues) for callers that need a rank
/// guard.
///
/// # Errors
///
/// Returns [`MathError::Singular`] if `AᵀA` is non-finite (a degenerate / NaN
/// design) or the decomposition yields no eigenvalues.
pub fn null_space(a: &DMatrix<Real>) -> Result<NullSpaceSolution, MathError> {
    let ata = a.transpose() * a;
    let eigen = ata.symmetric_eigen();
    if eigen.eigenvalues.iter().any(|v| !v.is_finite()) {
        return Err(MathError::Singular);
    }
    let min_idx = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, x), (_, y)| x.partial_cmp(y).expect("eigenvalues checked finite"))
        .map(|(idx, _)| idx)
        .ok_or(MathError::Singular)?;
    let vector = eigen.eigenvectors.column(min_idx).into_owned();
    let mut singular_values: Vec<Real> = eigen
        .eigenvalues
        .iter()
        .map(|&l| l.max(0.0).sqrt())
        .collect();
    singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    Ok(NullSpaceSolution {
        vector,
        singular_values,
    })
}

/// Reshape a 9-element vector into a 3×3 matrix (row-major).
///
/// The vector-valued analog of [`mat3_from_svd_row`], for use with the
/// null vector returned by [`null_space`].
pub fn mat3_from_vec(v: &DVector<Real>) -> Mat3 {
    let mut m = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            m[(r, c)] = v[3 * r + c];
        }
    }
    m
}

/// Reshape a 12-element vector into a 3×4 matrix (row-major).
///
/// The vector-valued analog of [`mat34_from_svd_row`], for use with the
/// null vector returned by [`null_space`].
pub fn mat34_from_vec(v: &DVector<Real>) -> Matrix3x4<Real> {
    let mut m = Matrix3x4::<Real>::zeros();
    for r in 0..3 {
        for c in 0..4 {
            m[(r, c)] = v[4 * r + c];
        }
    }
    m
}

/// Extract 3×3 matrix from an SVD result row.
///
/// Reshapes a 9-element row from SVD's `Vᵀ` matrix into a 3×3 matrix
/// (row-major order). Commonly used to extract a homography or fundamental
/// matrix from the null space of a design matrix.
pub fn mat3_from_svd_row(v_t: &DMatrix<Real>, row_idx: usize) -> Mat3 {
    assert_eq!(
        v_t.ncols(),
        9,
        "Expected 9 columns for 3x3 matrix extraction"
    );
    let mut m = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            m[(r, c)] = v_t[(row_idx, 3 * r + c)];
        }
    }
    m
}

/// Extract 3×4 matrix from an SVD result row.
///
/// Reshapes a 12-element row from SVD's `Vᵀ` matrix into a 3×4 matrix
/// (row-major order). Commonly used to extract a camera projection matrix
/// from a DLT null space.
pub fn mat34_from_svd_row(v_t: &DMatrix<Real>, row_idx: usize) -> Matrix3x4<Real> {
    assert_eq!(
        v_t.ncols(),
        12,
        "Expected 12 columns for 3x4 matrix extraction"
    );
    let mut m = Matrix3x4::<Real>::zeros();
    for r in 0..3 {
        for c in 0..4 {
            m[(r, c)] = v_t[(row_idx, 4 * r + c)];
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_2d_centering() {
        let points = vec![
            Pt2::new(100.0, 200.0),
            Pt2::new(200.0, 300.0),
            Pt2::new(150.0, 250.0),
        ];

        let (norm, _t) = normalize_points_2d(&points).unwrap();

        let cx: f64 = norm.iter().map(|p| p.x).sum::<f64>() / norm.len() as f64;
        let cy: f64 = norm.iter().map(|p| p.y).sum::<f64>() / norm.len() as f64;
        assert!(cx.abs() < 1e-10, "Centroid x not at origin: {}", cx);
        assert!(cy.abs() < 1e-10, "Centroid y not at origin: {}", cy);

        let mean_dist: f64 = norm
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .sum::<f64>()
            / norm.len() as f64;
        assert!(
            (mean_dist - 2.0_f64.sqrt()).abs() < 1e-10,
            "Mean distance not sqrt(2): {}",
            mean_dist
        );
    }

    #[test]
    fn normalize_3d_centering() {
        let points = vec![
            Pt3::new(1.0, 2.0, 3.0),
            Pt3::new(4.0, 5.0, 6.0),
            Pt3::new(7.0, 8.0, 9.0),
        ];

        let (norm, _t) = normalize_points_3d(&points).unwrap();

        let cx: f64 = norm.iter().map(|p| p.x).sum::<f64>() / norm.len() as f64;
        let cy: f64 = norm.iter().map(|p| p.y).sum::<f64>() / norm.len() as f64;
        let cz: f64 = norm.iter().map(|p| p.z).sum::<f64>() / norm.len() as f64;

        assert!(cx.abs() < 1e-10);
        assert!(cy.abs() < 1e-10);
        assert!(cz.abs() < 1e-10);

        let mean_dist: f64 = norm
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y + p.z * p.z).sqrt())
            .sum::<f64>()
            / norm.len() as f64;
        assert!((mean_dist - 3.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn quadratic_two_roots() {
        let roots = solve_quadratic_real(1.0, -3.0, 2.0);
        assert_eq!(roots.len(), 2);
        assert!((roots[0] - 1.0).abs() < 1e-10);
        assert!((roots[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn quadratic_one_root() {
        let roots = solve_quadratic_real(1.0, -2.0, 1.0);
        assert_eq!(roots.len(), 1);
        assert!((roots[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn quadratic_no_roots() {
        let roots = solve_quadratic_real(1.0, 0.0, 1.0);
        assert_eq!(roots.len(), 0);
    }

    #[test]
    fn cubic_three_roots() {
        let roots = solve_cubic_real(1.0, -6.0, 11.0, -6.0);
        assert_eq!(roots.len(), 3);
        assert!((roots[0] - 1.0).abs() < 1e-6);
        assert!((roots[1] - 2.0).abs() < 1e-6);
        assert!((roots[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn quartic_two_roots() {
        let roots = solve_quartic_real(1.0, 0.0, -5.0, 0.0, 4.0);
        assert_eq!(roots.len(), 4);
        assert!((roots[0] - -2.0).abs() < 1e-6);
        assert!((roots[1] - -1.0).abs() < 1e-6);
        assert!((roots[2] - 1.0).abs() < 1e-6);
        assert!((roots[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn svd_extraction_3x3() {
        let mut v_t = DMatrix::zeros(9, 9);
        for i in 0..9 {
            v_t[(8, i)] = (i + 1) as f64;
        }

        let m = mat3_from_svd_row(&v_t, 8);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(2, 2)], 9.0);
    }

    #[test]
    fn svd_extraction_3x4() {
        let mut v_t = DMatrix::zeros(12, 12);
        for i in 0..12 {
            v_t[(11, i)] = (i + 1) as f64;
        }

        let m = mat34_from_svd_row(&v_t, 11);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 3)], 4.0);
        assert_eq!(m[(1, 0)], 5.0);
        assert_eq!(m[(2, 3)], 12.0);
    }

    /// Rows spanning the orthogonal complement of a known unit vector recover
    /// that vector (up to sign) as the null space, with descending singular
    /// values.
    #[test]
    fn null_space_recovers_known_vector() {
        let target = DVector::from_vec(vec![1.0, -2.0, 0.5]);
        let target_unit = &target / target.norm();
        let mut a = DMatrix::<Real>::zeros(40, 3);
        for i in 0..40 {
            let t = i as Real * 0.31;
            // Both orthogonal to target=[1,-2,0.5]: e1 and target×e1.
            let e1 = DVector::from_vec(vec![2.0, 1.0, 0.0]);
            let e2 = DVector::from_vec(vec![-0.5, 1.0, 5.0]);
            let row = e1 * t.cos() + e2 * t.sin();
            for c in 0..3 {
                a[(i, c)] = row[c];
            }
        }
        let ns = null_space(&a).unwrap();
        let v = &ns.vector / ns.vector.norm();
        let diff: DVector<Real> = &v - &target_unit;
        let sum: DVector<Real> = &v + &target_unit;
        assert!(diff.norm().min(sum.norm()) < 1e-6);
        assert!(ns.singular_values.windows(2).all(|w| w[0] >= w[1]));
    }

    /// A non-finite design yields an error, never a hang or panic.
    #[test]
    fn null_space_rejects_non_finite() {
        let mut a = DMatrix::<Real>::zeros(8, 3);
        a[(0, 0)] = Real::NAN;
        assert!(null_space(&a).is_err());
    }
}
