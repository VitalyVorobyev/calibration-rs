//! Mathematical utilities for geometric solvers.
//!
//! Provides shared mathematical functions used across multiple solvers,
//! including:
//!
//! - **Hartley normalization** for 2D and 3D points (numerical conditioning)
//! - **Polynomial root-finding** for quadratic, cubic, and quartic equations
//! - **SVD matrix extraction** helpers for recovering matrices from SVD results
//!
//! # Hartley Normalization
//!
//! Normalizing points before DLT-style algorithms improves numerical stability
//! by centering the data and scaling to unit variance. This is critical for
//! accurate homography, fundamental matrix, and camera matrix estimation.
//!
//! # References
//!
//! Hartley & Zisserman, "Multiple View Geometry in Computer Vision", 2nd ed.

use nalgebra::{DMatrix, Matrix3x4, Schur};
use vision_calibration_core::{Mat3, Mat4, Pt2, Pt3, Real};

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

/// Extract 3×3 matrix from SVD result row.
///
/// Reshapes a 9-element row from SVD's `V^T` matrix into a 3×3 matrix
/// (row-major order). Commonly used to extract homography or fundamental
/// matrix from the nullspace of a design matrix.
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

/// Extract 3×4 matrix from SVD result row.
///
/// Reshapes a 12-element row from SVD's `V^T` matrix into a 3×4 matrix
/// (row-major order). Commonly used to extract camera projection matrix
/// from DLT nullspace.
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
}
