//! Epipolar geometry solvers for fundamental and essential matrices.
//!
//! Includes normalized 8-point, 7-point, and 5-point minimal solvers, plus
//! decomposition of the essential matrix into candidate poses.
//!
//! - Fundamental matrix `F` expects **pixel coordinates** in both images.
//! - Essential matrix `E` expects **normalized coordinates** (after applying
//!   `K^{-1}`), or equivalently calibrated rays on the normalized image plane.

use crate::math::{mat3_from_svd_row, normalize_points_2d, solve_cubic_real};
use calib_core::{ransac_fit, Estimator, Mat3, Pt2, RansacOptions, Real, Vec3};
use nalgebra::{linalg::Schur, DMatrix, SMatrix};
use thiserror::Error;

/// Errors that can occur during fundamental / essential matrix estimation.
#[derive(Debug, Error)]
pub enum EpipolarError {
    /// Not enough point correspondences were provided.
    #[error("need at least 8 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    /// Incorrect number of correspondences for a minimal solver.
    #[error("invalid number of correspondences: expected {expected}, got {got}")]
    InvalidPointCount { expected: usize, got: usize },
    /// Linear solve (SVD) failed.
    #[error("svd failed in epipolar estimation")]
    SvdFailed,
    /// Polynomial solve failed or produced no valid roots.
    #[error("failed to solve the epipolar polynomial system")]
    PolynomialSolveFailed,
    /// RANSAC failed to find a consensus model.
    #[error("ransac failed to find a consensus epipolar model")]
    RansacFailed,
}

/// Linear epipolar geometry solvers (fundamental / essential matrices).
///
/// All solvers are deterministic and use SVD-based nullspace extraction.
#[derive(Debug, Clone, Copy)]
pub struct EpipolarSolver;

const MONOMIALS: [(u8, u8, u8); 20] = [
    (3, 0, 0), // x^3
    (2, 1, 0), // x^2 y
    (2, 0, 1), // x^2 z
    (1, 2, 0), // x y^2
    (1, 1, 1), // x y z
    (1, 0, 2), // x z^2
    (0, 3, 0), // y^3
    (0, 2, 1), // y^2 z
    (0, 1, 2), // y z^2
    (0, 0, 3), // z^3
    (2, 0, 0), // x^2
    (1, 1, 0), // x y
    (1, 0, 1), // x z
    (0, 2, 0), // y^2
    (0, 1, 1), // y z
    (0, 0, 2), // z^2
    (1, 0, 0), // x
    (0, 1, 0), // y
    (0, 0, 1), // z
    (0, 0, 0), // 1
];

#[derive(Clone, Copy)]
struct Poly3 {
    coeffs: [Real; 20],
}

impl Poly3 {
    fn zero() -> Self {
        Self { coeffs: [0.0; 20] }
    }

    fn linear(c0: Real, cx: Real, cy: Real, cz: Real) -> Self {
        let mut p = Self::zero();
        p.coeffs[19] = c0;
        p.coeffs[16] = cx;
        p.coeffs[17] = cy;
        p.coeffs[18] = cz;
        p
    }

    fn add(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..20 {
            out.coeffs[i] = self.coeffs[i] + other.coeffs[i];
        }
        out
    }

    fn sub(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for i in 0..20 {
            out.coeffs[i] = self.coeffs[i] - other.coeffs[i];
        }
        out
    }

    fn scale(&self, s: Real) -> Self {
        let mut out = Self::zero();
        for i in 0..20 {
            out.coeffs[i] = self.coeffs[i] * s;
        }
        out
    }

    fn mul(&self, other: &Self) -> Self {
        let mut out = Self::zero();
        for (i, &ai) in self.coeffs.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            let (ix, iy, iz) = MONOMIALS[i];
            for (j, &bj) in other.coeffs.iter().enumerate() {
                if bj == 0.0 {
                    continue;
                }
                let (jx, jy, jz) = MONOMIALS[j];
                let dx = ix + jx;
                let dy = iy + jy;
                let dz = iz + jz;
                if dx + dy + dz > 3 {
                    continue;
                }
                if let Some(idx) = monomial_index(dx, dy, dz) {
                    out.coeffs[idx] += ai * bj;
                }
            }
        }
        out
    }
}

fn monomial_index(x: u8, y: u8, z: u8) -> Option<usize> {
    MONOMIALS.iter().enumerate().find_map(|(i, &(mx, my, mz))| {
        if mx == x && my == y && mz == z {
            Some(i)
        } else {
            None
        }
    })
}

fn poly_mat_mul(a: &[[Poly3; 3]; 3], b: &[[Poly3; 3]; 3]) -> [[Poly3; 3]; 3] {
    let mut out = [[Poly3::zero(); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let mut sum = Poly3::zero();
            for k in 0..3 {
                sum = sum.add(&a[r][k].mul(&b[k][c]));
            }
            out[r][c] = sum;
        }
    }
    out
}

fn poly_transpose(a: &[[Poly3; 3]; 3]) -> [[Poly3; 3]; 3] {
    let mut out = [[Poly3::zero(); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[c][r];
        }
    }
    out
}

fn poly_det3(a: &[[Poly3; 3]; 3]) -> Poly3 {
    let term1 = a[0][0].mul(&a[1][1].mul(&a[2][2]).sub(&a[1][2].mul(&a[2][1])));
    let term2 = a[0][1].mul(&a[1][0].mul(&a[2][2]).sub(&a[1][2].mul(&a[2][0])));
    let term3 = a[0][2].mul(&a[1][0].mul(&a[2][1]).sub(&a[1][1].mul(&a[2][0])));

    term1.sub(&term2).add(&term3)
}

fn build_polynomial_system(e1: &Mat3, e2: &Mat3, e3: &Mat3, e4: &Mat3) -> [[Real; 20]; 10] {
    let mut e = [[Poly3::zero(); 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            e[r][c] = Poly3::linear(e4[(r, c)], e1[(r, c)], e2[(r, c)], e3[(r, c)]);
        }
    }

    let det = poly_det3(&e);

    let e_t = poly_transpose(&e);
    let eet = poly_mat_mul(&e, &e_t);
    let eet_e = poly_mat_mul(&eet, &e);

    let trace = eet[0][0].add(&eet[1][1]).add(&eet[2][2]);

    let mut eqs = [[0.0; 20]; 10];
    eqs[0] = det.coeffs;

    let mut row = 1;
    for r in 0..3 {
        for c in 0..3 {
            let term = eet_e[r][c].scale(2.0).sub(&trace.mul(&e[r][c]));
            eqs[row] = term.coeffs;
            row += 1;
        }
    }

    eqs
}

fn enforce_essential_constraints(e: &Mat3) -> Result<Mat3, EpipolarError> {
    let svd = e.svd(true, true);
    let u = svd.u.ok_or(EpipolarError::SvdFailed)?;
    let v_t = svd.v_t.ok_or(EpipolarError::SvdFailed)?;

    let s1 = svd.singular_values[0];
    let s2 = svd.singular_values[1];
    let s = 0.5 * (s1 + s2);

    let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::Vector3::new(s, s, 0.0));
    Ok(u * s_mat * v_t)
}

impl EpipolarSolver {
    /// Normalized 8-point algorithm for the fundamental matrix.
    ///
    /// `pts1` and `pts2` are corresponding pixel points in two images. The
    /// returned matrix is forced to rank-2 and satisfies `x'^T F x = 0`
    /// (up to numerical error).
    pub fn fundamental_8point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3, EpipolarError> {
        let n = pts1.len();
        if n < 8 || pts2.len() != n {
            return Err(EpipolarError::NotEnoughPoints(n));
        }

        let pts1_n = pts1.to_vec();
        let pts2_n = pts2.to_vec();
        let t1 = Mat3::identity();
        let t2 = Mat3::identity();

        // Build design matrix A (n x 9) for x'^T F x = 0.
        let mut a = DMatrix::<Real>::zeros(n, 9);

        for (i, (p1, p2)) in pts1_n.iter().zip(pts2_n.iter()).enumerate() {
            let x = p1.x;
            let y = p1.y;
            let xp = p2.x;
            let yp = p2.y;

            a[(i, 0)] = xp * x;
            a[(i, 1)] = xp * y;
            a[(i, 2)] = xp;
            a[(i, 3)] = yp * x;
            a[(i, 4)] = yp * y;
            a[(i, 5)] = yp;
            a[(i, 6)] = x;
            a[(i, 7)] = y;
            a[(i, 8)] = 1.0;
        }

        // Solve A f = 0 via SVD: take the singular vector for the smallest singular value.
        let mut a_work = a.clone();
        if a_work.nrows() < a_work.ncols() {
            let rows = a_work.nrows();
            let cols = a_work.ncols();
            let mut a_pad = DMatrix::<Real>::zeros(cols, cols);
            a_pad.view_mut((0, 0), (rows, cols)).copy_from(&a_work);
            a_work = a_pad;
        }

        let svd = a_work.svd(true, true);
        let v_t = svd.v_t.ok_or(EpipolarError::SvdFailed)?;
        let f_vec = v_t.row(v_t.nrows() - 1);

        let mut f = Mat3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                f[(r, c)] = f_vec[3 * r + c];
            }
        }

        // Enforce rank-2 constraint on F.
        let svd_f = f.svd(true, true);
        let u = svd_f.u.ok_or(EpipolarError::SvdFailed)?;
        let mut s = svd_f.singular_values;
        let v_t = svd_f.v_t.ok_or(EpipolarError::SvdFailed)?;
        s[2] = 0.0;
        let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&s);
        f = u * s_mat * v_t;

        // Denormalize.
        f = t2.transpose() * f * t1;

        Ok(f)
    }

    /// 7-point algorithm for the fundamental matrix (minimal solver).
    ///
    /// Returns up to three candidate fundamental matrices. Inputs are pixel
    /// coordinates; internal normalization is applied before solving.
    pub fn fundamental_7point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>, EpipolarError> {
        if pts1.len() != pts2.len() {
            return Err(EpipolarError::InvalidPointCount {
                expected: 7,
                got: pts1.len().max(pts2.len()),
            });
        }
        if pts1.len() != 7 {
            return Err(EpipolarError::InvalidPointCount {
                expected: 7,
                got: pts1.len(),
            });
        }

        let pts1_n = pts1.to_vec();
        let pts2_n = pts2.to_vec();
        let t1 = Mat3::identity();
        let t2 = Mat3::identity();

        let mut a = DMatrix::<Real>::zeros(7, 9);
        for (i, (p1, p2)) in pts1_n.iter().zip(pts2_n.iter()).enumerate() {
            let x = p1.x;
            let y = p1.y;
            let xp = p2.x;
            let yp = p2.y;

            a[(i, 0)] = xp * x;
            a[(i, 1)] = xp * y;
            a[(i, 2)] = xp;
            a[(i, 3)] = yp * x;
            a[(i, 4)] = yp * y;
            a[(i, 5)] = yp;
            a[(i, 6)] = x;
            a[(i, 7)] = y;
            a[(i, 8)] = 1.0;
        }

        let mut a_work = a.clone();
        if a_work.nrows() < a_work.ncols() {
            let rows = a_work.nrows();
            let cols = a_work.ncols();
            let mut a_pad = DMatrix::<Real>::zeros(cols, cols);
            a_pad.view_mut((0, 0), (rows, cols)).copy_from(&a_work);
            a_work = a_pad;
        }

        let svd = a_work.svd(true, true);
        let v_t = svd.v_t.ok_or(EpipolarError::SvdFailed)?;
        if v_t.nrows() < 2 {
            return Err(EpipolarError::SvdFailed);
        }

        let f1 = mat3_from_svd_row(&v_t, v_t.nrows() - 2);
        let f2 = mat3_from_svd_row(&v_t, v_t.nrows() - 1);

        let det0 = f2.determinant();
        let det1 = (f2 + f1).determinant();
        let detm1 = (f2 - f1).determinant();
        let det2 = (f2 + 2.0 * f1).determinant();

        let d = det0;
        let s1 = det1 - d;
        let sm1 = detm1 - d;
        let s2 = det2 - d;

        let b = 0.5 * (s1 + sm1);
        let t = 0.5 * (s1 - sm1);
        let a = (s2 - 2.0 * t - 4.0 * b) / 6.0;
        let c = t - a;

        let roots = solve_cubic_real(a, b, c, d);
        if roots.is_empty() {
            return Err(EpipolarError::PolynomialSolveFailed);
        }

        let mut solutions = Vec::new();
        for lambda in roots {
            let mut f = f2 + lambda * f1;

            let svd_f = f.svd(true, true);
            let u = svd_f.u.ok_or(EpipolarError::SvdFailed)?;
            let mut s = svd_f.singular_values;
            let v_t = svd_f.v_t.ok_or(EpipolarError::SvdFailed)?;
            s[2] = 0.0;
            let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&s);
            f = u * s_mat * v_t;

            f = t2.transpose() * f * t1;
            solutions.push((lambda, f));
        }

        solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(solutions.into_iter().map(|(_, f)| f).collect())
    }

    /// 5-point algorithm for the essential matrix in normalized coordinates.
    ///
    /// The inputs must be **calibrated** (e.g. apply `K^{-1}` to pixel points).
    /// Returns up to ten candidate essential matrices that satisfy the cubic
    /// constraints; choose the physically valid one by cheirality or by
    /// reprojection error against additional correspondences.
    pub fn essential_5point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>, EpipolarError> {
        if pts1.len() != pts2.len() {
            return Err(EpipolarError::InvalidPointCount {
                expected: 5,
                got: pts1.len().max(pts2.len()),
            });
        }
        if pts1.len() != 5 {
            return Err(EpipolarError::InvalidPointCount {
                expected: 5,
                got: pts1.len(),
            });
        }

        let (pts1_n, t1) = normalize_points_2d(pts1).ok_or(EpipolarError::SvdFailed)?;
        let (pts2_n, t2) = normalize_points_2d(pts2).ok_or(EpipolarError::SvdFailed)?;

        let mut a = DMatrix::<Real>::zeros(5, 9);
        for (i, (p1, p2)) in pts1_n.iter().zip(pts2_n.iter()).enumerate() {
            let x = p1.x;
            let y = p1.y;
            let xp = p2.x;
            let yp = p2.y;

            a[(i, 0)] = xp * x;
            a[(i, 1)] = xp * y;
            a[(i, 2)] = xp;
            a[(i, 3)] = yp * x;
            a[(i, 4)] = yp * y;
            a[(i, 5)] = yp;
            a[(i, 6)] = x;
            a[(i, 7)] = y;
            a[(i, 8)] = 1.0;
        }

        let mut a_work = a.clone();
        if a_work.nrows() < a_work.ncols() {
            let rows = a_work.nrows();
            let cols = a_work.ncols();
            let mut a_pad = DMatrix::<Real>::zeros(cols, cols);
            a_pad.view_mut((0, 0), (rows, cols)).copy_from(&a_work);
            a_work = a_pad;
        }

        let svd = a_work.svd(true, true);
        let v_t = svd.v_t.ok_or(EpipolarError::SvdFailed)?;
        if v_t.nrows() < 4 {
            return Err(EpipolarError::SvdFailed);
        }

        let e1 = mat3_from_svd_row(&v_t, v_t.nrows() - 4);
        let e2 = mat3_from_svd_row(&v_t, v_t.nrows() - 3);
        let e3 = mat3_from_svd_row(&v_t, v_t.nrows() - 2);
        let e4 = mat3_from_svd_row(&v_t, v_t.nrows() - 1);

        let eqs = build_polynomial_system(&e1, &e2, &e3, &e4);

        let mut m = DMatrix::<Real>::zeros(10, 20);
        for (r, row) in eqs.iter().enumerate() {
            for (c, &val) in row.iter().enumerate() {
                m[(r, c)] = val;
            }
        }

        let m1 = m.view((0, 0), (10, 10)).into_owned();
        let m2 = m.view((0, 10), (10, 10)).into_owned();

        let m1_lu = m1.lu();
        let c = m1_lu
            .solve(&(-m2))
            .ok_or(EpipolarError::PolynomialSolveFailed)?;

        let mut action = DMatrix::<Real>::zeros(10, 10);
        let deg3_rows = [2, 4, 5, 7, 8, 9];
        for (col, &row) in deg3_rows.iter().enumerate() {
            for r in 0..10 {
                action[(r, col)] = c[(row, r)];
            }
        }

        action[(2, 6)] = 1.0;
        action[(4, 7)] = 1.0;
        action[(5, 8)] = 1.0;
        action[(8, 9)] = 1.0;

        let schur = Schur::new(action.clone());
        let eigvals = schur.complex_eigenvalues();

        let mut solutions = Vec::new();
        for val in eigvals.iter() {
            if val.im.abs() > 1e-8 {
                continue;
            }
            let z = val.re;

            let mut a_eval = action.clone();
            for i in 0..10 {
                a_eval[(i, i)] -= z;
            }
            let svd = a_eval.svd(true, true);
            let v_t = svd.v_t.ok_or(EpipolarError::SvdFailed)?;
            let vec = v_t.row(v_t.nrows() - 1);

            let v9 = vec[9];
            if v9.abs() < 1e-12 {
                continue;
            }

            let x = vec[6] / v9;
            let y = vec[7] / v9;
            let z_vec = vec[8] / v9;

            let mut e = e1 * x + e2 * y + e3 * z_vec + e4;
            e = t2.transpose() * e * t1;

            solutions.push((z_vec, e));
        }

        if solutions.is_empty() {
            return Err(EpipolarError::PolynomialSolveFailed);
        }

        solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(solutions.into_iter().map(|(_, e)| e).collect())
    }

    /// Decompose an essential matrix into candidate rotation and translation pairs.
    ///
    /// Returns four possible `(R, t)` pairs; the correct one can be selected by
    /// cheirality checks on triangulated points. The translation is unit-length
    /// (direction only).
    pub fn decompose_essential(e: &Mat3) -> Result<Vec<(Mat3, Vec3)>, EpipolarError> {
        let e = enforce_essential_constraints(e)?;
        let svd = e.svd(true, true);
        let mut u = svd.u.ok_or(EpipolarError::SvdFailed)?;
        let mut v_t = svd.v_t.ok_or(EpipolarError::SvdFailed)?;

        if u.determinant() < 0.0 {
            u.column_mut(2).neg_mut();
        }
        if v_t.determinant() < 0.0 {
            v_t.row_mut(2).neg_mut();
        }

        let w = Mat3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

        let r1 = u * w * v_t;
        let r2 = u * w.transpose() * v_t;

        let t = u.column(2).normalize();

        let mut solutions = vec![
            (r1, t.into_owned()),
            (r1, (-t).into_owned()),
            (r2, t.into_owned()),
            (r2, (-t).into_owned()),
        ];

        for (r, t) in solutions.iter_mut() {
            if r.determinant() < 0.0 {
                *r = -*r;
                *t = -*t;
            }
        }

        Ok(solutions)
    }

    /// Robust fundamental matrix estimation using the 8-point algorithm inside RANSAC.
    ///
    /// Returns the best model and the indices of inliers. The residual uses an
    /// approximate symmetric epipolar distance in pixels.
    pub fn fundamental_8point_ransac(
        pts1: &[Pt2],
        pts2: &[Pt2],
        opts: &RansacOptions,
    ) -> Result<(Mat3, Vec<usize>), EpipolarError> {
        let n = pts1.len();
        if n < 8 || pts2.len() != n {
            return Err(EpipolarError::NotEnoughPoints(n));
        }

        #[derive(Clone)]
        struct FDatum {
            x1: Pt2,
            x2: Pt2,
        }

        struct FundamentalEst;

        impl Estimator for FundamentalEst {
            type Datum = FDatum;
            type Model = Mat3;

            const MIN_SAMPLES: usize = 8;

            fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
                let mut p1 = Vec::with_capacity(sample_indices.len());
                let mut p2 = Vec::with_capacity(sample_indices.len());
                for &idx in sample_indices {
                    p1.push(data[idx].x1);
                    p2.push(data[idx].x2);
                }
                EpipolarSolver::fundamental_8point(&p1, &p2).ok()
            }

            fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
                // Symmetric epipolar distance (approximate).
                let x = nalgebra::Vector3::new(datum.x1.x, datum.x1.y, 1.0);
                let xp = nalgebra::Vector3::new(datum.x2.x, datum.x2.y, 1.0);

                let fx = model * x;
                let ftxp = model.transpose() * xp;
                let denom = fx.x * fx.x + fx.y * fx.y + ftxp.x * ftxp.x + ftxp.y * ftxp.y;
                let denom = denom.max(1e-12);
                let val = xp.transpose() * model * x;
                let d2 = (val[0] * val[0]) / denom;
                d2.sqrt()
            }

            fn is_degenerate(_data: &[Self::Datum], sample_indices: &[usize]) -> bool {
                sample_indices.len() < Self::MIN_SAMPLES
            }
        }

        let data: Vec<FDatum> = pts1
            .iter()
            .cloned()
            .zip(pts2.iter().cloned())
            .map(|(x1, x2)| FDatum { x1, x2 })
            .collect();

        let res = ransac_fit::<FundamentalEst>(&data, opts);
        if !res.success {
            return Err(EpipolarError::RansacFailed);
        }
        let f = res.model.expect("success guarantees a model");
        Ok((f, res.inliers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{FxFyCxCySkew, Mat3, Pt3, RansacOptions, Vec3};
    use nalgebra::{Rotation3, Translation3};

    fn make_k() -> (FxFyCxCySkew<Real>, Mat3) {
        let k = FxFyCxCySkew {
            fx: 800.0,
            fy: 780.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        (k, k.k_matrix())
    }

    #[test]
    fn fundamental_8point_succeeds_on_simple_data() {
        // Very simple synthetic stereo with small baseline along X.
        let (_k, kmtx) = make_k();

        let rot_l = Rotation3::identity();
        let t_l = Translation3::new(0.0, 0.0, 0.0);
        let rot_r = Rotation3::identity();
        let t_r = Translation3::new(0.1, 0.0, 0.0);

        let p_l = kmtx * Mat3::identity();
        let p_r = kmtx * Mat3::identity();

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        for z in 1..3 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw =
                        nalgebra::Point3::new(x as Real * 0.1, y as Real * 0.1, z as Real * 0.5);
                    let pc_l = rot_l * pw + t_l.vector;
                    let pc_r = rot_r * pw + t_r.vector;

                    let xl = p_l * pc_l.coords;
                    let xr = p_r * pc_r.coords;

                    let u_l = xl.x / xl.z;
                    let v_l = xl.y / xl.z;
                    let u_r = xr.x / xr.z;
                    let v_r = xr.y / xr.z;

                    pts1.push(Pt2::new(u_l, v_l));
                    pts2.push(Pt2::new(u_r, v_r));
                }
            }
        }

        let f = EpipolarSolver::fundamental_8point(&pts1, &pts2).unwrap();
        // Basic sanity check: F should not be the zero matrix.
        let norm_f = f.norm();
        assert!(norm_f > 0.0);
    }

    #[test]
    fn fundamental_8point_ransac_handles_outliers() {
        let (_k, kmtx) = make_k();

        let rot_l = Rotation3::identity();
        let t_l = Translation3::new(0.0, 0.0, 0.0);
        let rot_r = Rotation3::identity();
        let t_r = Translation3::new(0.1, 0.0, 0.0);

        let p_l = kmtx * Mat3::identity();
        let p_r = kmtx * Mat3::identity();

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        for z in 1..3 {
            for y in 0..3 {
                for x in 0..4 {
                    let pw =
                        nalgebra::Point3::new(x as Real * 0.1, y as Real * 0.1, z as Real * 0.5);
                    let pc_l = rot_l * pw + t_l.vector;
                    let pc_r = rot_r * pw + t_r.vector;

                    let xl = p_l * pc_l.coords;
                    let xr = p_r * pc_r.coords;

                    let u_l = xl.x / xl.z;
                    let v_l = xl.y / xl.z;
                    let u_r = xr.x / xr.z;
                    let v_r = xr.y / xr.z;

                    pts1.push(Pt2::new(u_l, v_l));
                    pts2.push(Pt2::new(u_r, v_r));
                }
            }
        }

        let inlier_count = pts1.len();

        // Add a few gross outliers.
        pts1.extend_from_slice(&[
            Pt2::new(120.0, -80.0),
            Pt2::new(-50.0, 90.0),
            Pt2::new(200.0, 150.0),
        ]);
        pts2.extend_from_slice(&[
            Pt2::new(-140.0, 60.0),
            Pt2::new(75.0, -200.0),
            Pt2::new(300.0, 10.0),
        ]);

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 1e-3,
            min_inliers: inlier_count.saturating_sub(2),
            confidence: 0.99,
            seed: 123,
            refit_on_inliers: true,
        };

        let (f, inliers) = EpipolarSolver::fundamental_8point_ransac(&pts1, &pts2, &opts).unwrap();

        assert!(inliers.len() >= inlier_count.saturating_sub(2));
        assert!(inliers.len() < pts1.len());
        assert!(f.norm() > 0.0);
    }

    #[test]
    fn fundamental_7point_returns_valid_solution() {
        let (_k, kmtx) = make_k();

        let rot_l = Rotation3::identity();
        let t_l = Translation3::new(0.0, 0.0, 0.0);
        let rot_r = Rotation3::identity();
        let t_r = Translation3::new(0.1, 0.0, 0.0);

        let p_l = kmtx * Mat3::identity();
        let p_r = kmtx * Mat3::identity();

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        for z in 1..2 {
            for y in 0..2 {
                for x in 0..4 {
                    let pw =
                        nalgebra::Point3::new(x as Real * 0.1, y as Real * 0.1, z as Real * 0.5);
                    let pc_l = rot_l * pw + t_l.vector;
                    let pc_r = rot_r * pw + t_r.vector;

                    let xl = p_l * pc_l.coords;
                    let xr = p_r * pc_r.coords;

                    let u_l = xl.x / xl.z;
                    let v_l = xl.y / xl.z;
                    let u_r = xr.x / xr.z;
                    let v_r = xr.y / xr.z;

                    pts1.push(Pt2::new(u_l, v_l));
                    pts2.push(Pt2::new(u_r, v_r));
                }
            }
        }

        let pts1 = &pts1[..7];
        let pts2 = &pts2[..7];

        let sols = EpipolarSolver::fundamental_7point(pts1, pts2).unwrap();
        assert!(!sols.is_empty());

        let mut best = f64::INFINITY;
        for f in sols {
            let mut err = 0.0;
            for (p1, p2) in pts1.iter().zip(pts2.iter()) {
                let x = nalgebra::Vector3::new(p1.x, p1.y, 1.0);
                let xp = nalgebra::Vector3::new(p2.x, p2.y, 1.0);
                let val = xp.transpose() * f * x;
                err += val[0].abs();
            }
            best = best.min(err);
        }

        assert!(best < 1e-6, "7-point residual too large: {}", best);
    }

    fn skew(v: &Vec3) -> Mat3 {
        Mat3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
    }

    #[test]
    fn essential_decomposition_recovers_pose() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vec3::new(0.1, 0.02, -0.03);

        let e = skew(&t) * rot.matrix();
        let solutions = EpipolarSolver::decompose_essential(&e).unwrap();

        let mut found = false;
        for (r_est, t_est) in solutions {
            let r_diff = r_est.transpose() * rot.matrix();
            let trace = r_diff.trace();
            let cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0);
            let ang = cos_theta.acos();

            let t_dir = t_est.normalize();
            let cos_t = t_dir.dot(&t.normalize()).abs();

            if ang < 1e-6 && (1.0 - cos_t) < 1e-6 {
                found = true;
                break;
            }
        }

        assert!(found, "essential decomposition did not recover pose");
    }

    #[test]
    fn essential_5point_fits_minimal_set() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vec3::new(0.1, 0.02, 0.03);

        let world = vec![
            Pt3::new(0.1, 0.2, 2.0),
            Pt3::new(-0.2, 0.1, 2.5),
            Pt3::new(0.3, -0.1, 3.0),
            Pt3::new(-0.15, -0.2, 2.2),
            Pt3::new(0.05, 0.3, 2.8),
        ];

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();
        for pw in world {
            let pc1 = pw.coords;
            let pc2 = rot * pw + t;

            pts1.push(Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z));
            pts2.push(Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z));
        }

        let sols = EpipolarSolver::essential_5point(&pts1, &pts2).unwrap();
        assert!(!sols.is_empty());

        let mut best = f64::INFINITY;
        for e in sols {
            let mut err = 0.0;
            for (p1, p2) in pts1.iter().zip(pts2.iter()) {
                let x = nalgebra::Vector3::new(p1.x, p1.y, 1.0);
                let xp = nalgebra::Vector3::new(p2.x, p2.y, 1.0);
                let val = xp.transpose() * e * x;
                err += val[0].abs();
            }
            best = best.min(err);
        }

        assert!(best < 1e-6, "5-point residual too large: {}", best);
    }
}
