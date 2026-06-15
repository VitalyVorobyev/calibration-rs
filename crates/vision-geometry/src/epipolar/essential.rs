//! Essential matrix estimation using the 5-point algorithm and a linear solver.
//!
//! Implements Nistér's minimal solver for essential matrices from five
//! point correspondences in normalized coordinates, and a linear (≥5-point)
//! overdetermined solver for use in RANSAC refit on large inlier sets.

use super::polynomial::build_polynomial_system;
use crate::math::mat3_from_svd_row;
use anyhow::Result;
use nalgebra::{DMatrix, SMatrix, linalg::Schur};
use vision_calibration_core::{Mat3, Pt2, Real};

/// 5-point algorithm for the essential matrix in normalized coordinates.
///
/// The inputs must be **calibrated** (e.g. apply `K⁻¹` to pixel points).
/// Returns up to ten candidate essential matrices that satisfy the cubic
/// constraints; choose the physically valid one by cheirality or by
/// reprojection error against additional correspondences.
///
/// Unlike fundamental matrix solvers, this does **not** apply Hartley
/// normalization — calibrated coordinates are already well-conditioned.
pub fn essential_5point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>> {
    if pts1.len() != pts2.len() {
        anyhow::bail!(
            "Point count mismatch: expected {}, got {}",
            pts1.len(),
            pts2.len()
        );
    }
    if pts1.len() != 5 {
        anyhow::bail!("Point count mismatch: expected 5, got {}", pts1.len());
    }

    let mut a = DMatrix::<Real>::zeros(5, 9);
    for (i, (p1, p2)) in pts1.iter().zip(pts2.iter()).enumerate() {
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
    let v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
    if v_t.nrows() < 4 {
        anyhow::bail!("SVD failed");
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
        .ok_or(anyhow::anyhow!("Polynomial solve failed"))?;

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

    let action = action.transpose();

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
        let v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
        let vec = v_t.row(v_t.nrows() - 1);

        let v9 = vec[9];
        if v9.abs() < 1e-12 {
            continue;
        }

        let x = vec[6] / v9;
        let y = vec[7] / v9;
        let z_vec = vec[8] / v9;

        let e = e1 * x + e2 * y + e3 * z_vec + e4;

        solutions.push((z_vec, e));
    }

    if solutions.is_empty() {
        anyhow::bail!("Polynomial solve failed");
    }

    solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(solutions.into_iter().map(|(_, e)| e).collect())
}

/// Linear (≥8-point) essential matrix estimator for overdetermined systems.
///
/// Inputs are **calibrated / normalized camera coordinates** (i.e. apply `K⁻¹`
/// first). Requires `pts1.len() == pts2.len() >= 8`.
///
/// **Why ≥8?** The 9-column epipolar design matrix has 9 unknowns. Its null
/// space is 1-dimensional only when `rank(A) == 8`, which requires at least 8
/// linearly independent rows. With 6 or 7 correspondences the null space has
/// dimension ≥ 2 and the smallest-singular-value vector is an arbitrary element
/// of that null space — the result would be a garbage essential matrix. Use
/// [`essential_5point`] for the minimal (5-point) case.
///
/// The algorithm:
/// 1. Build the 9-column epipolar design matrix `A` (same as the 8-point
///    fundamental method, but operating on calibrated coords that are already
///    well-conditioned, so Hartley normalization is omitted).
/// 2. Solve for the null-space vector (right singular vector of smallest
///    singular value) → reshape to 3×3 matrix `E_raw`.
///    When `n == 8` the 8×9 matrix is padded to 9×9 with a zero row so that
///    nalgebra's full SVD returns a 9×9 `Vᵀ`; the zero row leaves the null
///    space unchanged (still 1-dimensional).
/// 3. Project `E_raw` onto the essential manifold: SVD `E_raw = U Σ Vᵀ`;
///    replace singular values with `diag(1, 1, 0)` (up to an overall scale)
///    → `E = U diag(1,1,0) Vᵀ`.
///
/// Returns the projected essential matrix, or `Err` if the SVD fails.
pub fn essential_linear(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3> {
    if pts1.len() != pts2.len() {
        anyhow::bail!(
            "Point count mismatch: pts1 has {}, pts2 has {}",
            pts1.len(),
            pts2.len()
        );
    }
    if pts1.len() < 8 {
        anyhow::bail!(
            "Need at least 8 correspondences for the linear essential solver, got {}",
            pts1.len()
        );
    }

    let n = pts1.len();
    let mut a = DMatrix::<Real>::zeros(n, 9);
    for (i, (p1, p2)) in pts1.iter().zip(pts2.iter()).enumerate() {
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

    // Pad to 9×9 when n == 8 (the only case where nrows < ncols after the ≥8
    // guard above). Adding one zero row leaves the null space unchanged: A has
    // rank 8 → a 1-dimensional null space → the smallest right singular vector
    // is uniquely defined.
    let mut a_work = a.clone();
    if a_work.nrows() < a_work.ncols() {
        let rows = a_work.nrows();
        let cols = a_work.ncols();
        let mut a_pad = DMatrix::<Real>::zeros(cols, cols);
        a_pad.view_mut((0, 0), (rows, cols)).copy_from(&a_work);
        a_work = a_pad;
    }

    let svd = a_work.svd(true, true);
    let v_t = svd.v_t.ok_or_else(|| anyhow::anyhow!("SVD failed"))?;
    let e_vec = v_t.row(v_t.nrows() - 1);

    let mut e = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            e[(r, c)] = e_vec[3 * r + c];
        }
    }

    // Project onto the essential manifold: singular values → (1, 1, 0).
    let svd_e = e.svd(true, true);
    let u = svd_e.u.ok_or_else(|| anyhow::anyhow!("SVD failed on E"))?;
    let v_t_e = svd_e
        .v_t
        .ok_or_else(|| anyhow::anyhow!("SVD failed on E"))?;
    // Average the two largest singular values to enforce the (σ, σ, 0) constraint.
    let sv = svd_e.singular_values;
    let sigma = (sv[0] + sv[1]) * 0.5;
    let s_ess = SMatrix::<Real, 3, 3>::from_diagonal(&nalgebra::Vector3::new(sigma, sigma, 0.0));
    let e_proj = u * s_ess * v_t_e;

    Ok(e_proj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt3, Vec3};

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

        let sols = essential_5point(&pts1, &pts2).unwrap();
        assert!(!sols.is_empty());

        let mut best = f64::INFINITY;
        for e in &sols {
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

        // Also verify decomposition recovers the ground truth pose.
        let mut found_good_decomp = false;
        for e in &sols {
            let decomps = crate::epipolar::decompose_essential(e).unwrap();
            for (r_est, t_est) in &decomps {
                let r_diff: Mat3 = r_est.transpose() * rot.matrix();
                let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
                let ang_deg = cos_theta.acos().to_degrees();
                let cos_t: Real = t_est.normalize().dot(&t.normalize()).abs();
                if ang_deg < 1.0 && cos_t > 0.99 {
                    found_good_decomp = true;
                }
            }
        }
        assert!(
            found_good_decomp,
            "5-point solver did not produce an E that decomposes to the correct pose"
        );
    }

    /// Helper: build synthetic calibrated correspondences from a known rotation + translation.
    fn make_calibrated_corrs(
        rot: &Rotation3<Real>,
        t: &Vec3,
        n_pts: usize,
    ) -> (Vec<Pt2>, Vec<Pt2>) {
        use vision_calibration_core::Pt3;

        // Use a reproducible set of world points.
        let world: Vec<Pt3> = (0..n_pts)
            .map(|i| {
                let fi = i as Real;
                Pt3::new(
                    0.15 * (fi * 0.7).sin(),
                    0.12 * (fi * 0.5).cos(),
                    2.0 + fi * 0.1,
                )
            })
            .collect();

        let pts1: Vec<Pt2> = world
            .iter()
            .map(|pw| Pt2::new(pw.x / pw.z, pw.y / pw.z))
            .collect();
        let pts2: Vec<Pt2> = world
            .iter()
            .map(|pw| {
                let pc2 = rot * pw + t;
                Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z)
            })
            .collect();

        (pts1, pts2)
    }

    #[test]
    fn essential_linear_recovers_known_essential_from_8pts() {
        let rot = Rotation3::from_euler_angles(0.1, -0.05, 0.2);
        let t = Vec3::new(0.1, 0.02, 0.03);

        let (pts1, pts2) = make_calibrated_corrs(&rot, &t, 8);

        let e = essential_linear(&pts1, &pts2).unwrap();

        // Epipolar constraint: |x2^T E x1| must be near-zero for all points.
        let e_norm = e.norm();
        assert!(e_norm > 0.0, "essential_linear returned zero matrix");

        let mut max_residual: Real = 0.0;
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            let x1 = nalgebra::Vector3::new(p1.x, p1.y, 1.0);
            let x2 = nalgebra::Vector3::new(p2.x, p2.y, 1.0);
            let val = (x2.transpose() * e * x1)[0].abs() / e_norm;
            if val > max_residual {
                max_residual = val;
            }
        }
        assert!(
            max_residual < 1e-6,
            "Epipolar residual {:.3e} too large (expected < 1e-6)",
            max_residual
        );

        // Essential manifold: singular values must satisfy (σ, σ, 0).
        let svd = e.svd(false, false);
        let sv = svd.singular_values;
        // Normalized singular values: divide by the average of the two larger.
        let s_avg = (sv[0] + sv[1]) * 0.5;
        assert!(
            (sv[0] / s_avg - 1.0).abs() < 1e-6,
            "sv[0]/avg = {} (expected ~1.0)",
            sv[0] / s_avg
        );
        assert!(
            (sv[1] / s_avg - 1.0).abs() < 1e-6,
            "sv[1]/avg = {} (expected ~1.0)",
            sv[1] / s_avg
        );
        assert!(
            sv[2] / s_avg < 1e-6,
            "sv[2]/avg = {} (expected ~0.0)",
            sv[2] / s_avg
        );
    }

    #[test]
    fn essential_linear_rejects_too_few_points() {
        // The linear solver requires ≥8 correspondences (1-D null space).
        // 7 points leave a ≥2-D null space → must be rejected.
        let pts = vec![Pt2::new(0.0, 0.0); 7];
        assert!(essential_linear(&pts, &pts).is_err());
    }

    #[test]
    fn essential_linear_rejects_mismatched_lengths() {
        let p1 = vec![Pt2::new(0.0, 0.0); 6];
        let p2 = vec![Pt2::new(0.0, 0.0); 5];
        assert!(essential_linear(&p1, &p2).is_err());
    }
}
