//! Essential matrix estimation using the 5-point algorithm.
//!
//! Implements Nistér's minimal solver for essential matrices from five
//! point correspondences in normalized coordinates.

use super::polynomial::build_polynomial_system;
use crate::math::{mat3_from_svd_row, normalize_points_2d};
use anyhow::Result;
use calib_core::{Mat3, Pt2, Real};
use nalgebra::{linalg::Schur, DMatrix};

/// 5-point algorithm for the essential matrix in normalized coordinates.
///
/// The inputs must be **calibrated** (e.g. apply `K^{-1}` to pixel points).
/// Returns up to ten candidate essential matrices that satisfy the cubic
/// constraints; choose the physically valid one by cheirality or by
/// reprojection error against additional correspondences.
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

    let (pts1_n, t1) = normalize_points_2d(pts1)
        .ok_or(anyhow::anyhow!("SVD failed during point normalization"))?;
    let (pts2_n, t2) = normalize_points_2d(pts2)
        .ok_or(anyhow::anyhow!("SVD failed during point normalization"))?;

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

        let mut e = e1 * x + e2 * y + e3 * z_vec + e4;
        e = t2.transpose() * e * t1;

        solutions.push((z_vec, e));
    }

    if solutions.is_empty() {
        anyhow::bail!("Polynomial solve failed");
    }

    solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(solutions.into_iter().map(|(_, e)| e).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::{Pt3, Vec3};
    use nalgebra::Rotation3;

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
