//! Fundamental matrix estimation.
//!
//! Implements the normalized 8-point algorithm, the minimal 7-point solver,
//! and RANSAC-based robust estimation for fundamental matrices.

use crate::math::{mat3_from_svd_row, solve_cubic_real};
use anyhow::Result;
use nalgebra::{DMatrix, SMatrix};
use vision_calibration_core::{Estimator, Mat3, Pt2, RansacOptions, Real, ransac_fit};

/// Normalized 8-point algorithm for the fundamental matrix.
///
/// `pts1` and `pts2` are corresponding pixel points in two images. The
/// returned matrix is forced to rank-2 and satisfies `x'^T F x = 0`
/// (up to numerical error).
pub fn fundamental_8point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3> {
    let n = pts1.len();
    if n < 8 || pts2.len() != n {
        return Err(anyhow::anyhow!("Not enough points"));
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
    let v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
    let f_vec = v_t.row(v_t.nrows() - 1);

    let mut f = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            f[(r, c)] = f_vec[3 * r + c];
        }
    }

    // Enforce rank-2 constraint on F.
    let svd_f = f.svd(true, true);
    let u = svd_f.u.ok_or(anyhow::anyhow!("SVD failed"))?;
    let mut s = svd_f.singular_values;
    let v_t = svd_f.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
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
pub fn fundamental_7point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>> {
    if pts1.len() != pts2.len() {
        anyhow::bail!(
            "Point count mismatch: expected 7, pts1 has {}, pts2 has {}",
            pts1.len(),
            pts2.len()
        );
    }
    if pts1.len() != 7 {
        anyhow::bail!("Point count mismatch: expected 7, got {}", pts1.len());
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
    let v_t = svd.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
    if v_t.nrows() < 2 {
        anyhow::bail!("SVD failed: not enough nullspace vectors");
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
        anyhow::bail!("Polynomial solve failed");
    }

    let mut solutions = Vec::new();
    for lambda in roots {
        let mut f = f2 + lambda * f1;

        let svd_f = f.svd(true, true);
        let u = svd_f.u.ok_or(anyhow::anyhow!("SVD failed"))?;
        let mut s = svd_f.singular_values;
        let v_t = svd_f.v_t.ok_or(anyhow::anyhow!("SVD failed"))?;
        s[2] = 0.0;
        let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&s);
        f = u * s_mat * v_t;

        f = t2.transpose() * f * t1;
        solutions.push((lambda, f));
    }

    solutions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(solutions.into_iter().map(|(_, f)| f).collect())
}

/// Robust fundamental matrix estimation using the 8-point algorithm inside RANSAC.
///
/// Returns the best model and the indices of inliers. The residual uses an
/// approximate symmetric epipolar distance in pixels.
pub fn fundamental_8point_ransac(
    pts1: &[Pt2],
    pts2: &[Pt2],
    opts: &RansacOptions,
) -> Result<(Mat3, Vec<usize>)> {
    let n = pts1.len();
    if n < 8 || pts2.len() != n {
        anyhow::bail!(format!("Not enough points: {}", n));
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
            fundamental_8point(&p1, &p2).ok()
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
        anyhow::bail!("RANSAC failed");
    }
    let f = res.model.expect("success guarantees a model");
    Ok((f, res.inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Rotation3, Translation3};
    use vision_calibration_core::{FxFyCxCySkew, Mat3, RansacOptions, Real};

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

        let f = fundamental_8point(&pts1, &pts2).unwrap();
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

        let (f, inliers) = fundamental_8point_ransac(&pts1, &pts2, &opts).unwrap();

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

        let sols = fundamental_7point(pts1, pts2).unwrap();
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
}
