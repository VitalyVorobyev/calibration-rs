//! Fundamental matrix estimation.
//!
//! Implements the normalized 8-point algorithm, the minimal 7-point solver,
//! and RANSAC-based robust estimation for fundamental matrices.

use crate::math::{
    dlt_rank_ok, mat3_from_svd_row, mat3_from_vec, normalize_points_2d, null_space,
    solve_cubic_real,
};
use crate::{GeometryError, Result};
use nalgebra::{DMatrix, SMatrix};
use vision_calibration_core::{Estimator, Mat3, Pt2, RansacOptions, Real, ransac_fit};

/// Normalized 8-point algorithm for the fundamental matrix.
///
/// `pts1` and `pts2` are corresponding pixel points in two images. The
/// returned matrix is forced to rank-2 and satisfies `x'^T F x = 0`
/// (up to numerical error).
pub fn fundamental_8point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3> {
    let n = pts1.len();
    if n < 8 {
        return Err(GeometryError::InsufficientData { need: 8, got: n });
    }
    if pts2.len() != n {
        return Err(GeometryError::CountMismatch {
            expected: n,
            got: pts2.len(),
        });
    }

    let (pts1_n, t1) = normalize_points_2d(pts1).ok_or_else(|| {
        GeometryError::degenerate("Degenerate point set (pts1): collinear or coincident")
    })?;
    let (pts2_n, t2) = normalize_points_2d(pts2).ok_or_else(|| {
        GeometryError::degenerate("Degenerate point set (pts2): collinear or coincident")
    })?;

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

    // Solve `A f = 0` via `AᵀA` symmetric eigen (see [`null_space`]); `AᵀA` is
    // always 9×9 so the wide `n == 8` row-padding is unnecessary, and it cannot
    // trigger nalgebra's hang-prone dense SVD.
    let ns = null_space(&a)?;

    // The 9-column epipolar design matrix must have rank exactly 8 (1-D null
    // space).  Collinear or coincident points collapse sv[7] toward zero.
    if !dlt_rank_ok(&ns.singular_values, 9, 1, 1e-7) {
        return Err(GeometryError::degenerate(
            "rank-deficient 8-point system: points are collinear or degenerate \
             (need 8 points in general position)",
        ));
    }

    let mut f = mat3_from_vec(&ns.vector);

    let svd_f = f.svd(true, true);
    let u = svd_f
        .u
        .ok_or_else(|| GeometryError::numerical("SVD failed"))?;
    let mut s = svd_f.singular_values;
    let v_t = svd_f
        .v_t
        .ok_or_else(|| GeometryError::numerical("SVD failed"))?;
    s[2] = 0.0;
    let s_mat = SMatrix::<Real, 3, 3>::from_diagonal(&s);
    f = u * s_mat * v_t;

    // De-normalize: F_raw = T2^T * F_norm * T1
    f = t2.transpose() * f * t1;

    Ok(f)
}

/// 7-point algorithm for the fundamental matrix (minimal solver).
///
/// Returns up to three candidate fundamental matrices. Inputs are pixel
/// coordinates; internal normalization is applied before solving.
pub fn fundamental_7point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>> {
    if pts1.len() != pts2.len() {
        return Err(GeometryError::CountMismatch {
            expected: pts1.len(),
            got: pts2.len(),
        });
    }
    if pts1.len() != 7 {
        return Err(GeometryError::InsufficientData {
            need: 7,
            got: pts1.len(),
        });
    }

    let (pts1_n, t1) = normalize_points_2d(pts1).ok_or_else(|| {
        GeometryError::degenerate("Degenerate point set (pts1): collinear or coincident")
    })?;
    let (pts2_n, t2) = normalize_points_2d(pts2).ok_or_else(|| {
        GeometryError::degenerate("Degenerate point set (pts2): collinear or coincident")
    })?;

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
    let sv: Vec<Real> = svd.singular_values.iter().cloned().collect();
    let v_t = svd
        .v_t
        .ok_or_else(|| GeometryError::numerical("SVD failed"))?;
    if v_t.nrows() < 2 {
        return Err(GeometryError::numerical(
            "SVD failed: not enough nullspace vectors",
        ));
    }

    // The 7-point method intentionally uses a 2-D null space (it solves a
    // cubic over two null vectors).  The 7th singular value (index 6) must
    // be meaningfully positive — if it collapses the input is degenerate
    // (e.g. all 7 points collinear) and both null vectors are arbitrary.
    if !dlt_rank_ok(&sv, 9, 2, 1e-7) {
        return Err(GeometryError::degenerate(
            "rank-deficient 7-point system: points are collinear or degenerate \
             (need 7 points in general position)",
        ));
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
        return Err(GeometryError::numerical("Polynomial solve failed"));
    }

    let mut solutions = Vec::new();
    for lambda in roots {
        let mut f = f2 + lambda * f1;

        let svd_f = f.svd(true, true);
        let u = svd_f
            .u
            .ok_or_else(|| GeometryError::numerical("SVD failed"))?;
        let mut s = svd_f.singular_values;
        let v_t = svd_f
            .v_t
            .ok_or_else(|| GeometryError::numerical("SVD failed"))?;
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
    if n < 8 {
        return Err(GeometryError::InsufficientData { need: 8, got: n });
    }
    if pts2.len() != n {
        return Err(GeometryError::CountMismatch {
            expected: n,
            got: pts2.len(),
        });
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
        return Err(GeometryError::NoConsensus);
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

    /// Regression test: Hartley normalization is required for pixel coordinates.
    ///
    /// Without normalization the x'x columns (hundreds×hundreds) dominate the
    /// 9-column design matrix, the SVD is ill-conditioned, and F fails the
    /// epipolar constraint by orders of magnitude.  With normalization the
    /// per-point residual |x2^T F x1| / ||F|| must be < 1e-6.
    #[test]
    fn fundamental_8point_epipolar_constraint_pixel_coords() {
        let (_k, kmtx) = make_k();

        // Camera 1 at origin, camera 2 translated along X by 0.5 m.
        let rot_r = Rotation3::from_euler_angles(0.0, 0.0, 0.0);
        let t_r = Translation3::new(0.5, 0.0, 0.0);

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        // Points at pixel scale (hundreds to ~1280×720 range).
        for z in 1..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let pw = nalgebra::Point3::new(
                        x as Real * 0.4 - 0.6,
                        y as Real * 0.3 - 0.45,
                        z as Real * 1.0 + 1.5,
                    );
                    // Camera 1 = world frame
                    let xl = kmtx * pw.coords;
                    // Camera 2 = shifted by t_r
                    let pc_r = rot_r * pw + t_r.vector;
                    let xr = kmtx * pc_r.coords;

                    pts1.push(Pt2::new(xl.x / xl.z, xl.y / xl.z));
                    pts2.push(Pt2::new(xr.x / xr.z, xr.y / xr.z));
                }
            }
        }

        // Pixel coordinates land in [~400, ~900] range — this is the regime
        // where unnormalized DLT is numerically unstable.
        assert!(pts1[0].x > 100.0, "Expected pixel-scale points");

        let f = fundamental_8point(&pts1, &pts2).unwrap();
        let f_norm = f.norm();
        assert!(f_norm > 0.0);

        // Verify epipolar constraint |x2^T F x1| / ||F|| is tight for all pts.
        let mut max_residual: Real = 0.0;
        for (p1, p2) in pts1.iter().zip(pts2.iter()) {
            let x1 = nalgebra::Vector3::new(p1.x, p1.y, 1.0);
            let x2 = nalgebra::Vector3::new(p2.x, p2.y, 1.0);
            let val = (x2.transpose() * f * x1)[0];
            let rel = val.abs() / f_norm;
            if rel > max_residual {
                max_residual = rel;
            }
        }
        assert!(
            max_residual < 1e-6,
            "Epipolar residual |x2^T F x1| / ||F|| = {:.3e} (expected < 1e-6). \
             This would be >> 1e-6 without Hartley normalization.",
            max_residual
        );
    }

    #[test]
    fn fundamental_8point_succeeds_on_simple_data() {
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

    /// Regression: 8 points all on a single image line in both views must return Err.
    ///
    /// Collinear configurations collapse the 8th singular value of the 9-column
    /// design matrix toward zero (the null space becomes ≥2-D), so the last
    /// singular vector is arbitrary.  The rank check must catch this.
    #[test]
    fn fundamental_8point_rejects_collinear_points() {
        // Both image point sets on a single line: y = x.
        let pts: Vec<Pt2> = (0..8)
            .map(|i| {
                let x = i as Real * 50.0 + 100.0;
                Pt2::new(x, x)
            })
            .collect();
        // Shift the second image set slightly along the same line direction.
        let pts2: Vec<Pt2> = pts
            .iter()
            .map(|p| Pt2::new(p.x + 10.0, p.y + 10.0))
            .collect();
        assert!(
            fundamental_8point(&pts, &pts2).is_err(),
            "collinear 8-point input must return Err"
        );
    }

    #[test]
    fn fundamental_7point_returns_valid_solution() {
        let (_k, kmtx) = make_k();

        // Use a non-coplanar set of 7 world points (varying x, y, AND z) so
        // that the epipolar design matrix has rank 7 (2-D null space).
        // A pure-z grid (all points at the same depth) is coplanar and yields a
        // degenerate system for which dlt_rank_ok correctly returns false.
        let rot_r = Rotation3::from_euler_angles(0.0, 0.05, 0.0);
        let t_r = Translation3::new(0.15, 0.0, 0.0);

        let p_l = kmtx * Mat3::identity();

        let world: Vec<nalgebra::Point3<Real>> = vec![
            nalgebra::Point3::new(0.0, 0.0, 2.0),
            nalgebra::Point3::new(0.2, 0.0, 2.3),
            nalgebra::Point3::new(-0.1, 0.15, 2.5),
            nalgebra::Point3::new(0.3, -0.1, 1.8),
            nalgebra::Point3::new(-0.2, 0.2, 3.0),
            nalgebra::Point3::new(0.1, -0.2, 2.7),
            nalgebra::Point3::new(-0.3, -0.1, 1.5),
        ];

        let mut pts1 = Vec::new();
        let mut pts2 = Vec::new();

        for pw in &world {
            let xl = p_l * pw.coords;
            let pc_r = rot_r * pw + t_r.vector;
            let xr = kmtx * pc_r.coords;

            pts1.push(Pt2::new(xl.x / xl.z, xl.y / xl.z));
            pts2.push(Pt2::new(xr.x / xr.z, xr.y / xr.z));
        }

        let sols = fundamental_7point(&pts1, &pts2).unwrap();
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

    /// Regression: 7 points all on a single image line must return Err.
    ///
    /// Collinear 7-point input produces a rank-≤6 design matrix (vs the
    /// required rank 7 for a 2-D null space), so the two "null" vectors are
    /// arbitrary and the cubic has no meaningful roots.  The rank check
    /// (`dlt_rank_ok` with `expected_null_dim=2`) must catch this.
    #[test]
    fn fundamental_7point_rejects_collinear_points() {
        // All 7 points on the line y = 2x in both views.
        let pts: Vec<Pt2> = (0..7)
            .map(|i| {
                let x = i as Real * 40.0 + 80.0;
                Pt2::new(x, 2.0 * x)
            })
            .collect();
        let pts2: Vec<Pt2> = pts
            .iter()
            .map(|p| Pt2::new(p.x + 15.0, p.y + 30.0))
            .collect();
        assert!(
            fundamental_7point(&pts, &pts2).is_err(),
            "collinear 7-point input must return Err"
        );
    }
}
