//! Homography estimation (plane-induced projective transform).
//!
//! Implements the normalized Direct Linear Transform (DLT) and a robust
//! RANSAC wrapper. The homography `H` maps points from one plane to another:
//! `x' ~ H x`.
//!
//! Input points should be in consistent units; normalization is applied
//! internally for numerical stability and the output is de-normalized.

use crate::math::{mat3_from_vec, normalize_points_2d, null_space};
use crate::{GeometryError, Result};
use nalgebra::DMatrix;
use vision_calibration_core::{
    Estimator, Mat3, Pt2, RansacOptions, from_homogeneous, ransac_fit, to_homogeneous,
};

/// Return `true` if all points are (approximately) collinear.
///
/// Anchors the reference direction on the **first distinct pair** of points
/// (skipping duplicates of `pts[0]`). If all points are identical the set is
/// trivially degenerate and `true` is returned. Otherwise every remaining
/// point is tested: if any cross-product magnitude exceeds `1e-6` the set is
/// not collinear.
///
/// Only three or more points can be collinear, so fewer than 3 points always
/// return `false`.
fn points_are_collinear(pts: &[Pt2]) -> bool {
    if pts.len() < 3 {
        return false;
    }

    let p0 = pts[0];

    // Find the first point that is meaningfully different from pts[0].
    // This avoids a zero baseline when the leading points are duplicates.
    const SAME_PT_EPS: f64 = 1e-9;
    let Some(p1) = pts
        .iter()
        .skip(1)
        .find(|p| {
            let dx = p.x - p0.x;
            let dy = p.y - p0.y;
            (dx * dx + dy * dy).sqrt() > SAME_PT_EPS
        })
        .copied()
    else {
        // All points are identical → degenerate.
        return true;
    };

    // Test every other point against the p0–p1 baseline.
    for p2 in pts.iter() {
        if (p2.x - p0.x).hypot(p2.y - p0.y) <= SAME_PT_EPS {
            continue; // duplicate of p0 — collinear by definition
        }
        let cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if cross.abs() > 1e-6 {
            return false; // found a non-collinear point
        }
    }
    true
}

/// Estimate `H` such that `x' ~ H x` using normalized DLT.
///
/// `src` and `dst` are corresponding 2D points. The returned homography
/// is scaled so that `H[2,2] == 1` when possible.
pub fn dlt_homography(src: &[Pt2], dst: &[Pt2]) -> Result<Mat3> {
    let n = src.len();
    if n < 4 {
        return Err(GeometryError::InsufficientData { need: 4, got: n });
    }
    if dst.len() != n {
        return Err(GeometryError::CountMismatch {
            expected: n,
            got: dst.len(),
        });
    }

    // Reject collinear configurations early: a valid homography requires at
    // least 4 points in general position (no 3 collinear) in BOTH the source
    // and destination sets.  Collinear src makes the design matrix rank < 8;
    // collinear dst yields a degenerate (rank-2) homography that maps the
    // entire plane to a line and cannot be reliably inverted or applied.
    if points_are_collinear(src) {
        return Err(GeometryError::degenerate(
            "rank-deficient point configuration: source points are collinear \
             (need 4 correspondences in general position, no 3 collinear)",
        ));
    }
    if points_are_collinear(dst) {
        return Err(GeometryError::degenerate(
            "rank-deficient point configuration: destination points are collinear \
             (need 4 correspondences in general position, no 3 collinear)",
        ));
    }

    let (src_n, t_w) = normalize_points_2d(src).ok_or_else(|| {
        GeometryError::degenerate("degenerate point configuration for normalization")
    })?;
    let (dst_n, t_i) = normalize_points_2d(dst).ok_or_else(|| {
        GeometryError::degenerate("degenerate point configuration for normalization")
    })?;

    let mut a = DMatrix::<f64>::zeros(2 * n, 9);

    for (i, (pw, pi)) in src_n.iter().zip(dst_n.iter()).enumerate() {
        let x = pw.x;
        let y = pw.y;
        let u = pi.x;
        let v = pi.y;

        let r0 = 2 * i;
        let r1 = 2 * i + 1;

        a[(r0, 0)] = -x;
        a[(r0, 1)] = -y;
        a[(r0, 2)] = -1.0;
        a[(r0, 6)] = u * x;
        a[(r0, 7)] = u * y;
        a[(r0, 8)] = u;

        a[(r1, 3)] = -x;
        a[(r1, 4)] = -y;
        a[(r1, 5)] = -1.0;
        a[(r1, 6)] = v * x;
        a[(r1, 7)] = v * y;
        a[(r1, 8)] = v;
    }

    // Solve `A h = 0` via the `AᵀA` symmetric eigendecomposition (see
    // [`null_space`]). A skip-U dense SVD (`svd(false, true)`) is *insufficient*:
    // nalgebra's non-convergence is in the Golub-Kahan QR sweep itself, which
    // runs regardless of `compute_u`, so on dense 2N×9 targets it can still hang
    // for minutes. `AᵀA` is always 9×9 and its symmetric eigensolve converges
    // quickly; with the Hartley normalization above the squared conditioning is
    // harmless.
    let ns = null_space(&a)?;
    let sv = &ns.singular_values;

    // Rank-deficiency check: for a well-posed homography the 2n×9 design matrix
    // must have rank exactly 8 (a 1-D null space). The singular values are
    // sorted descending.
    if sv[0] <= f64::EPSILON {
        return Err(GeometryError::degenerate(
            "degenerate (all-zero) homography design matrix",
        ));
    }
    // sv[7] is the second-smallest singular value.  For a valid configuration
    // (4 points in general position) it is well above zero. For collinear input
    // the null space is ≥2-D and sv[7] collapses toward zero like sv[8].
    if sv[7] / sv[0] < 1e-7 {
        return Err(GeometryError::degenerate(
            "rank-deficient point configuration: homography is underdetermined \
             (need 4 correspondences in general position, no 3 collinear)",
        ));
    }

    let mut h_mat = mat3_from_vec(&ns.vector);

    let t_i_inv = t_i.try_inverse().ok_or(GeometryError::Singular)?;
    h_mat = t_i_inv * h_mat * t_w;

    let scale = h_mat[(2, 2)];
    if scale.abs() > f64::EPSILON {
        h_mat /= scale;
    }

    Ok(h_mat)
}

/// Estimate a homography using DLT inside a RANSAC loop.
///
/// Returns the best homography and the indices of inliers. The residual is
/// Euclidean reprojection error.
pub fn dlt_homography_ransac(
    src: &[Pt2],
    dst: &[Pt2],
    opts: &RansacOptions,
) -> Result<(Mat3, Vec<usize>)> {
    let n = src.len();
    if n < 4 {
        return Err(GeometryError::InsufficientData { need: 4, got: n });
    }
    if dst.len() != n {
        return Err(GeometryError::CountMismatch {
            expected: n,
            got: dst.len(),
        });
    }

    #[derive(Clone)]
    struct HomographyDatum {
        w: Pt2,
        i: Pt2,
    }

    struct HomographyEst;

    impl Estimator for HomographyEst {
        type Datum = HomographyDatum;
        type Model = Mat3;

        const MIN_SAMPLES: usize = 4;

        fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
            let mut src = Vec::with_capacity(sample_indices.len());
            let mut dst = Vec::with_capacity(sample_indices.len());
            for &idx in sample_indices {
                src.push(data[idx].w);
                dst.push(data[idx].i);
            }
            dlt_homography(&src, &dst).ok()
        }

        fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
            let p = to_homogeneous(&datum.w);
            let proj = model * p;
            let proj_pt = from_homogeneous(&proj);
            let du = proj_pt.x - datum.i.x;
            let dv = proj_pt.y - datum.i.y;
            (du * du + dv * dv).sqrt()
        }

        fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize]) -> bool {
            if sample_indices.len() < 3 {
                return false;
            }
            let p0 = data[sample_indices[0]].w;
            let p1 = data[sample_indices[1]].w;
            let p2 = data[sample_indices[2]].w;
            let area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
            area.abs() < 1e-9
        }

        fn refit(data: &[Self::Datum], inliers: &[usize]) -> Option<Self::Model> {
            if inliers.len() < 4 {
                return None;
            }
            let mut src = Vec::with_capacity(inliers.len());
            let mut dst = Vec::with_capacity(inliers.len());
            for &idx in inliers {
                src.push(data[idx].w);
                dst.push(data[idx].i);
            }
            dlt_homography(&src, &dst).ok()
        }
    }

    let data: Vec<HomographyDatum> = src
        .iter()
        .cloned()
        .zip(dst.iter().cloned())
        .map(|(w, i)| HomographyDatum { w, i })
        .collect();

    let res = ransac_fit::<HomographyEst>(&data, opts);
    if !res.success {
        return Err(GeometryError::NoConsensus);
    }

    let h = res.model.expect("success guarantees a model");
    Ok((h, res.inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::{Pt2, RansacOptions};

    #[test]
    fn basic_homography() {
        let w = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(1.0, 0.0),
            Pt2::new(1.0, 1.0),
            Pt2::new(0.0, 1.0),
        ];
        let img = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(2.0, 0.0),
            Pt2::new(2.0, 2.0),
            Pt2::new(0.0, 2.0),
        ];

        let h = dlt_homography(&w, &img).unwrap();
        let s = h[(0, 0)];
        assert!((s - 2.0).abs() < 1e-6);
    }

    #[test]
    fn dlt_homography_rejects_collinear_src_points() {
        // Source points all on the line y = 2x + 1 → collinear → Err.
        let src: Vec<Pt2> = (0..5)
            .map(|t| {
                let x = t as f64;
                Pt2::new(x, 2.0 * x + 1.0)
            })
            .collect();
        // Destination points are in general position.
        let dst = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(3.0, 1.0),
            Pt2::new(1.0, 4.0),
            Pt2::new(5.0, 2.0),
            Pt2::new(2.0, 6.0),
        ];
        assert!(
            dlt_homography(&src, &dst).is_err(),
            "collinear src points must produce Err"
        );
    }

    #[test]
    fn dlt_homography_rejects_collinear_dst_points() {
        // Destination points all on the line y = 3x.
        // Source points form an irregular quadrilateral + one off-center
        // interior point — all in general position (verified: no 3 collinear).
        let src = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(5.0, 1.0),
            Pt2::new(4.0, 4.0),
            Pt2::new(1.0, 3.0),
            Pt2::new(3.0, 1.5),
        ];
        let dst: Vec<Pt2> = (0..5)
            .map(|t| {
                let x = t as f64;
                Pt2::new(x, 3.0 * x)
            })
            .collect();
        assert!(
            dlt_homography(&src, &dst).is_err(),
            "collinear dst points must produce Err"
        );
    }

    /// Regression: pts[0] == pts[1] must NOT cause a false collinearity verdict.
    ///
    /// The old implementation anchored the reference direction on pts[0]–pts[1].
    /// When those two are duplicates the baseline vector is zero, every cross
    /// product evaluates to zero, and the entire (otherwise general-position)
    /// set was wrongly flagged as collinear — causing `dlt_homography` to
    /// reject valid overdetermined inputs.
    #[test]
    fn dlt_homography_accepts_duplicate_leading_points_in_general_position() {
        // pts[0] and pts[1] are identical; the remaining points form a valid
        // (non-collinear) set together with pts[0].  The duplicate does NOT
        // make the whole set collinear.
        let dup = Pt2::new(0.0, 0.0);
        let src = vec![
            dup,
            dup, // duplicate of pts[0]
            Pt2::new(1.0, 0.0),
            Pt2::new(1.0, 1.0),
            Pt2::new(0.0, 1.0),
        ];
        // Correspondences under a 2× scale homography.
        let dst = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(0.0, 0.0), // duplicate too
            Pt2::new(2.0, 0.0),
            Pt2::new(2.0, 2.0),
            Pt2::new(0.0, 2.0),
        ];
        let h = dlt_homography(&src, &dst)
            .expect("duplicate leading point must not cause false collinearity rejection");
        // The homography should approximate a 2× scale.
        let s = h[(0, 0)];
        assert!((s - 2.0).abs() < 1e-4, "unexpected scale: {}", s);
    }

    /// Regression: an all-identical point set is genuinely degenerate.
    #[test]
    fn dlt_homography_rejects_all_identical_src_points() {
        let same = Pt2::new(1.0, 1.0);
        let src = vec![same; 5];
        let dst = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(3.0, 1.0),
            Pt2::new(1.0, 4.0),
            Pt2::new(5.0, 2.0),
            Pt2::new(2.0, 6.0),
        ];
        assert!(
            dlt_homography(&src, &dst).is_err(),
            "all-identical src points must produce Err"
        );
    }

    /// Dense-target regression guard. A 15×15 grid yields 225 correspondences →
    /// a 450×9 design matrix. The previous `svd(true, true)` drove nalgebra's
    /// Golub-Kahan iteration into a multi-minute hang on inputs this size; the
    /// `AᵀA` eigen-solve path is sub-millisecond.
    /// We assert both exact recovery and a generous wall-clock bound so a future
    /// regression that reinstates the hang fails fast instead of wedging CI.
    #[test]
    fn dense_homography_is_exact_and_fast() {
        use std::time::Instant;
        use vision_calibration_core::Mat3;

        // Ground-truth homography with genuine perspective (h31, h32 ≠ 0).
        let h_gt = Mat3::new(
            1.2, 0.05, 3.0, //
            0.1, 0.9, -2.0, //
            0.0008, -0.0006, 1.0,
        );

        let mut world = Vec::with_capacity(225);
        let mut image = Vec::with_capacity(225);
        for iy in 0..15 {
            for ix in 0..15 {
                let x = ix as f64;
                let y = iy as f64;
                let p = h_gt * nalgebra::Vector3::new(x, y, 1.0);
                world.push(Pt2::new(x, y));
                image.push(Pt2::new(p.x / p.z, p.y / p.z));
            }
        }
        assert_eq!(world.len(), 225);

        let start = Instant::now();
        let h = dlt_homography(&world, &image).unwrap();
        let elapsed = start.elapsed();
        eprintln!("dense 225-correspondence DLT solved in {elapsed:?}");

        // Recovered H is normalized to H[2,2] = 1, matching h_gt's convention.
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (h[(r, c)] - h_gt[(r, c)]).abs() < 1e-6,
                    "H[{r},{c}] = {} != {}",
                    h[(r, c)],
                    h_gt[(r, c)]
                );
            }
        }

        // Generous bound: the fixed path is sub-millisecond; the old hang was
        // >15 min. Anything under 2s confirms the pathological path is gone.
        assert!(
            elapsed.as_secs_f64() < 2.0,
            "dense DLT took {elapsed:?} — perf regression (AᵀA eigen path lost?)"
        );
    }

    #[test]
    fn homography_ransac_handles_outliers() {
        let w = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(1.0, 0.0),
            Pt2::new(1.0, 1.0),
            Pt2::new(0.0, 1.0),
        ];
        let img_inliers = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(2.0, 0.0),
            Pt2::new(2.0, 2.0),
            Pt2::new(0.0, 2.0),
        ];

        let mut w_all = w.clone();
        let mut img_all = img_inliers.clone();
        w_all.push(Pt2::new(0.5, 0.5));
        img_all.push(Pt2::new(10.0, -3.0));
        w_all.push(Pt2::new(-0.2, 0.3));
        img_all.push(Pt2::new(-5.0, 7.0));

        let opts = RansacOptions {
            max_iters: 200,
            thresh: 0.1,
            min_inliers: 4,
            confidence: 0.99,
            seed: 7,
            refit_on_inliers: true,
        };

        let (h, inliers) = dlt_homography_ransac(&w_all, &img_all, &opts).unwrap();

        assert!(inliers.len() >= 4);
        let scale = h[(0, 0)];
        assert!((scale - 2.0).abs() < 1e-2);
    }
}
