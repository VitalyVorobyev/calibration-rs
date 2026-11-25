//! Homography estimation (plane-induced projective transform).
//!
//! Implements the normalized Direct Linear Transform (DLT); the robust
//! (RANSAC) estimate is `vision_mvg::robust::estimate_homography`. The homography `H` maps points from one plane to another:
//! `x' ~ H x`.
//!
//! Input points should be in consistent units; normalization is applied
//! internally for numerical stability and the output is de-normalized.

use crate::math::{dlt_rank_ok, ensure_finite};
use crate::{GeometryError, Result};
use nalgebra::DMatrix;
use vision_calibration_core::linalg::{mat3_from_vec, normalize_points_2d, null_space};
use vision_calibration_core::{Mat3, Pt2};

/// Return `true` if all normalized points are (approximately) collinear.
///
/// Must be called on Hartley-normalized coordinates (mean distance `√2`) so
/// the cross-product threshold is scale-invariant (small metric targets like
/// 1 mm × 1 mm grids are not misclassified as collinear).
fn points_are_collinear(pts: &[Pt2]) -> bool {
    if pts.len() < 3 {
        return false;
    }

    let p0 = pts[0];

    const SAME_PT_EPS: f64 = 1e-9;
    let Some(p1) = pts
        .iter()
        .skip(1)
        .find(|p| (p.x - p0.x).hypot(p.y - p0.y) > SAME_PT_EPS)
        .copied()
    else {
        return true;
    };

    let d1 = (p1.x - p0.x).hypot(p1.y - p0.y);
    for p2 in pts.iter() {
        let d2 = (p2.x - p0.x).hypot(p2.y - p0.y);
        if d2 <= SAME_PT_EPS {
            continue;
        }
        let cross = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if cross.abs() > 1e-6 * d1 * d2 {
            return false;
        }
    }
    true
}

/// Estimate `H` such that `x' ~ H x` using normalized DLT.
///
/// `src` and `dst` are corresponding 2D points. The returned homography
/// is scaled so that `H[2,2] == 1` when possible.
pub fn dlt_homography(src: &[Pt2], dst: &[Pt2]) -> Result<Mat3> {
    ensure_finite("source point", src.iter().flat_map(|p| p.coords.iter()))?;
    ensure_finite(
        "destination point",
        dst.iter().flat_map(|p| p.coords.iter()),
    )?;
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

    let (src_n, t_w) = normalize_points_2d(src).ok_or_else(|| {
        GeometryError::degenerate("degenerate point configuration for normalization")
    })?;
    let (dst_n, t_i) = normalize_points_2d(dst).ok_or_else(|| {
        GeometryError::degenerate("degenerate point configuration for normalization")
    })?;

    // Reject collinear configurations on the normalized coordinates: a valid
    // homography requires at least 4 points in general position (no 3 collinear)
    // in BOTH sets.
    if points_are_collinear(&src_n) {
        return Err(GeometryError::degenerate(
            "rank-deficient point configuration: source points are collinear \
             (need 4 correspondences in general position, no 3 collinear)",
        ));
    }
    if points_are_collinear(&dst_n) {
        return Err(GeometryError::degenerate(
            "rank-deficient point configuration: destination points are collinear \
             (need 4 correspondences in general position, no 3 collinear)",
        ));
    }

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
    if !dlt_rank_ok(&ns.singular_values, 9, 1, 1e-7) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::Pt2;

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
    fn dlt_homography_accepts_millimeter_scale_targets() {
        let src = vec![
            Pt2::new(0.0, 0.0),
            Pt2::new(1e-3, 0.0),
            Pt2::new(1e-3, 1e-3),
            Pt2::new(0.0, 1e-3),
        ];
        let dst = vec![
            Pt2::new(100.0, 200.0),
            Pt2::new(200.0, 200.0),
            Pt2::new(200.0, 300.0),
            Pt2::new(100.0, 300.0),
        ];
        let h = dlt_homography(&src, &dst).expect("1mm square should not be treated as collinear");
        let p = h * nalgebra::Vector3::new(1e-3, 1e-3, 1.0);
        assert!((p.x / p.z - 200.0).abs() < 1e-6);
        assert!((p.y / p.z - 300.0).abs() < 1e-6);
    }
}
