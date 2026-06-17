//! Homography estimation (plane-induced projective transform).
//!
//! Implements the normalized Direct Linear Transform (DLT) and a robust
//! RANSAC wrapper. The homography `H` maps **world/board points** on a plane
//! to **image points** in pixels: `x' ~ H x`.
//!
//! Input points should be in consistent units; normalization is applied
//! internally for numerical stability and the output is de-normalized.

use crate::Error;
use crate::math::normalize_points_2d;
use nalgebra::DMatrix;
use vision_calibration_core::{
    Estimator, Mat3, Pt2, RansacOptions, from_homogeneous, ransac_fit, to_homogeneous,
};

/// High-level entry point for homography estimation.
///
/// This is a thin wrapper around the DLT and DLT+RANSAC helpers in this
/// module and is provided mainly for API consistency with other solvers.
#[derive(Debug, Clone, Copy)]
pub struct HomographySolver;

/// Estimate `H` such that `x' ~ H x` using normalized DLT.
///
/// `world` are planar points in a board or world coordinate frame, and `image`
/// are their pixel coordinates. The returned homography is scaled so that
/// `H[2,2] == 1` when possible.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if fewer than 4 correspondences are given,
/// or [`Error::Singular`] if the configuration is degenerate.
pub fn dlt_homography(world: &[Pt2], image: &[Pt2]) -> Result<Mat3, Error> {
    HomographySolver::dlt(world, image)
}

impl HomographySolver {
    /// Estimate a homography `H` such that `x' ~ H x` using the normalized DLT.
    ///
    /// This uses Hartley-style point normalization (zero-mean, average distance
    /// sqrt(2)) and solves `A h = 0` via SVD on the design matrix `A`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 4 correspondences are given,
    /// or [`Error::Singular`] if the configuration is degenerate.
    pub fn dlt(world: &[Pt2], image: &[Pt2]) -> Result<Mat3, Error> {
        let n = world.len();
        if n < 4 || image.len() != n {
            return Err(Error::InsufficientData { need: 4, got: n });
        }

        let (world_n, t_w) = normalize_points_2d(world).ok_or(Error::Singular)?;
        let (image_n, t_i) = normalize_points_2d(image).ok_or(Error::Singular)?;

        let mut a = DMatrix::<f64>::zeros(2 * n, 9);

        for (i, (pw, pi)) in world_n.iter().zip(image_n.iter()).enumerate() {
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

        // Solve `A h = 0` as the smallest-eigenvalue eigenvector of the 9×9
        // normal matrix `AᵀA`. We deliberately avoid `A.svd(...)`: nalgebra's
        // Golub-Kahan SVD runs an *unbounded* QR iteration (`max_niter = 0`) that
        // fails to converge on some real detected-corner matrices — observed
        // hanging for minutes on dense targets, in `delimit_subproblem`, even
        // with `compute_u = false`. `AᵀA` is always 9×9 regardless of the
        // correspondence count, and a symmetric eigendecomposition of it
        // converges in a handful of sweeps; with the Hartley normalization above
        // the squared conditioning is harmless. This mirrors OpenCV's
        // `findHomography` DLT, which accumulates a 9×9 `LtL` and eigen-solves it.
        let ata = a.transpose() * &a;
        let eigen = ata.symmetric_eigen();
        // A well-posed (Hartley-normalized) configuration yields all-finite
        // eigenvalues. A non-finite one means a degenerate view (e.g. coincident
        // or wildly out-of-range correspondences) — report it as singular rather
        // than panicking on a NaN comparison.
        if eigen.eigenvalues.iter().any(|v| !v.is_finite()) {
            return Err(Error::Singular);
        }
        let min_idx = eigen
            .eigenvalues
            .iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| x.partial_cmp(y).expect("eigenvalues checked finite"))
            .map(|(idx, _)| idx)
            .ok_or(Error::Singular)?;
        let h_vec = eigen.eigenvectors.column(min_idx);

        let mut h_mat = Mat3::zeros();
        for r in 0..3 {
            for c in 0..3 {
                h_mat[(r, c)] = h_vec[3 * r + c];
            }
        }

        let t_i_inv = t_i.try_inverse().ok_or(Error::Singular)?;
        h_mat = t_i_inv * h_mat * t_w;

        // normalise such that H[2,2] = 1
        let scale = h_mat[(2, 2)];
        if scale.abs() > f64::EPSILON {
            h_mat /= scale;
        }

        Ok(h_mat)
    }
}

/// Estimate a homography using DLT inside a RANSAC loop.
///
/// Returns the best homography and the indices of inliers.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] if fewer than 4 correspondences are given,
/// or [`Error::Numerical`] if RANSAC fails to find a consensus.
pub fn dlt_homography_ransac(
    world: &[Pt2],
    image: &[Pt2],
    opts: &RansacOptions,
) -> Result<(Mat3, Vec<usize>), Error> {
    HomographySolver::dlt_ransac(world, image, opts)
}

impl HomographySolver {
    /// Estimate a homography using DLT inside a RANSAC loop.
    ///
    /// Returns the best homography and the indices of inliers. The residual is
    /// Euclidean reprojection error in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 4 correspondences are given,
    /// or [`Error::Numerical`] if RANSAC fails to find a consensus.
    pub fn dlt_ransac(
        world: &[Pt2],
        image: &[Pt2],
        opts: &RansacOptions,
    ) -> Result<(Mat3, Vec<usize>), Error> {
        let n = world.len();
        if n < 4 || image.len() != n {
            return Err(Error::InsufficientData { need: 4, got: n });
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
                let mut world = Vec::with_capacity(sample_indices.len());
                let mut image = Vec::with_capacity(sample_indices.len());
                for &idx in sample_indices {
                    world.push(data[idx].w);
                    image.push(data[idx].i);
                }
                HomographySolver::dlt(&world, &image).ok()
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
                let mut world = Vec::with_capacity(inliers.len());
                let mut image = Vec::with_capacity(inliers.len());
                for &idx in inliers {
                    world.push(data[idx].w);
                    image.push(data[idx].i);
                }
                HomographySolver::dlt(&world, &image).ok()
            }
        }

        let data: Vec<HomographyDatum> = world
            .iter()
            .cloned()
            .zip(image.iter().cloned())
            .map(|(w, i)| HomographyDatum { w, i })
            .collect();

        let res = ransac_fit::<HomographyEst>(&data, opts);
        if !res.success {
            return Err(Error::numerical(
                "ransac failed to find a consensus homography",
            ));
        }

        let h = res.model.expect("success guarantees a model");
        Ok((h, res.inliers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_core::Pt2;
    use vision_calibration_core::RansacOptions;

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

        // Add a couple of outlier correspondences
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

        // The inliers should at least include the four good correspondences.
        assert!(inliers.len() >= 4);
        let scale = h[(0, 0)];
        assert!((scale - 2.0).abs() < 1e-2);
    }

    /// Dense-target regression guard. A 15×15 grid yields 225 correspondences →
    /// a 450×9 design matrix. The previous `svd(true, true)` drove nalgebra's
    /// Golub-Kahan iteration into a multi-minute hang on inputs this size; the
    /// `svd(false, true)` (skip-U) fix solves it in well under a millisecond.
    /// We assert both exact recovery and a generous wall-clock bound so a future
    /// regression that reinstates the hang fails fast instead of wedging CI.
    #[test]
    fn dense_homography_is_exact_and_fast() {
        use std::time::Instant;

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
            "dense DLT took {elapsed:?} — perf regression (skip-U path lost?)"
        );
    }
}
