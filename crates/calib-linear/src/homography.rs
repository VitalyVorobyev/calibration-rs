use calib_core::{from_homogeneous, ransac, to_homogeneous, Estimator, Mat3, Pt2, RansacOptions};
use nalgebra::{DMatrix, DVector};
use std::cmp::Ordering;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HomographyError {
    #[error("need at least 4 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    #[error("svd failed")]
    SvdFailed,
    #[error("ransac failed to find a consensus homography")]
    RansacFailed,
}

/// Estimate H such that x' ~ H x using DLT.
pub fn dlt_homography(world: &[Pt2], image: &[Pt2]) -> Result<Mat3, HomographyError> {
    let n = world.len();
    if n < 4 || image.len() != n {
        return Err(HomographyError::NotEnoughPoints(n));
    }

    let mut a = DMatrix::<f64>::zeros(2 * n, 9);

    for (i, (pw, pi)) in world.iter().zip(image.iter()).enumerate() {
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

    // Solve A h = 0 by finding the eigenvector of A^T A with the smallest eigenvalue.
    let ata = a.transpose() * &a;
    let eig = ata.symmetric_eigen();
    let min_idx = eig
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| idx)
        .ok_or(HomographyError::SvdFailed)?;
    let h_vec: DVector<f64> = eig.eigenvectors.column(min_idx).into();

    let mut h_mat = Mat3::zeros();
    for r in 0..3 {
        for c in 0..3 {
            h_mat[(r, c)] = h_vec[3 * r + c];
        }
    }

    // normalise such that H[2,2] = 1
    let scale = h_mat[(2, 2)];
    if scale.abs() > f64::EPSILON {
        h_mat /= scale;
    }

    Ok(h_mat)
}

/// Estimate a homography using DLT inside a RANSAC loop.
///
/// Returns the best homography and the indices of inliers.
pub fn dlt_homography_ransac(
    world: &[Pt2],
    image: &[Pt2],
    opts: &RansacOptions,
) -> Result<(Mat3, Vec<usize>), HomographyError> {
    let n = world.len();
    if n < 4 || image.len() != n {
        return Err(HomographyError::NotEnoughPoints(n));
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
            dlt_homography(&world, &image).ok()
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
            dlt_homography(&world, &image).ok()
        }
    }

    let data: Vec<HomographyDatum> = world
        .iter()
        .cloned()
        .zip(image.iter().cloned())
        .map(|(w, i)| HomographyDatum { w, i })
        .collect();

    let res = ransac::<HomographyEst>(&data, opts);
    if !res.success {
        return Err(HomographyError::RansacFailed);
    }

    let h = res.model.expect("success guarantees a model");
    Ok((h, res.inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::Pt2;
    use calib_core::RansacOptions;

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
}
