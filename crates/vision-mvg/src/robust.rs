//! Robust estimation using RANSAC for epipolar and homography models.
//!
//! Provides RANSAC-based estimators that work with [`Correspondence2D`]:
//!
//! - [`estimate_essential`]: 5-point algorithm + Sampson distance
//! - [`estimate_homography`]: 4-point DLT + symmetric transfer error
//! - [`recover_relative_pose_robust`]: full pipeline from correspondences to pose

use crate::residuals;
use crate::types::Correspondence2D;
use anyhow::Result;
use vision_calibration_core::{Estimator, Mat3, RansacOptions, Real, ransac_fit};

/// Result of robust essential matrix estimation.
#[derive(Debug, Clone)]
pub struct EssentialEstimate {
    /// The estimated essential matrix.
    pub essential: Mat3,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
    /// RMS Sampson distance over inliers.
    pub inlier_rms: Real,
}

/// Result of robust homography estimation.
#[derive(Debug, Clone)]
pub struct HomographyEstimate {
    /// The estimated homography matrix.
    pub homography: Mat3,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
    /// RMS symmetric transfer error over inliers.
    pub inlier_rms: Real,
}

/// Result of robust relative pose recovery.
#[derive(Debug, Clone)]
pub struct RobustRelativePose {
    /// Rotation from camera 1 to camera 2.
    pub r: Mat3,
    /// Unit translation direction from camera 1 to camera 2.
    pub t: vision_calibration_core::Vec3,
    /// The essential matrix used.
    pub essential: Mat3,
    /// Indices of inlier correspondences.
    pub inliers: Vec<usize>,
    /// RMS Sampson distance over inliers.
    pub inlier_rms: Real,
}

// --- Essential matrix RANSAC estimator ---

struct EssentialEstimator;

impl Estimator for EssentialEstimator {
    type Datum = Correspondence2D;
    type Model = Mat3;

    const MIN_SAMPLES: usize = 5;

    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
        let mut pts1 = Vec::with_capacity(sample_indices.len());
        let mut pts2 = Vec::with_capacity(sample_indices.len());
        for &idx in sample_indices {
            pts1.push(data[idx].pt1);
            pts2.push(data[idx].pt2);
        }
        let candidates = vision_geometry::epipolar::essential_5point(&pts1, &pts2).ok()?;

        // Pick the candidate with lowest total Sampson distance over the sample.
        candidates.into_iter().min_by(|a, b| {
            let sa: Real = sample_indices
                .iter()
                .map(|&i| residuals::sampson_distance(a, &data[i]))
                .sum();
            let sb: Real = sample_indices
                .iter()
                .map(|&i| residuals::sampson_distance(b, &data[i]))
                .sum();
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        residuals::sampson_distance(model, datum)
    }

    fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize]) -> bool {
        // Check if any 3 points are nearly collinear in either image.
        if sample_indices.len() < 3 {
            return false;
        }
        let p0 = data[sample_indices[0]].pt1;
        let p1 = data[sample_indices[1]].pt1;
        let p2 = data[sample_indices[2]].pt1;
        let area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        area.abs() < 1e-9
    }
}

// --- Homography RANSAC estimator ---

struct HomographyEstimator;

impl Estimator for HomographyEstimator {
    type Datum = Correspondence2D;
    type Model = Mat3;

    const MIN_SAMPLES: usize = 4;

    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
        let mut src = Vec::with_capacity(sample_indices.len());
        let mut dst = Vec::with_capacity(sample_indices.len());
        for &idx in sample_indices {
            src.push(data[idx].pt1);
            dst.push(data[idx].pt2);
        }
        vision_geometry::homography::dlt_homography(&src, &dst).ok()
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        residuals::symmetric_transfer_error(model, datum)
    }

    fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize]) -> bool {
        if sample_indices.len() < 3 {
            return false;
        }
        let p0 = data[sample_indices[0]].pt1;
        let p1 = data[sample_indices[1]].pt1;
        let p2 = data[sample_indices[2]].pt1;
        let area = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        area.abs() < 1e-9
    }
}

/// Estimate an essential matrix from correspondences using RANSAC.
///
/// Uses the 5-point algorithm as the minimal solver and Sampson distance
/// as the residual metric. Input points must be in **normalized camera
/// coordinates**.
pub fn estimate_essential(
    corrs: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<EssentialEstimate> {
    if corrs.len() < 5 {
        anyhow::bail!("need at least 5 correspondences, got {}", corrs.len());
    }

    let result = ransac_fit::<EssentialEstimator>(corrs, opts);
    if !result.success {
        anyhow::bail!("RANSAC failed to find essential matrix consensus");
    }

    Ok(EssentialEstimate {
        essential: result.model.expect("success guarantees model"),
        inliers: result.inliers,
        inlier_rms: result.inlier_rms,
    })
}

/// Estimate a homography from correspondences using RANSAC.
///
/// Uses the 4-point DLT as the minimal solver and symmetric transfer
/// error as the residual metric.
pub fn estimate_homography(
    corrs: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<HomographyEstimate> {
    if corrs.len() < 4 {
        anyhow::bail!("need at least 4 correspondences, got {}", corrs.len());
    }

    let result = ransac_fit::<HomographyEstimator>(corrs, opts);
    if !result.success {
        anyhow::bail!("RANSAC failed to find homography consensus");
    }

    Ok(HomographyEstimate {
        homography: result.model.expect("success guarantees model"),
        inliers: result.inliers,
        inlier_rms: result.inlier_rms,
    })
}

/// Recover relative pose robustly from correspondences.
///
/// Combines essential matrix RANSAC with decomposition and cheirality
/// selection. Input points must be in **normalized camera coordinates**.
pub fn recover_relative_pose_robust(
    corrs: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<RobustRelativePose> {
    let est = estimate_essential(corrs, opts)?;
    let (pts1, pts2) = Correspondence2D::split(corrs);

    let (r, t) = crate::cheirality::recover_pose_from_essential(&est.essential, &pts1, &pts2)?;

    Ok(RobustRelativePose {
        r,
        t,
        essential: est.essential,
        inliers: est.inliers,
        inlier_rms: est.inlier_rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt2, Pt3, Vec3};

    fn make_stereo_corrs_with_outliers() -> (Vec<Correspondence2D>, Mat3, Vec3) {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);

        let world = [
            Pt3::new(0.5, 0.3, 3.0),
            Pt3::new(-0.4, 0.2, 4.0),
            Pt3::new(0.6, -0.3, 5.0),
            Pt3::new(-0.3, -0.4, 3.5),
            Pt3::new(0.1, 0.6, 4.5),
            Pt3::new(0.4, -0.5, 6.0),
            Pt3::new(-0.2, 0.3, 3.0),
            Pt3::new(0.3, 0.5, 5.5),
            Pt3::new(-0.5, -0.1, 4.0),
            Pt3::new(0.2, -0.4, 3.2),
            Pt3::new(0.0, 0.0, 2.5),
            Pt3::new(-0.3, 0.5, 4.8),
            Pt3::new(0.4, 0.1, 3.8),
            Pt3::new(-0.1, -0.3, 5.2),
            Pt3::new(0.5, -0.2, 4.3),
        ];

        let mut corrs: Vec<_> = world
            .iter()
            .map(|pw| {
                let pc1 = pw.coords;
                let pc2 = r * pw.coords + t;
                Correspondence2D::new(
                    Pt2::new(pc1.x / pc1.z, pc1.y / pc1.z),
                    Pt2::new(pc2.x / pc2.z, pc2.y / pc2.z),
                )
            })
            .collect();

        // Add outliers (20%).
        corrs.push(Correspondence2D::new(
            Pt2::new(0.1, 0.2),
            Pt2::new(5.0, -3.0),
        ));
        corrs.push(Correspondence2D::new(
            Pt2::new(-0.3, 0.1),
            Pt2::new(-2.0, 4.0),
        ));
        corrs.push(Correspondence2D::new(
            Pt2::new(0.05, -0.15),
            Pt2::new(1.5, 1.5),
        ));

        (corrs, r, t)
    }

    #[test]
    fn essential_ransac_with_outliers() {
        let (corrs, r_gt, t_gt) = make_stereo_corrs_with_outliers();

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 42,
            refit_on_inliers: true,
        };

        let est = estimate_essential(&corrs, &opts).unwrap();
        assert!(
            est.inliers.len() >= 12,
            "expected ≥12 inliers, got {}",
            est.inliers.len()
        );

        // Verify decomposition recovers pose.
        let (pts1, pts2) = Correspondence2D::split(&corrs);
        let (r_est, t_est) =
            crate::cheirality::recover_pose_from_essential(&est.essential, &pts1, &pts2).unwrap();

        let r_diff = r_est.transpose() * r_gt;
        let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang_deg = cos_theta.acos().to_degrees();
        assert!(ang_deg < 2.0, "rotation error too large: {} deg", ang_deg);

        let cos_t = t_est.normalize().dot(&t_gt.normalize()).abs();
        assert!(
            (cos_t - 1.0).abs() < 0.01,
            "translation direction error: cos = {}",
            cos_t
        );
    }

    #[test]
    fn homography_ransac_with_outliers() {
        let rot = Rotation3::from_euler_angles(0.05, -0.03, 0.02);
        let r = *rot.matrix();
        let t = Vec3::new(0.3, 0.01, 0.005);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let d = 3.0;
        let h = crate::homography::homography_from_pose_and_plane(&r, &t, &n, d);

        let plane_pts = [
            Pt3::new(0.5, 0.3, d),
            Pt3::new(-0.4, 0.2, d),
            Pt3::new(0.6, -0.3, d),
            Pt3::new(-0.3, -0.4, d),
            Pt3::new(0.1, 0.6, d),
            Pt3::new(0.4, -0.5, d),
            Pt3::new(-0.2, 0.3, d),
            Pt3::new(0.3, 0.5, d),
            Pt3::new(-0.5, -0.1, d),
            Pt3::new(0.2, -0.4, d),
        ];

        let mut corrs: Vec<_> = plane_pts
            .iter()
            .map(|pw| {
                let pt1 = Pt2::new(pw.x / pw.z, pw.y / pw.z);
                let pt2 = crate::homography::homography_transfer(&h, &pt1);
                Correspondence2D::new(pt1, pt2)
            })
            .collect();

        // Add outliers.
        corrs.push(Correspondence2D::new(
            Pt2::new(0.1, 0.2),
            Pt2::new(5.0, -3.0),
        ));
        corrs.push(Correspondence2D::new(
            Pt2::new(-0.3, 0.1),
            Pt2::new(-2.0, 4.0),
        ));

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 0.001,
            min_inliers: 6,
            confidence: 0.99,
            seed: 42,
            refit_on_inliers: true,
        };

        let est = estimate_homography(&corrs, &opts).unwrap();
        assert!(
            est.inliers.len() >= 8,
            "expected ≥8 inliers, got {}",
            est.inliers.len()
        );
        assert!(
            est.inlier_rms < 1e-6,
            "inlier RMS too large: {}",
            est.inlier_rms
        );
    }

    #[test]
    fn recover_pose_robust_with_outliers() {
        let (corrs, r_gt, t_gt) = make_stereo_corrs_with_outliers();

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 42,
            refit_on_inliers: true,
        };

        let result = recover_relative_pose_robust(&corrs, &opts).unwrap();

        let r_diff = result.r.transpose() * r_gt;
        let cos_theta = ((r_diff.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let ang_deg = cos_theta.acos().to_degrees();
        assert!(ang_deg < 2.0, "rotation error: {} deg", ang_deg);

        let cos_t = result.t.normalize().dot(&t_gt.normalize()).abs();
        assert!(
            (cos_t - 1.0).abs() < 0.01,
            "translation direction error: cos = {}",
            cos_t
        );

        assert!(result.inliers.len() >= 12);
    }

    #[test]
    fn essential_ransac_deterministic_with_same_seed() {
        let (corrs, _, _) = make_stereo_corrs_with_outliers();

        let opts = RansacOptions {
            max_iters: 200,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 123,
            refit_on_inliers: true,
        };

        let r1 = estimate_essential(&corrs, &opts).unwrap();
        let r2 = estimate_essential(&corrs, &opts).unwrap();

        assert_eq!(r1.inliers, r2.inliers);
        assert!((r1.inlier_rms - r2.inlier_rms).abs() < 1e-15);
    }
}
