//! Robust estimation using RANSAC for epipolar and homography models.
//!
//! Provides RANSAC-based estimators that work with [`Correspondence2D`]:
//!
//! - [`estimate_essential`]: 5-point algorithm on bearings + angular epipolar error
//! - [`estimate_homography`]: 4-point DLT + symmetric transfer error
//! - [`recover_relative_pose_robust`]: full pipeline from matched pixels to pose

use crate::bearing::bearings;
use crate::pose_recovery::iso_from_rt;
use crate::residuals;
use crate::types::Correspondence2D;
use crate::{MvgError, Result};
use vision_calibration_core::{
    CalibratedRig, Estimator, Iso3, Mat3, RansacOptions, Real, Vec3, ransac_fit,
};

/// Result of robust essential matrix estimation.
#[derive(Debug, Clone)]
pub struct EssentialEstimate {
    /// The estimated essential matrix: `b2ᵀ E b1 = 0` on the bearings of
    /// matched pixels.
    pub essential: Mat3,
    /// Indices of inlier matches.
    pub inliers: Vec<usize>,
    /// RMS angular epipolar error over inliers, in radians.
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
    /// Pose of the first camera's frame in the second's (`second_se3_first`),
    /// with a unit-norm translation: the scale is not observable.
    pub second_se3_first: Iso3,
    /// The essential matrix used: `b2ᵀ E b1 = 0` on the bearings.
    pub essential: Mat3,
    /// Indices of inlier matches.
    pub inliers: Vec<usize>,
    /// RMS angular epipolar error over inliers, in radians.
    pub inlier_rms: Real,
}

// --- Essential matrix RANSAC estimator ---

/// The unit bearings of one match, in the two cameras' frames.
#[derive(Debug, Clone, Copy)]
struct BearingPair {
    b1: Vec3,
    b2: Vec3,
}

struct EssentialEstimator;

impl Estimator for EssentialEstimator {
    type Datum = BearingPair;
    type Model = Mat3;

    const MIN_SAMPLES: usize = 5;

    fn fit(data: &[Self::Datum], sample_indices: &[usize]) -> Option<Self::Model> {
        let n = sample_indices.len();

        // Shared helper: pick the candidate E that minimises the total angular
        // epipolar error over *all* sample_indices (not just the 5 used to
        // construct the candidates).  Used for both the minimal and the 6-7
        // point paths.
        let best_of = |candidates: Vec<Mat3>| -> Option<Mat3> {
            let total = |e: &Mat3| -> Real {
                sample_indices
                    .iter()
                    .map(|&i| residuals::angular_epipolar_error(e, &data[i].b1, &data[i].b2))
                    .sum()
            };
            candidates.into_iter().min_by(|a, b| {
                total(a)
                    .partial_cmp(&total(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        };

        if n >= 8 {
            // Overdetermined case (refit on inliers): the linear solver has a
            // 1-dimensional null space and projects onto the essential manifold.
            let b1: Vec<_> = sample_indices.iter().map(|&i| data[i].b1).collect();
            let b2: Vec<_> = sample_indices.iter().map(|&i| data[i].b2).collect();
            vision_geometry::epipolar::essential_linear(&b1, &b2).ok()
        } else if n >= 5 {
            // Minimal or near-minimal case (5, 6, or 7 points): use the 5-point
            // solver on the first five matches, then score all candidates
            // against the full sample_indices set to pick the best one.
            // RANSAC guarantees n >= MIN_SAMPLES == 5, so n < 5 is unreachable.
            let b1: Vec<_> = sample_indices[..5].iter().map(|&i| data[i].b1).collect();
            let b2: Vec<_> = sample_indices[..5].iter().map(|&i| data[i].b2).collect();
            let candidates = vision_geometry::epipolar::essential_5point(&b1, &b2).ok()?;
            best_of(candidates)
        } else {
            None
        }
    }

    fn residual(model: &Self::Model, datum: &Self::Datum) -> f64 {
        residuals::angular_epipolar_error(model, &datum.b1, &datum.b2)
    }

    fn is_degenerate(data: &[Self::Datum], sample_indices: &[usize]) -> bool {
        // Three coplanar first-camera rays (a vanishing triple product of their
        // bearings) make the sample degenerate.
        if sample_indices.len() < 3 {
            return false;
        }
        let b0 = data[sample_indices[0]].b1;
        let b1 = data[sample_indices[1]].b1;
        let b2 = data[sample_indices[2]].b1;
        b0.dot(&b1.cross(&b2)).abs() < 1e-9
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

/// The matches as bearing pairs through the rig cameras `cameras`.
fn bearing_pairs(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[Correspondence2D],
) -> Result<Vec<BearingPair>> {
    let b1 = bearings(rig, cameras[0], matches.iter().map(|c| c.pt1))?;
    let b2 = bearings(rig, cameras[1], matches.iter().map(|c| c.pt2))?;
    Ok(b1
        .into_iter()
        .zip(b2)
        .map(|(b1, b2)| BearingPair { b1, b2 })
        .collect())
}

/// Estimate an essential matrix from matched pixels using RANSAC.
///
/// `matches[i].pt1` is a pixel of rig camera `cameras[0]` in the first
/// capture and `matches[i].pt2` of `cameras[1]` in the second (`[0, 0]` for one
/// moving camera). Pixels are unprojected through each camera's own model into
/// bearings; the 5-point algorithm is the minimal solver and the first-order
/// angular epipolar error the residual.
///
/// **Threshold unit:** `opts.thresh` is compared with that angular error in
/// **radians**, not pixels. A pixel threshold `t` is about `t / f` radians for a
/// focal length `f` in pixels near the image centre (`1.0e-3` rad is about
/// 0.8 px at `f = 800`). A pixel-scale value such as `2.0` accepts every match.
///
/// # Errors
///
/// [`MvgError::Core`] if `opts.thresh` is not finite and positive,
/// [`MvgError::InsufficientData`] for fewer than 5 matches,
/// [`MvgError::InvalidInput`] for a camera index out of range or a pixel with no
/// finite viewing ray, and [`MvgError::NoConsensus`] if no model reaches
/// `opts.min_inliers`.
pub fn estimate_essential(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<EssentialEstimate> {
    opts.validate()?;
    if matches.len() < 5 {
        return Err(MvgError::InsufficientData {
            need: 5,
            got: matches.len(),
        });
    }
    let pairs = bearing_pairs(rig, cameras, matches)?;
    essential_from_pairs(&pairs, opts)
}

fn essential_from_pairs(pairs: &[BearingPair], opts: &RansacOptions) -> Result<EssentialEstimate> {
    let result = ransac_fit::<EssentialEstimator>(pairs, opts);
    if !result.success {
        return Err(MvgError::NoConsensus);
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
///
/// **Threshold unit:** `opts.thresh` is compared with the (non-squared)
/// symmetric transfer error `sqrt(|H x1 - x2|^2 + |H^-1 x2 - x1|^2)`, in the
/// unit of the input points (pixels for pixel correspondences). The two
/// transfer directions add in quadrature, so a point with an error `e` in each
/// direction scores `sqrt(2) e`.
///
/// # Errors
///
/// [`MvgError::Core`] if `opts.thresh` is not finite and positive,
/// [`MvgError::InsufficientData`] for fewer than 4 correspondences and
/// [`MvgError::NoConsensus`] if no model reaches `opts.min_inliers`.
pub fn estimate_homography(
    corrs: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<HomographyEstimate> {
    opts.validate()?;
    if corrs.len() < 4 {
        return Err(MvgError::InsufficientData {
            need: 4,
            got: corrs.len(),
        });
    }

    let result = ransac_fit::<HomographyEstimator>(corrs, opts);
    if !result.success {
        return Err(MvgError::NoConsensus);
    }

    Ok(HomographyEstimate {
        homography: result.model.expect("success guarantees model"),
        inliers: result.inliers,
        inlier_rms: result.inlier_rms,
    })
}

/// Recover relative pose robustly from matched pixels.
///
/// Combines essential matrix RANSAC with decomposition and cheirality
/// selection, with the arguments of [`estimate_essential`] (the rig, the
/// camera of each capture and the matches). The returned `second_se3_first` has
/// a unit-norm translation.
///
/// **Threshold unit:** as for [`estimate_essential`], `opts.thresh` is the
/// angular epipolar error in radians (about `t / f` for a pixel threshold `t`
/// and focal length `f` in pixels).
///
/// # Errors
///
/// The errors of [`estimate_essential`], plus the cheirality errors of the
/// pose decomposition.
pub fn recover_relative_pose_robust(
    rig: &CalibratedRig,
    cameras: [usize; 2],
    matches: &[Correspondence2D],
    opts: &RansacOptions,
) -> Result<RobustRelativePose> {
    opts.validate()?;
    if matches.len() < 5 {
        return Err(MvgError::InsufficientData {
            need: 5,
            got: matches.len(),
        });
    }
    let pairs = bearing_pairs(rig, cameras, matches)?;
    let est = essential_from_pairs(&pairs, opts)?;

    // Use only inliers for cheirality disambiguation, so outliers don't
    // corrupt the positive-depth vote.
    let b1: Vec<Vec3> = est.inliers.iter().map(|&i| pairs[i].b1).collect();
    let b2: Vec<Vec3> = est.inliers.iter().map(|&i| pairs[i].b2).collect();

    let (r, t) = crate::cheirality::recover_pose_from_essential(&est.essential, &b1, &b2)?;

    Ok(RobustRelativePose {
        second_se3_first: iso_from_rt(&r, &t),
        essential: est.essential,
        inliers: est.inliers,
        inlier_rms: est.inlier_rms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Lcg, iso, pinhole_rig, pixel_of};
    use nalgebra::Rotation3;
    use vision_calibration_core::{Pt2, Pt3};

    /// The ground-truth pose of the second view.
    fn gt_pose() -> Iso3 {
        iso(0.05, -0.03, 0.02, Vec3::new(0.3, 0.01, 0.005))
    }

    fn rig() -> CalibratedRig {
        // Distortion makes the pixel path matter.
        pinhole_rig(-0.15)
    }

    /// 15 noise-free matches of an 800 px camera plus 3 gross outliers.
    fn stereo_matches_with_outliers() -> Vec<Correspondence2D> {
        let rig = rig();
        let gt = gt_pose();
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
        let mut matches: Vec<_> = world
            .iter()
            .map(|p| {
                Correspondence2D::new(
                    pixel_of(&rig, 0, &p.coords),
                    pixel_of(&rig, 0, &(gt * p).coords),
                )
            })
            .collect();
        // Add outliers.
        matches.push(Correspondence2D::new(
            Pt2::new(700.0, 500.0),
            Pt2::new(100.0, 90.0),
        ));
        matches.push(Correspondence2D::new(
            Pt2::new(400.0, 520.0),
            Pt2::new(900.0, 100.0),
        ));
        matches.push(Correspondence2D::new(
            Pt2::new(660.0, 400.0),
            Pt2::new(300.0, 700.0),
        ));
        matches
    }

    fn rotation_error_deg(a: &Iso3, b: &Iso3) -> f64 {
        a.rotation.angle_to(&b.rotation).to_degrees()
    }

    #[test]
    fn essential_ransac_with_outliers() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 42,
            refit_on_inliers: true,
        };

        let est = estimate_essential(&rig, [0, 0], &matches, &opts).unwrap();
        assert!(
            est.inliers.len() >= 12,
            "expected ≥12 inliers, got {}",
            est.inliers.len()
        );
        assert!(est.inliers.iter().all(|&i| i < 15), "an outlier survived");
        assert!(est.inlier_rms < 1e-3);

        // Verify decomposition recovers pose (using inliers only).
        let pairs = bearing_pairs(&rig, [0, 0], &matches).unwrap();
        let b1: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b1).collect();
        let b2: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b2).collect();
        let (r_est, t_est) =
            crate::cheirality::recover_pose_from_essential(&est.essential, &b1, &b2).unwrap();
        let est_pose = iso_from_rt(&r_est, &t_est);
        assert!(rotation_error_deg(&est_pose, &gt_pose()) < 0.1);
        let cos_t = t_est
            .normalize()
            .dot(&gt_pose().translation.vector.normalize());
        assert!(
            (cos_t - 1.0).abs() < 1e-3,
            "translation direction: cos = {cos_t}"
        );
    }

    /// The 15 noise-free inliers of the fixture plus 6 gross outliers (29 %);
    /// the inliers keep indices `0..15`.
    fn thirty_percent_outliers() -> (Vec<Correspondence2D>, usize) {
        const N_IN: usize = 15;
        let mut matches = stereo_matches_with_outliers();
        matches.truncate(N_IN);
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        for _ in 0..6 {
            let mut px = || Pt2::new(640.0 + 400.0 * rng.next(), 480.0 + 300.0 * rng.next());
            let (a, b) = (px(), px());
            matches.push(Correspondence2D::new(a, b));
        }
        (matches, N_IN)
    }

    #[test]
    fn essential_ransac_rejects_outliers_at_angular_threshold() {
        let rig = rig();
        let (matches, n_in) = thirty_percent_outliers();
        let opts = RansacOptions {
            max_iters: 2000,
            min_inliers: 8,
            seed: 7,
            ..RansacOptions::new(1.0e-3)
        };
        let est = estimate_essential(&rig, [0, 0], &matches, &opts).unwrap();
        assert_eq!(est.inliers.len(), n_in, "inlier count");
        assert!(est.inliers.iter().all(|&i| i < n_in), "an outlier survived");

        // A pixel-scale threshold read as radians accepts everything.
        let loose = RansacOptions {
            max_iters: 50,
            min_inliers: 8,
            seed: 7,
            ..RansacOptions::new(2.0)
        };
        let est = estimate_essential(&rig, [0, 0], &matches, &loose).unwrap();
        assert_eq!(est.inliers.len(), matches.len());
    }

    #[test]
    fn ransac_threshold_must_be_finite_and_positive() {
        let rig = rig();
        let (matches, _) = thirty_percent_outliers();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let opts = RansacOptions::new(bad);
            assert!(
                matches!(
                    estimate_essential(&rig, [0, 0], &matches, &opts),
                    Err(MvgError::Core(_))
                ),
                "essential accepted thresh {bad}"
            );
            assert!(
                matches!(estimate_homography(&matches, &opts), Err(MvgError::Core(_))),
                "homography accepted thresh {bad}"
            );
            assert!(
                matches!(
                    recover_relative_pose_robust(&rig, [0, 0], &matches, &opts),
                    Err(MvgError::Core(_))
                ),
                "pose accepted thresh {bad}"
            );
        }
    }

    #[test]
    fn camera_index_out_of_range_is_invalid_input() {
        let (matches, _) = thirty_percent_outliers();
        let opts = RansacOptions::new(1e-3);
        assert!(matches!(
            estimate_essential(&rig(), [0, 1], &matches, &opts),
            Err(MvgError::InvalidInput { .. })
        ));
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
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts = RansacOptions {
            max_iters: 500,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 42,
            refit_on_inliers: true,
        };

        let result = recover_relative_pose_robust(&rig, [0, 0], &matches, &opts).unwrap();

        assert!(rotation_error_deg(&result.second_se3_first, &gt_pose()) < 0.1);
        let t = result.second_se3_first.translation.vector;
        assert!((t.norm() - 1.0).abs() < 1e-12);
        let cos_t = t.dot(&gt_pose().translation.vector.normalize());
        assert!(
            (cos_t - 1.0).abs() < 1e-3,
            "translation direction: cos = {cos_t}"
        );
        assert!(result.inliers.len() >= 12);
    }

    #[test]
    fn essential_ransac_deterministic_with_same_seed() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts = RansacOptions {
            max_iters: 200,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 123,
            refit_on_inliers: true,
        };

        let r1 = estimate_essential(&rig, [0, 0], &matches, &opts).unwrap();
        let r2 = estimate_essential(&rig, [0, 0], &matches, &opts).unwrap();

        assert_eq!(r1.inliers, r2.inliers);
        assert!((r1.inlier_rms - r2.inlier_rms).abs() < 1e-15);
    }

    /// Regression test for the RANSAC refit no-op bug.
    ///
    /// When `refit_on_inliers: true` and the consensus set has >5 points, the
    /// old code called `essential_5point` with all inliers → `Err` → silent
    /// no-op (model tied to a single 5-point sample).  The fix routes the
    /// overdetermined case to `essential_linear`, which accepts any ≥5 points.
    #[test]
    fn essential_ransac_refit_on_inliers_uses_linear_solver() {
        let rig = rig();
        let matches = stereo_matches_with_outliers();

        let opts_refit = RansacOptions {
            max_iters: 500,
            thresh: 0.001,
            min_inliers: 10,
            confidence: 0.99,
            seed: 77,
            refit_on_inliers: true,
        };

        let est = estimate_essential(&rig, [0, 0], &matches, &opts_refit)
            .expect("RANSAC with refit must succeed");
        assert!(
            est.inliers.len() >= 12,
            "expected ≥12 inliers after refit, got {}",
            est.inliers.len()
        );

        // The refit model must be geometrically valid: decompose and check pose.
        let pairs = bearing_pairs(&rig, [0, 0], &matches).unwrap();
        let b1: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b1).collect();
        let b2: Vec<_> = est.inliers.iter().map(|&i| pairs[i].b2).collect();
        let (r_est, t_est) =
            crate::cheirality::recover_pose_from_essential(&est.essential, &b1, &b2)
                .expect("pose decomposition must succeed");
        assert!(rotation_error_deg(&iso_from_rt(&r_est, &t_est), &gt_pose()) < 0.1);
        let cos_t = t_est
            .normalize()
            .dot(&gt_pose().translation.vector.normalize());
        assert!((cos_t - 1.0).abs() < 1e-3, "cos_t = {cos_t}");
    }

    /// The 6-/7-point dispatch path in `EssentialEstimator::fit` produces a
    /// valid essential matrix on bearings, including ones beyond 90° off the
    /// axis.
    #[test]
    fn essential_fit_7pts_uses_5point_scored_on_all() {
        let rot = Rotation3::from_euler_angles(0.08, -0.04, 0.15);
        let t = Vec3::new(0.25, 0.05, 0.01);

        let world = [
            Pt3::new(0.4, 0.2, 3.0),
            Pt3::new(-0.3, 0.1, 4.0),
            Pt3::new(0.5, -0.2, 3.5),
            Pt3::new(-0.2, -0.3, 2.8),
            Pt3::new(0.1, 0.4, 4.5),
            Pt3::new(0.3, -0.4, 5.0),
            Pt3::new(-0.4, 0.3, -3.2),
        ];

        let pairs: Vec<BearingPair> = world
            .iter()
            .map(|pw| BearingPair {
                b1: pw.coords.normalize(),
                b2: (rot * pw.coords + t).normalize(),
            })
            .collect();

        let indices: Vec<usize> = (0..7).collect();
        let e =
            EssentialEstimator::fit(&pairs, &indices).expect("fit with 7 points must return Some");

        // 1. Angular epipolar error near zero for all 7 points.
        assert!(e.norm() > 0.0, "fit returned zero matrix");
        for p in &pairs {
            let res = residuals::angular_epipolar_error(&e, &p.b1, &p.b2);
            assert!(res < 1e-5, "epipolar residual too large: {:.3e}", res);
        }

        // 2. Essential manifold check: singular values ≈ (σ, σ, 0).
        let sv = e.svd(false, false).singular_values;
        let s_avg = (sv[0] + sv[1]) * 0.5;
        assert!(
            (sv[0] / s_avg - 1.0).abs() < 1e-4,
            "sv[0] not close to sv[1]: {:?}",
            sv
        );
        assert!(sv[2] / s_avg < 1e-4, "sv[2] not near zero: {:?}", sv);
    }
}
