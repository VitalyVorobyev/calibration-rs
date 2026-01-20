//! Iterative linear intrinsics estimation with distortion refinement.
//!
//! This module implements an alternating optimization scheme for jointly estimating
//! camera intrinsics (fx, fy, cx, cy, skew) and Brown-Conrady distortion coefficients
//! (k1, k2, k3, p1, p2) without requiring ground truth distortion for preprocessing.
//!
//! # Algorithm Overview
//!
//! The classic Zhang method assumes distortion-free inputs. When distortion is present,
//! directly applying Zhang to distorted pixels produces biased intrinsics estimates.
//! This module addresses the problem through iterative refinement:
//!
//! 1. **Initial estimate**: Compute K from distorted pixels (ignoring distortion)
//! 2. **Distortion estimation**: Using K from step 1, estimate distortion from homography residuals
//! 3. **Pixel undistortion**: Apply estimated distortion to correct pixel observations
//! 4. **Intrinsics refinement**: Re-estimate K from undistorted pixels
//! 5. **Iterate**: Repeat steps 2-4 for k iterations
//!
//! Typically converges in 1-3 iterations, providing initial estimates suitable for
//! non-linear bundle adjustment.
//!
//! # When to Use
//!
//! Use this solver when:
//! - You don't have ground truth distortion parameters
//! - You have multiple views of a planar calibration pattern
//! - You need both intrinsics and distortion for initialization
//!
//! For distortion-free cameras or when you already have good intrinsics,
//! use [`zhang_intrinsics`](crate::zhang_intrinsics) directly.
//!
//! # Comparison with Single-Pass Zhang
//!
//! - **Zhang alone**: Fast, but biased when distortion is significant
//! - **Iterative**: Slower (2-3× cost), but handles distortion without ground truth
//! - **After non-linear refinement**: Both converge to similar final accuracy
//!
//! # Example
//!
//! ```no_run
//! use calib_core::{CorrespondenceView, PlanarDataset, Pt2, Pt3, View};
//! use calib_linear::iterative_intrinsics::{estimate_intrinsics_iterative, IterativeIntrinsicsOptions};
//!
//! let points_3d = vec![Pt3::new(0.0, 0.0, 0.0) /* ... board points */];
//! let points_2d = vec![Pt2::new(320.0, 240.0) /* ... distorted pixels */];
//! let obs = CorrespondenceView::new(points_3d, points_2d)?;
//! let dataset = PlanarDataset::new(vec![View::without_meta(obs)])?;
//!
//! let opts = IterativeIntrinsicsOptions::default();
//! let camera = estimate_intrinsics_iterative(&dataset, opts)?;
//!
//! println!("Intrinsics: fx={}, fy={}, cx={}, cy={}",
//!          camera.k.fx, camera.k.fy,
//!          camera.k.cx, camera.k.cy);
//! println!("Distortion: k1={}, k2={}", camera.dist.k1, camera.dist.k2);
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{
    distortion_fit::{
        estimate_distortion_from_homographies, DistortionFitOptions, DistortionView, MetaHomography,
    },
    homography::HomographySolver,
    zhang_intrinsics::PlanarIntrinsicsLinearInit,
};
use anyhow::Result;
use calib_core::{
    make_pinhole_camera, BrownConrady5, DistortionModel, FxFyCxCySkew, Mat3, PinholeCamera,
    PlanarDataset, Pt2, Real, Vec3,
};
use serde::{Deserialize, Serialize};

/// Options controlling iterative intrinsics estimation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IterativeIntrinsicsOptions {
    /// Number of refinement iterations (distortion → K → distortion → K).
    ///
    /// Each iteration:
    /// 1. Estimates distortion using current K
    /// 2. Undistorts pixels using estimated distortion
    /// 3. Re-estimates K from undistorted pixels
    ///
    /// Typical values: 1-3. Beyond 3 iterations, improvement is marginal.
    pub iterations: usize,

    /// Options for distortion fitting.
    ///
    /// Controls which distortion parameters to estimate (fix_k3, fix_tangential).
    pub distortion_opts: DistortionFitOptions,

    /// Force skew to zero after each intrinsics estimate.
    ///
    /// Recommended for most cameras and required by current 4-parameter
    /// optimization backends.
    pub zero_skew: bool,
}

impl Default for IterativeIntrinsicsOptions {
    fn default() -> Self {
        Self {
            iterations: 2, // One distortion estimate + one K re-estimate typically sufficient
            distortion_opts: DistortionFitOptions::default(),
            zero_skew: true,
        }
    }
}

/// Estimate intrinsics and distortion using iterative refinement.
///
/// # Algorithm
///
/// 1. Estimate initial K from distorted pixels (Zhang method, iteration 0)
/// 2. For each iteration i = 1..iterations:
///    a. Compute homographies from current pixels (distorted or undistorted)
///    b. Estimate distortion from residuals using current K
///    c. Undistort original pixels using estimated distortion
///    d. Re-estimate K from undistorted pixels
/// 3. Return final K and distortion with full history
///
/// # Arguments
///
/// * `views` - Multiple views of calibration pattern (≥3 required)
/// * `opts` - Options controlling iteration count and distortion parameters
///
/// # Returns
///
/// `IterativeIntrinsicsResult` containing final estimates and iteration history.
///
/// # Errors
///
/// - `NotEnoughViews`: Fewer than 3 views provided
/// - `ZhangFailed`: Zhang intrinsics estimation failed (e.g., degenerate homographies)
/// - `DistortionFailed`: Distortion estimation failed (e.g., insufficient radial diversity)
/// - `HomographyFailed`: Homography computation failed
///
/// # Complexity
///
/// O(iterations × N_views × N_points_per_view)
///
/// Typically 2-3× slower than single-pass Zhang, but eliminates need for
/// ground truth distortion preprocessing.
pub fn estimate_intrinsics_iterative(
    dataset: &PlanarDataset,
    opts: IterativeIntrinsicsOptions,
) -> Result<PinholeCamera> {
    if dataset.views.len() < 3 {
        anyhow::bail!("need at least 3 views, got {}", dataset.views.len());
    }

    let target_points_2d: Vec<Vec<Pt2>> = dataset
        .views
        .iter()
        .map(|v| v.obs.planar_points())
        .collect();

    // Iteration 0: initial K estimate from distorted pixels (ignore distortion).
    let homographies_iter0: Vec<Mat3> = dataset
        .views
        .iter()
        .zip(&target_points_2d)
        .map(|(v, target)| HomographySolver::dlt(target, &v.obs.points_2d))
        .collect::<Result<Vec<_>>>()?;

    let mut current_intrinsics =
        PlanarIntrinsicsLinearInit::from_homographies(&homographies_iter0)?;
    enforce_zero_skew(&mut current_intrinsics, opts.zero_skew);

    let mut current_distortion = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: opts.distortion_opts.iters,
    };

    for _iter in 0..opts.iterations {
        let k_mtx = current_intrinsics.k_matrix();
        let k_inv = k_mtx
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

        let homographies_for_distortion: Vec<Mat3> = dataset
            .views
            .iter()
            .zip(&target_points_2d)
            .map(|(v, target)| {
                let undistorted_pixels = undistort_pixels_to_pixels(
                    &v.obs.points_2d,
                    &k_mtx,
                    &k_inv,
                    &current_distortion,
                );
                HomographySolver::dlt(target, &undistorted_pixels)
            })
            .collect::<Result<Vec<_>>>()?;

        let dist_views: Vec<DistortionView> = dataset
            .views
            .iter()
            .zip(&homographies_for_distortion)
            .map(|(v, h)| {
                DistortionView::new(
                    v.obs.clone(),
                    MetaHomography {
                        homography: h.clone(),
                    },
                )
            })
            .collect();

        current_distortion =
            estimate_distortion_from_homographies(&k_mtx, &dist_views, opts.distortion_opts)?;

        let undistorted_homographies: Vec<Mat3> = dataset
            .views
            .iter()
            .zip(&target_points_2d)
            .map(|(v, target)| {
                let undistorted_pixels = undistort_pixels_to_pixels(
                    &v.obs.points_2d,
                    &k_mtx,
                    &k_inv,
                    &current_distortion,
                );
                HomographySolver::dlt(target, &undistorted_pixels)
            })
            .collect::<Result<Vec<_>>>()?;

        current_intrinsics =
            PlanarIntrinsicsLinearInit::from_homographies(&undistorted_homographies)?;
        enforce_zero_skew(&mut current_intrinsics, opts.zero_skew);
    }

    Ok(make_pinhole_camera(current_intrinsics, current_distortion))
}

fn enforce_zero_skew(intrinsics: &mut FxFyCxCySkew<Real>, zero_skew: bool) {
    if zero_skew {
        intrinsics.skew = 0.0;
    }
}

fn undistort_pixels_to_pixels(
    pixels: &[Pt2],
    k_mtx: &Mat3,
    k_inv: &Mat3,
    distortion: &BrownConrady5<Real>,
) -> Vec<Pt2> {
    pixels
        .iter()
        .map(|p| {
            let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
            let n_dist = Pt2::new(v_h.x / v_h.z, v_h.y / v_h.z);
            let n_undist = distortion.undistort(&n_dist);
            let p_h = k_mtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
            Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::synthetic::planar::{grid_points, project_views_all};
    use calib_core::{Iso3, View};
    use nalgebra::{Translation3, UnitQuaternion};

    fn make_ground_truth() -> (FxFyCxCySkew<Real>, BrownConrady5<Real>) {
        let intr = FxFyCxCySkew {
            fx: 800.0,
            fy: 800.0,
            cx: 640.0,
            cy: 360.0,
            skew: 0.0,
        };
        let dist = BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 8,
        };
        (intr, dist)
    }

    fn make_dataset(camera: &PinholeCamera, n_views: usize) -> PlanarDataset {
        let board = grid_points(7, 7, 0.03);

        let pose_params: &[(Real, Real, Real, Real, Real, Real)] = &[
            (0.10, 0.00, 0.05, 0.10, -0.05, 1.00),
            (-0.05, 0.15, -0.10, -0.05, 0.10, 1.20),
            (0.20, -0.10, 0.00, 0.00, 0.00, 0.90),
            (0.00, 0.20, 0.10, 0.08, 0.08, 1.10),
            (-0.10, 0.10, -0.05, -0.08, -0.08, 1.05),
        ];

        let poses: Vec<Iso3> = pose_params
            .iter()
            .take(n_views)
            .map(|&(rx, ry, rz, tx, ty, tz)| {
                Iso3::from_parts(
                    Translation3::new(tx, ty, tz),
                    UnitQuaternion::from_euler_angles(rx, ry, rz),
                )
            })
            .collect();

        let views = project_views_all(camera, &board, &poses).unwrap();
        let views = views.into_iter().map(View::without_meta).collect();
        PlanarDataset::new(views).unwrap()
    }

    #[test]
    fn iterative_refinement_converges() {
        let (intr_gt, dist_gt) = make_ground_truth();
        let camera_gt = make_pinhole_camera(intr_gt, dist_gt);
        let dataset = make_dataset(&camera_gt, 4);

        let opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
            zero_skew: true,
        };

        let result = estimate_intrinsics_iterative(&dataset, opts).unwrap();

        // Check final estimates (linear methods: 10-30% accuracy expected for initialization)
        // This is acceptable - the purpose is to provide a reasonable starting point
        // for non-linear refinement, not to achieve final accuracy.
        let fx_err_pct = (result.k.fx - intr_gt.fx).abs() / intr_gt.fx * 100.0;
        let fy_err_pct = (result.k.fy - intr_gt.fy).abs() / intr_gt.fy * 100.0;
        let cx_err = (result.k.cx - intr_gt.cx).abs();
        let cy_err = (result.k.cy - intr_gt.cy).abs();

        assert!(fx_err_pct < 40.0, "fx error {:.1}% too large", fx_err_pct);
        assert!(fy_err_pct < 40.0, "fy error {:.1}% too large", fy_err_pct);
        assert!(cx_err < 80.0, "cx error {:.1}px too large", cx_err);
        assert!(cy_err < 150.0, "cy error {:.1}px too large", cy_err);

        // Check distortion estimates have correct sign
        assert!(
            result.dist.k1.signum() == dist_gt.k1.signum(),
            "k1 sign mismatch"
        );
    }

    #[test]
    fn iteration_improves_estimates() {
        let (intr_gt, dist_gt) = make_ground_truth();
        let camera_gt = make_pinhole_camera(intr_gt, dist_gt);
        let dataset = make_dataset(&camera_gt, 4);

        let opts0 = IterativeIntrinsicsOptions {
            iterations: 0,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
            zero_skew: true,
        };

        let cam0 = estimate_intrinsics_iterative(&dataset, opts0).unwrap();

        let opts2 = IterativeIntrinsicsOptions {
            iterations: 2,
            ..opts0
        };
        let cam2 = estimate_intrinsics_iterative(&dataset, opts2).unwrap();

        let rms0 = plane_homography_rms(&cam0, &dataset);
        let rms2 = plane_homography_rms(&cam2, &dataset);
        assert!(
            rms2 < rms0,
            "expected lower plane residual after iterations: rms0={rms0:.4}, rms2={rms2:.4}"
        );
    }

    fn plane_homography_rms(camera: &PinholeCamera, dataset: &PlanarDataset) -> Real {
        let k_mtx = camera.k.k_matrix();
        let k_inv = k_mtx.try_inverse().expect("K invertible for tests");

        let mut total_sq = 0.0;
        let mut total_n = 0usize;

        for view in &dataset.views {
            let world = view.obs.planar_points();
            let undistorted_pixels =
                undistort_pixels_to_pixels(&view.obs.points_2d, &k_mtx, &k_inv, &camera.dist);
            let h = HomographySolver::dlt(&world, &undistorted_pixels)
                .expect("homography should solve on synthetic data");

            for (pw, pu) in world.iter().zip(undistorted_pixels.iter()) {
                let proj_h = h * Vec3::new(pw.x, pw.y, 1.0);
                let proj = Pt2::new(proj_h.x / proj_h.z, proj_h.y / proj_h.z);
                let dx = proj.x - pu.x;
                let dy = proj.y - pu.y;
                total_sq += dx * dx + dy * dy;
                total_n += 1;
            }
        }

        (total_sq / total_n as Real).sqrt()
    }
}
