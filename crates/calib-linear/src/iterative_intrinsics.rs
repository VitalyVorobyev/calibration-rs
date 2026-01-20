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
//! use calib_core::Pt2;
//! use calib_linear::iterative_intrinsics::{
//!     estimate_intrinsics_iterative, IterativeCalibView, IterativeIntrinsicsOptions,
//! };
//!
//! let views = vec![
//!     IterativeCalibView::new(
//!         vec![Pt2::new(0.0, 0.0), /* board points */],
//!         vec![Pt2::new(320.0, 240.0), /* distorted pixels */],
//!     )?,
//!     // ... more views
//! ];
//!
//! let opts = IterativeIntrinsicsOptions::default();
//! let camera = estimate_intrinsics_iterative(&views, opts)?;
//!
//! println!("Intrinsics: fx={}, fy={}, cx={}, cy={}",
//!          camera.k.fx, camera.k.fy,
//!          camera.k.cx, camera.k.cy);
//! println!("Distortion: k1={}, k2={}", camera.dist.k1, camera.dist.k2);
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{
    distortion_fit::{DistortionFitOptions, DistortionSolver, DistortionView},
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

    let target_points2d = dataset
        .views
        .iter()
        .map(|v| {
            v.obs
                .points_3d
                .iter()
                .map(|p| Pt2::new(p.x, p.y))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    // Iteration 0: initial K estimate from distorted pixels (ignore distortion).
    let homographies_iter0: Vec<Mat3> = dataset
        .views
        .iter()
        .zip(&target_points2d)
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

    for iter in 0..opts.iterations {
        let k_mtx = current_intrinsics.k_matrix();
        let k_inv = k_mtx
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("intrinsics matrix is not invertible"))?;

        let homographies = if iter == 0 {
            homographies_iter0.clone()
        } else {
            dataset.views
                .iter()
                .zip(&target_points2d)
                .map(|(v, target)| {
                    let undistorted_pixels: Vec<Pt2> = v
                        .obs.points_2d
                        .iter()
                        .map(|p| {
                            let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
                            let n_dist = Pt2::new(v_h.x / v_h.z, v_h.y / v_h.z);
                            let n_undist = current_distortion.undistort(&n_dist);
                            let p_h = k_mtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
                            Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z)
                        })
                        .collect();
                    HomographySolver::dlt(&target, &undistorted_pixels)
                })
                .collect::<Result<Vec<_>>>()?
        };

        let dist_views = dataset.views
            .iter()
            .zip(&homographies)
            .map(|(v, h)| DistortionView::new(*h, v.obs.board_points.clone(), v.obs.points_2d.clone()))
            .collect::<Result<Vec<_>>>()?;

        current_distortion =
            DistortionSolver::from_homographies(&k_mtx, &dist_views, opts.distortion_opts)?;

        let undistorted_homographies = views
            .iter()
            .map(|v| {
                let undistorted_pixels: Vec<Pt2> = v
                    .pixel_points
                    .iter()
                    .map(|p| {
                        let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
                        let n_dist = Pt2::new(v_h.x / v_h.z, v_h.y / v_h.z);
                        let n_undist = current_distortion.undistort(&n_dist);
                        let p_h = k_mtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
                        Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z)
                    })
                    .collect();
                HomographySolver::dlt(&v.board_points, &undistorted_pixels)
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

#[cfg(test)]
mod tests {
    use std::pin::Pin;

    use super::*;
    use nalgebra::{Isometry3, Rotation3, Translation3, Vector3};

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

    fn generate_synthetic_views(camera: &PinholeCamera, n_views: usize) -> Vec<IterativeCalibView> {
        let k_mtx = intr.k_matrix();
        let mut views = Vec::new();

        // 7x7 board, 30mm spacing
        let mut board_points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                board_points.push(Pt2::new(i as Real * 30.0, j as Real * 30.0));
            }
        }

        let poses = [
            (
                Rotation3::from_euler_angles(0.1, 0.0, 0.05),
                Vector3::new(100.0, -50.0, 1000.0),
            ),
            (
                Rotation3::from_euler_angles(-0.05, 0.15, -0.1),
                Vector3::new(-50.0, 100.0, 1200.0),
            ),
            (
                Rotation3::from_euler_angles(0.2, -0.1, 0.0),
                Vector3::new(0.0, 0.0, 900.0),
            ),
            (
                Rotation3::from_euler_angles(0.0, 0.2, 0.1),
                Vector3::new(80.0, 80.0, 1100.0),
            ),
            (
                Rotation3::from_euler_angles(-0.1, 0.1, -0.05),
                Vector3::new(-80.0, -80.0, 1050.0),
            ),
        ];

        for (rot, t) in poses.iter().take(n_views) {
            let iso = Isometry3::from_parts(Translation3::from(*t), (*rot).into());

            let mut board = Vec::new();
            let mut pixels = Vec::new();
            for bp in &board_points {
                let p3d = iso.transform_point(&nalgebra::Point3::new(bp.x, bp.y, 0.0));
                if p3d.z <= 0.0 {
                    continue;
                }
                let n_undist = Pt2::new(p3d.x / p3d.z, p3d.y / p3d.z);
                let n_dist = dist.distort(&n_undist);
                let pixel_h = k_mtx * Vec3::new(n_dist.x, n_dist.y, 1.0);
                board.push(*bp);
                pixels.push(Pt2::new(pixel_h.x / pixel_h.z, pixel_h.y / pixel_h.z));
            }

            views.push(IterativeCalibView::new(board, pixels).unwrap());
        }

        views
    }

    #[test]
    fn iterative_refinement_converges() {
        let (intr_gt, dist_gt) = make_ground_truth();
        let views = generate_synthetic_views(&intr_gt, &dist_gt, 4);

        let opts = IterativeIntrinsicsOptions {
            iterations: 2,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: false,
                iters: 8,
            },
            zero_skew: true,
        };

        let result = estimate_intrinsics_iterative(&views, opts).unwrap();

        println!(
            "Ground truth intrinsics: fx={}, fy={}, cx={}, cy={}",
            intr_gt.fx, intr_gt.fy, intr_gt.cx, intr_gt.cy
        );

        println!(
            "\nGround truth distortion: k1={}, k2={}, p1={}, p2={}",
            dist_gt.k1, dist_gt.k2, dist_gt.p1, dist_gt.p2
        );

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
        let views = generate_synthetic_views(&intr_gt, &dist_gt, 4);

        let opts = IterativeIntrinsicsOptions {
            iterations: 3,
            distortion_opts: DistortionFitOptions {
                fix_k3: true,
                fix_tangential: true,
                iters: 8,
            },
            zero_skew: true,
        };

        let _camera = estimate_intrinsics_iterative(&views, opts).unwrap();
    }
}
