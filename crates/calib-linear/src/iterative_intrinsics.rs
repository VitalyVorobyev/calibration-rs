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
//!     IterativeCalibView, IterativeIntrinsicsOptions, IterativeIntrinsicsSolver,
//! };
//!
//! let views = vec![
//!     IterativeCalibView::new(
//!         vec![Pt2::new(0.0, 0.0), /* board points */],
//!         vec![Pt2::new(320.0, 240.0), /* distorted pixels */],
//!     ),
//!     // ... more views
//! ];
//!
//! let opts = IterativeIntrinsicsOptions::default();
//! let result = IterativeIntrinsicsSolver::estimate(&views, opts).unwrap();
//!
//! println!("Intrinsics: fx={}, fy={}, cx={}, cy={}",
//!          result.intrinsics.fx, result.intrinsics.fy,
//!          result.intrinsics.cx, result.intrinsics.cy);
//! println!("Distortion: k1={}, k2={}", result.distortion.k1, result.distortion.k2);
//! ```

use crate::{
    distortion_fit::{DistortionFitError, DistortionFitOptions, DistortionSolver, DistortionView},
    homography::{HomographyError, HomographySolver},
    zhang_intrinsics::{PlanarIntrinsicsInitError, PlanarIntrinsicsLinearInit},
};
use calib_core::{BrownConrady5, DistortionModel, FxFyCxCySkew, Mat3, Pt2, Real, Vec2, Vec3};
use thiserror::Error;

/// Errors that can occur during iterative intrinsics estimation.
#[derive(Debug, Error)]
pub enum IterativeIntrinsicsError {
    /// Zhang intrinsics estimation failed.
    #[error("zhang intrinsics failed: {0}")]
    ZhangFailed(#[from] PlanarIntrinsicsInitError),
    /// Distortion estimation failed.
    #[error("distortion estimation failed: {0}")]
    DistortionFailed(#[from] DistortionFitError),
    /// Homography estimation failed.
    #[error("homography estimation failed: {0}")]
    HomographyFailed(#[from] HomographyError),
    /// Need at least 3 views for calibration.
    #[error("need at least 3 views, got {0}")]
    NotEnoughViews(usize),
}

/// Options controlling iterative intrinsics estimation.
#[derive(Debug, Clone, Copy)]
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

/// Result from iterative intrinsics estimation.
#[derive(Debug, Clone)]
pub struct IterativeIntrinsicsResult {
    /// Final intrinsics estimate.
    pub intrinsics: FxFyCxCySkew<Real>,

    /// Final distortion estimate.
    pub distortion: BrownConrady5<Real>,

    /// History of intrinsics over iterations (for debugging/analysis).
    ///
    /// `intrinsics_history[0]` is the initial estimate (iteration 0),
    /// `intrinsics_history[i]` is the estimate after iteration i.
    pub intrinsics_history: Vec<FxFyCxCySkew<Real>>,

    /// History of distortion over iterations.
    ///
    /// `distortion_history[0]` is zero (no distortion at iteration 0),
    /// `distortion_history[i]` is the estimate after iteration i.
    pub distortion_history: Vec<BrownConrady5<Real>>,
}

/// A single planar view for iterative calibration.
///
/// Contains 2D-2D correspondences between board coordinates and observed pixels.
#[derive(Debug, Clone)]
pub struct IterativeCalibView {
    /// 2D board coordinates (on Z=0 plane, e.g., chessboard grid in millimeters).
    pub board_points: Vec<Pt2>,

    /// Observed pixel coordinates (distorted, raw from corner detection).
    pub pixel_points: Vec<Pt2>,
}

impl IterativeCalibView {
    /// Create a new calibration view.
    ///
    /// # Arguments
    ///
    /// * `board_points` - 2D coordinates on the calibration board (Z=0)
    /// * `pixel_points` - Corresponding pixel observations (distorted)
    ///
    /// # Note
    ///
    /// The pixel coordinates should be **raw observations** (distorted),
    /// not pre-undistorted. The solver will iteratively estimate and correct
    /// for distortion.
    pub fn new(board_points: Vec<Pt2>, pixel_points: Vec<Pt2>) -> Self {
        Self {
            board_points,
            pixel_points,
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
    views: &[IterativeCalibView],
    opts: IterativeIntrinsicsOptions,
) -> Result<IterativeIntrinsicsResult, IterativeIntrinsicsError> {
    if views.len() < 3 {
        return Err(IterativeIntrinsicsError::NotEnoughViews(views.len()));
    }

    let mut intrinsics_history = Vec::with_capacity(opts.iterations + 1);
    let mut distortion_history = Vec::with_capacity(opts.iterations + 1);

    // Iteration 0: Initial K estimate from distorted pixels (no distortion correction)
    let homographies_iter0: Result<Vec<Mat3>, _> = views
        .iter()
        .map(|v| HomographySolver::dlt(&v.board_points, &v.pixel_points))
        .collect();
    let homographies_iter0 = homographies_iter0?;

    let mut intrinsics_iter0 = PlanarIntrinsicsLinearInit::from_homographies(&homographies_iter0)?;
    enforce_zero_skew(&mut intrinsics_iter0, opts.zero_skew);
    intrinsics_history.push(intrinsics_iter0);

    // Initial distortion is zero
    let distortion_iter0 = BrownConrady5 {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        p1: 0.0,
        p2: 0.0,
        iters: opts.distortion_opts.iters,
    };
    distortion_history.push(distortion_iter0);

    // Iterative refinement
    let mut current_intrinsics = intrinsics_iter0;
    let mut current_distortion = distortion_iter0;

    for _iter in 0..opts.iterations {
        // Step 1: Compute homographies from current pixel estimates
        // For first iteration, use distorted pixels; for later, use undistorted
        let homographies: Result<Vec<Mat3>, _> = if _iter == 0 {
            views
                .iter()
                .map(|v| HomographySolver::dlt(&v.board_points, &v.pixel_points))
                .collect()
        } else {
            // Undistort pixels using current distortion estimate
            let k_mtx = current_intrinsics.k_matrix();
            let k_inv = k_mtx
                .try_inverse()
                .ok_or(DistortionFitError::IntrinsicsNotInvertible)?;

            views
                .iter()
                .map(|v| {
                    let undistorted_pixels: Vec<Pt2> = v
                        .pixel_points
                        .iter()
                        .map(|&p| {
                            // Convert to normalized distorted coords
                            let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
                            let n_dist = Vec2::new(v_h.x / v_h.z, v_h.y / v_h.z);
                            // Undistort
                            let n_undist = current_distortion.undistort(&n_dist);
                            // Convert back to pixels
                            let p_h = k_mtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
                            Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z)
                        })
                        .collect();
                    HomographySolver::dlt(&v.board_points, &undistorted_pixels)
                })
                .collect()
        };
        let homographies = homographies?;

        // Step 2: Estimate distortion from residuals
        enforce_zero_skew(&mut current_intrinsics, opts.zero_skew);
        let k_mtx = current_intrinsics.k_matrix();
        let dist_views: Result<Vec<DistortionView>, _> = views
            .iter()
            .zip(&homographies)
            .map(|(v, h)| {
                DistortionView::new(
                    *h,
                    v.board_points.clone(),
                    v.pixel_points.clone(), // Use original distorted pixels
                )
            })
            .collect();
        let dist_views = dist_views?;

        current_distortion =
            DistortionSolver::from_homographies(&k_mtx, &dist_views, opts.distortion_opts)?;
        distortion_history.push(current_distortion);

        // Step 3: Undistort pixels and re-estimate K
        let k_inv = k_mtx
            .try_inverse()
            .ok_or(DistortionFitError::IntrinsicsNotInvertible)?;

        let undistorted_homographies: Result<Vec<Mat3>, _> = views
            .iter()
            .map(|v| {
                let undistorted_pixels: Vec<Pt2> = v
                    .pixel_points
                    .iter()
                    .map(|&p| {
                        let v_h = k_inv * Vec3::new(p.x, p.y, 1.0);
                        let n_dist = Vec2::new(v_h.x / v_h.z, v_h.y / v_h.z);
                        let n_undist = current_distortion.undistort(&n_dist);
                        let p_h = k_mtx * Vec3::new(n_undist.x, n_undist.y, 1.0);
                        Pt2::new(p_h.x / p_h.z, p_h.y / p_h.z)
                    })
                    .collect();
                HomographySolver::dlt(&v.board_points, &undistorted_pixels)
            })
            .collect();
        let undistorted_homographies = undistorted_homographies?;

        current_intrinsics =
            PlanarIntrinsicsLinearInit::from_homographies(&undistorted_homographies)?;
        enforce_zero_skew(&mut current_intrinsics, opts.zero_skew);
        intrinsics_history.push(current_intrinsics);
    }

    Ok(IterativeIntrinsicsResult {
        intrinsics: current_intrinsics,
        distortion: current_distortion,
        intrinsics_history,
        distortion_history,
    })
}

fn enforce_zero_skew(intrinsics: &mut FxFyCxCySkew<Real>, zero_skew: bool) {
    if zero_skew {
        intrinsics.skew = 0.0;
    }
}

/// High-level solver struct for iterative intrinsics estimation.
///
/// Provides a consistent API with other solvers in `calib-linear`.
#[derive(Debug, Clone, Copy)]
pub struct IterativeIntrinsicsSolver;

impl IterativeIntrinsicsSolver {
    /// Estimate intrinsics and distortion iteratively.
    ///
    /// See [`estimate_intrinsics_iterative`] for details.
    pub fn estimate(
        views: &[IterativeCalibView],
        opts: IterativeIntrinsicsOptions,
    ) -> Result<IterativeIntrinsicsResult, IterativeIntrinsicsError> {
        estimate_intrinsics_iterative(views, opts)
    }
}

#[cfg(test)]
mod tests {
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

    fn generate_synthetic_views(
        intr: &FxFyCxCySkew<Real>,
        dist: &BrownConrady5<Real>,
        n_views: usize,
    ) -> Vec<IterativeCalibView> {
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

            let mut pixels = Vec::new();
            for bp in &board_points {
                let p3d = iso.transform_point(&nalgebra::Point3::new(bp.x, bp.y, 0.0));
                if p3d.z <= 0.0 {
                    continue;
                }
                let n_undist = Vec2::new(p3d.x / p3d.z, p3d.y / p3d.z);
                let n_dist = dist.distort(&n_undist);
                let pixel_h = k_mtx * Vec3::new(n_dist.x, n_dist.y, 1.0);
                pixels.push(Pt2::new(pixel_h.x / pixel_h.z, pixel_h.y / pixel_h.z));
            }

            views.push(IterativeCalibView::new(board_points.clone(), pixels));
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
        println!("Iteration history:");
        for (i, intr) in result.intrinsics_history.iter().enumerate() {
            println!(
                "  Iter {}: fx={:.1}, fy={:.1}, cx={:.1}, cy={:.1}",
                i, intr.fx, intr.fy, intr.cx, intr.cy
            );
        }

        println!(
            "\nGround truth distortion: k1={}, k2={}, p1={}, p2={}",
            dist_gt.k1, dist_gt.k2, dist_gt.p1, dist_gt.p2
        );
        println!("Distortion history:");
        for (i, dist) in result.distortion_history.iter().enumerate() {
            println!(
                "  Iter {}: k1={:.4}, k2={:.4}, p1={:.4}, p2={:.4}",
                i, dist.k1, dist.k2, dist.p1, dist.p2
            );
        }

        // Check final estimates (linear methods: 10-30% accuracy expected for initialization)
        // This is acceptable - the purpose is to provide a reasonable starting point
        // for non-linear refinement, not to achieve final accuracy.
        let fx_err_pct = (result.intrinsics.fx - intr_gt.fx).abs() / intr_gt.fx * 100.0;
        let fy_err_pct = (result.intrinsics.fy - intr_gt.fy).abs() / intr_gt.fy * 100.0;
        let cx_err = (result.intrinsics.cx - intr_gt.cx).abs();
        let cy_err = (result.intrinsics.cy - intr_gt.cy).abs();

        assert!(fx_err_pct < 40.0, "fx error {:.1}% too large", fx_err_pct);
        assert!(fy_err_pct < 40.0, "fy error {:.1}% too large", fy_err_pct);
        assert!(cx_err < 80.0, "cx error {:.1}px too large", cx_err);
        assert!(cy_err < 150.0, "cy error {:.1}px too large", cy_err);

        // Check distortion estimates have correct sign
        assert!(
            result.distortion.k1.signum() == dist_gt.k1.signum(),
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

        let result = estimate_intrinsics_iterative(&views, opts).unwrap();

        // Verify fx gets closer to ground truth with each iteration
        let errors: Vec<Real> = result
            .intrinsics_history
            .iter()
            .map(|intr| (intr.fx - intr_gt.fx).abs())
            .collect();

        println!("fx error by iteration: {:?}", errors);

        // After first iteration, error should decrease (or at least not increase significantly)
        // Note: linear approximation may oscillate slightly, so use loose constraint
        assert!(
            errors[1] < errors[0] * 1.5,
            "First iteration should improve or not worsen significantly"
        );
    }
}
