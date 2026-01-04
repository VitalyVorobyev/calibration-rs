//! Closed-form distortion coefficient estimation from homography residuals.
//!
//! This module implements a linear least-squares approximation to estimate
//! Brown-Conrady distortion parameters (k1, k2, k3, p1, p2) from pixel residuals
//! observed when comparing homography-predicted positions with actual observations.
//!
//! # Algorithm Overview
//!
//! Given camera intrinsics K and multiple views with known homographies H_i:
//!
//! 1. For each 2D-2D correspondence (board point → pixel):
//!    - Compute ideal pixel: `p_ideal = K * H * board_point`
//!    - Convert both to normalized coords: `n_ideal = K^-1 * p_ideal`, `n_obs = K^-1 * p_obs`
//!    - The residual `n_obs - n_ideal` contains distortion effects
//!
//! 2. Model the distortion as a linear function of radial and tangential coefficients:
//!    - Radial: `n_dist = n_undist * (1 + k1*r² + k2*r⁴ + k3*r⁶)`
//!    - Tangential: additional terms with p1, p2
//!
//! 3. Linearize and solve the overdetermined system `A*x = b` via SVD
//!
//! # When to Use
//!
//! This estimator is intended for **initialization** before non-linear refinement.
//! It works well for small-to-moderate distortion but may be inaccurate for
//! wide-angle lenses with severe distortion.
//!
//! # Limitations
//!
//! - Assumes distortion is small enough that linearization is valid
//! - Requires sufficient radial diversity (points at various distances from principal point)
//! - Cannot handle degenerate cases where all points are near the image center
//!
//! # References
//!
//! - Z. Zhang, "A Flexible New Technique for Camera Calibration," PAMI 2000
//! - OpenCV calibration implementation

use calib_core::{BrownConrady5, Mat3, Pt2, Real, Vec2, Vec3};
use nalgebra::DMatrix;
use thiserror::Error;

/// Errors that can occur during distortion estimation.
#[derive(Debug, Error, Clone, Copy)]
pub enum DistortionFitError {
    /// Not enough points for the requested parameter fit.
    #[error("need at least {0} points for distortion estimation, got {1}")]
    NotEnoughPoints(usize, usize),
    /// SVD failed during parameter estimation.
    #[error("svd failed during distortion estimation")]
    SvdFailed,
    /// Intrinsics matrix is not invertible.
    #[error("intrinsics matrix is not invertible")]
    IntrinsicsNotInvertible,
    /// Degenerate configuration: insufficient radial diversity.
    #[error("degenerate configuration: all points near image center")]
    DegenerateConfiguration,
}

/// Options controlling distortion parameter estimation.
#[derive(Debug, Clone, Copy)]
pub struct DistortionFitOptions {
    /// Fix tangential distortion coefficients (p1, p2) to zero.
    ///
    /// When `true`, only radial coefficients (k1, k2, k3) are estimated.
    /// Tangential distortion models decentering and thin prism effects,
    /// which are often negligible for well-manufactured lenses.
    pub fix_tangential: bool,

    /// Fix the third radial coefficient (k3) to zero.
    ///
    /// The k3 term (r⁶) can overfit with typical calibration data.
    /// Recommended to keep fixed unless using wide-angle lenses or
    /// high-quality calibration patterns with many diverse views.
    pub fix_k3: bool,

    /// Number of undistortion iterations for the returned `BrownConrady5` model.
    ///
    /// This does not affect the estimation process, only the iterative
    /// undistortion behavior of the returned distortion model.
    pub iters: u32,
}

impl Default for DistortionFitOptions {
    fn default() -> Self {
        Self {
            fix_tangential: false,
            fix_k3: true, // Conservative default: 3-parameter radial only
            iters: 8,
        }
    }
}

/// A single view's observations for distortion fitting.
///
/// Each view contains:
/// - A homography mapping 2D board points to pixels (computed WITHOUT distortion correction)
/// - Corresponding 2D board points and observed pixel coordinates
///
/// The homography represents the "ideal" pinhole projection, and residuals
/// between homography predictions and observations reveal distortion effects.
#[derive(Debug, Clone)]
pub struct DistortionView {
    /// Homography mapping board 2D coordinates to pixels.
    ///
    /// This should be computed from the **distorted** pixel observations
    /// (not pre-undistorted), as we want the residuals to contain distortion.
    pub homography: Mat3,

    /// 2D board coordinates (Z=0 plane, e.g., grid points in millimeters).
    pub board_points: Vec<Pt2>,

    /// Observed pixel coordinates (distorted).
    pub pixel_points: Vec<Pt2>,
}

impl DistortionView {
    /// Create a new distortion view.
    ///
    /// # Errors
    ///
    /// Returns `NotEnoughPoints` if `board_points` and `pixel_points` have different lengths.
    pub fn new(
        homography: Mat3,
        board_points: Vec<Pt2>,
        pixel_points: Vec<Pt2>,
    ) -> Result<Self, DistortionFitError> {
        if board_points.len() != pixel_points.len() {
            return Err(DistortionFitError::NotEnoughPoints(
                board_points.len(),
                pixel_points.len(),
            ));
        }
        Ok(Self {
            homography,
            board_points,
            pixel_points,
        })
    }
}

/// Estimate Brown-Conrady distortion from multiple views with known intrinsics.
///
/// Each view must have a homography mapping board points to pixels. This function
/// assumes the homographies were computed WITHOUT distortion correction (i.e., using
/// distorted pixel observations).
///
/// # Arguments
///
/// * `intrinsics` - Camera intrinsics matrix K (3x3)
/// * `views` - Multiple views with homographies and point correspondences
/// * `opts` - Options controlling which parameters to estimate
///
/// # Returns
///
/// Estimated distortion coefficients as a `BrownConrady5<Real>` struct.
///
/// # Errors
///
/// - `NotEnoughPoints`: Insufficient points for the requested parameter count
/// - `IntrinsicsNotInvertible`: K matrix is singular
/// - `DegenerateConfiguration`: All points near image center (no radial diversity)
/// - `SvdFailed`: Numerical issues during linear solve
///
/// # Coordinate Conventions
///
/// - Input pixels are **distorted** (raw observations from detections)
/// - Homographies computed from **distorted** pixels
/// - K represents the **undistorted** camera model
///
/// # Example
///
/// ```no_run
/// use calib_core::{Mat3, Pt2};
/// use calib_linear::distortion_fit::{DistortionView, DistortionFitOptions, estimate_distortion_from_homographies};
///
/// let k = Mat3::identity(); // Your intrinsics
/// let views = vec![
///     DistortionView::new(
///         Mat3::identity(), // homography
///         vec![Pt2::new(0.0, 0.0), /* ... */],
///         vec![Pt2::new(320.0, 240.0), /* ... */],
///     ).unwrap(),
/// ];
/// let opts = DistortionFitOptions::default();
/// let distortion = estimate_distortion_from_homographies(&k, &views, opts).unwrap();
/// ```
pub fn estimate_distortion_from_homographies(
    intrinsics: &Mat3,
    views: &[DistortionView],
    opts: DistortionFitOptions,
) -> Result<BrownConrady5<Real>, DistortionFitError> {
    // Count total points
    let total_points: usize = views.iter().map(|v| v.board_points.len()).sum();

    // Determine required parameter count
    let n_params = match (opts.fix_tangential, opts.fix_k3) {
        (true, true) => 2,   // k1, k2 only
        (true, false) => 3,  // k1, k2, k3
        (false, true) => 4,  // k1, k2, p1, p2
        (false, false) => 5, // All parameters
    };

    #[allow(clippy::manual_div_ceil)] // Type ambiguity with div_ceil on usize
    let min_points = (n_params + 1) / 2 + 2; // Need overdetermined system
    if total_points < min_points {
        return Err(DistortionFitError::NotEnoughPoints(
            min_points,
            total_points,
        ));
    }

    // Invert intrinsics once
    let k_inv = intrinsics
        .try_inverse()
        .ok_or(DistortionFitError::IntrinsicsNotInvertible)?;

    // Build design matrix A and observation vector b
    // Each point contributes 2 rows (x and y residuals)
    let mut a = DMatrix::<Real>::zeros(2 * total_points, n_params);
    let mut b = nalgebra::DVector::<Real>::zeros(2 * total_points);

    let mut row_idx = 0;
    for view in views {
        for (board_pt, pixel_obs) in view.board_points.iter().zip(&view.pixel_points) {
            // Compute ideal pixel via homography
            let board_h = Vec3::new(board_pt.x, board_pt.y, 1.0);
            let pixel_ideal_h = view.homography * board_h;
            let pixel_ideal = Pt2::new(
                pixel_ideal_h.x / pixel_ideal_h.z,
                pixel_ideal_h.y / pixel_ideal_h.z,
            );

            // Convert both to normalized coordinates
            let n_ideal_h = k_inv * Vec3::new(pixel_ideal.x, pixel_ideal.y, 1.0);
            let n_ideal = Vec2::new(n_ideal_h.x / n_ideal_h.z, n_ideal_h.y / n_ideal_h.z);

            let n_obs_h = k_inv * Vec3::new(pixel_obs.x, pixel_obs.y, 1.0);
            let n_obs = Vec2::new(n_obs_h.x / n_obs_h.z, n_obs_h.y / n_obs_h.z);

            // Residual (contains distortion effects)
            let residual = n_obs - n_ideal;

            // Radial distance squared
            let x = n_ideal.x;
            let y = n_ideal.y;
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;

            // Build row for this point
            // Distortion model: n_obs ≈ n_ideal + distortion(n_ideal)
            // For radial: distortion_x = x * (k1*r² + k2*r⁴ + k3*r⁶)
            // For tangential: distortion_x += 2*p1*xy + p2*(r² + 2*x²)

            let mut col_idx = 0;

            // k1 contribution
            a[(row_idx, col_idx)] = x * r2;
            a[(row_idx + 1, col_idx)] = y * r2;
            col_idx += 1;

            // k2 contribution
            a[(row_idx, col_idx)] = x * r4;
            a[(row_idx + 1, col_idx)] = y * r4;
            col_idx += 1;

            // k3 contribution (if not fixed)
            if !opts.fix_k3 {
                a[(row_idx, col_idx)] = x * r6;
                a[(row_idx + 1, col_idx)] = y * r6;
                col_idx += 1;
            }

            // Tangential terms (if not fixed)
            if !opts.fix_tangential {
                let xy = x * y;
                let x2 = x * x;
                let y2 = y * y;

                // p1 contribution
                a[(row_idx, col_idx)] = 2.0 * xy;
                a[(row_idx + 1, col_idx)] = r2 + 2.0 * y2;
                col_idx += 1;

                // p2 contribution
                a[(row_idx, col_idx)] = r2 + 2.0 * x2;
                a[(row_idx + 1, col_idx)] = 2.0 * xy;
            }

            // Observation vector
            b[row_idx] = residual.x;
            b[row_idx + 1] = residual.y;

            row_idx += 2;
        }
    }

    // Check for degenerate configuration (all r² too small)
    let mut max_r2 = 0.0;
    for view in views {
        for board_pt in &view.board_points {
            let board_h = Vec3::new(board_pt.x, board_pt.y, 1.0);
            let pixel_ideal_h = view.homography * board_h;
            let pixel_ideal = Pt2::new(
                pixel_ideal_h.x / pixel_ideal_h.z,
                pixel_ideal_h.y / pixel_ideal_h.z,
            );
            let n_ideal_h = k_inv * Vec3::new(pixel_ideal.x, pixel_ideal.y, 1.0);
            let n_ideal = Vec2::new(n_ideal_h.x / n_ideal_h.z, n_ideal_h.y / n_ideal_h.z);
            let r2 = n_ideal.x * n_ideal.x + n_ideal.y * n_ideal.y;
            if r2 > max_r2 {
                max_r2 = r2;
            }
        }
    }

    if max_r2 < 1e-6 {
        return Err(DistortionFitError::DegenerateConfiguration);
    }

    // Solve least-squares: x = A \ b via SVD (handles overdetermined systems)
    let svd = a.svd(true, true);
    let x = svd
        .solve(&b, 1e-10)
        .map_err(|_| DistortionFitError::SvdFailed)?;

    // Extract parameters
    let mut col_idx = 0;
    let k1 = x[col_idx];
    col_idx += 1;
    let k2 = x[col_idx];
    col_idx += 1;
    let k3 = if opts.fix_k3 {
        0.0
    } else {
        let val = x[col_idx];
        col_idx += 1;
        val
    };
    let (p1, p2) = if opts.fix_tangential {
        (0.0, 0.0)
    } else {
        let p1 = x[col_idx];
        col_idx += 1;
        let p2 = x[col_idx];
        (p1, p2)
    };

    Ok(BrownConrady5 {
        k1,
        k2,
        k3,
        p1,
        p2,
        iters: opts.iters,
    })
}

/// High-level solver struct for distortion estimation.
///
/// Provides a consistent API with other solvers in `calib-linear`.
#[derive(Debug, Clone, Copy)]
pub struct DistortionSolver;

impl DistortionSolver {
    /// Estimate distortion coefficients from homography residuals.
    ///
    /// See [`estimate_distortion_from_homographies`] for details.
    pub fn from_homographies(
        intrinsics: &Mat3,
        views: &[DistortionView],
        opts: DistortionFitOptions,
    ) -> Result<BrownConrady5<Real>, DistortionFitError> {
        estimate_distortion_from_homographies(intrinsics, views, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use calib_core::DistortionModel;
    use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, Vector3};

    fn make_kmtx() -> Mat3 {
        Matrix3::new(800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0)
    }

    fn synthetic_homography_with_distortion(
        kmtx: &Mat3,
        dist: &BrownConrady5<Real>,
        rot: Rotation3<Real>,
        t: Vector3<Real>,
        board_points: &[Pt2],
    ) -> (Mat3, Vec<Pt2>) {
        // Construct pose
        let iso = Isometry3::from_parts(Translation3::from(t), rot.into());

        // Generate distorted pixels
        let mut pixels = Vec::new();
        for bp in board_points {
            let p3d = iso.transform_point(&nalgebra::Point3::new(bp.x, bp.y, 0.0));
            if p3d.z <= 0.0 {
                continue;
            }
            let n_undist = Vec2::new(p3d.x / p3d.z, p3d.y / p3d.z);
            let n_dist = dist.distort(&n_undist);
            let pixel_h = kmtx * Vec3::new(n_dist.x, n_dist.y, 1.0);
            pixels.push(Pt2::new(pixel_h.x / pixel_h.z, pixel_h.y / pixel_h.z));
        }

        // Compute homography from undistorted normalized coordinates
        // H = K [r1 r2 t]
        let binding = iso.rotation.to_rotation_matrix();
        let r_mat = binding.matrix();
        let r1 = r_mat.column(0);
        let r2 = r_mat.column(1);

        let mut hmtx = Mat3::zeros();
        hmtx.set_column(0, &(kmtx * r1));
        hmtx.set_column(1, &(kmtx * r2));
        hmtx.set_column(2, &(kmtx * t));

        (hmtx, pixels)
    }

    #[test]
    fn synthetic_radial_only_recovers_k1_k2() {
        let kmtx = make_kmtx();
        let dist_gt = BrownConrady5 {
            k1: -0.2,
            k2: 0.05,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 8,
        };

        // Generate board points (7x7 grid, 30mm spacing)
        let mut board_points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                board_points.push(Pt2::new(i as Real * 30.0, j as Real * 30.0));
            }
        }

        // Three diverse views
        let poses = vec![
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
        ];

        let mut views = Vec::new();
        for (rot, t) in poses {
            let (h, pixels) =
                synthetic_homography_with_distortion(&kmtx, &dist_gt, rot, t, &board_points);
            views.push(DistortionView::new(h, board_points.clone(), pixels).unwrap());
        }

        let opts = DistortionFitOptions {
            fix_tangential: true,
            fix_k3: true,
            iters: 8,
        };

        let dist_est = estimate_distortion_from_homographies(&kmtx, &views, opts).unwrap();

        println!("Ground truth: k1={}, k2={}", dist_gt.k1, dist_gt.k2);
        println!("Estimated:    k1={}, k2={}", dist_est.k1, dist_est.k2);

        // Linear approximation, expect ~20-30% accuracy
        assert!((dist_est.k1 - dist_gt.k1).abs() < 0.1, "k1 error too large");
        assert!(
            (dist_est.k2 - dist_gt.k2).abs() < 0.03,
            "k2 error too large"
        );
        assert_eq!(dist_est.k3, 0.0);
        assert_eq!(dist_est.p1, 0.0);
        assert_eq!(dist_est.p2, 0.0);
    }

    #[test]
    fn tangential_distortion_estimation_reasonable() {
        let kmtx = make_kmtx();
        let dist_gt = BrownConrady5 {
            k1: -0.15,
            k2: 0.02,
            k3: 0.0,
            p1: 0.001,
            p2: -0.002,
            iters: 8,
        };

        let mut board_points = Vec::new();
        for i in 0..7 {
            for j in 0..7 {
                board_points.push(Pt2::new(i as Real * 30.0, j as Real * 30.0));
            }
        }

        let poses = vec![
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
        ];

        let mut views = Vec::new();
        for (rot, t) in poses {
            let (h, pixels) =
                synthetic_homography_with_distortion(&kmtx, &dist_gt, rot, t, &board_points);
            views.push(DistortionView::new(h, board_points.clone(), pixels).unwrap());
        }

        let opts = DistortionFitOptions {
            fix_tangential: false,
            fix_k3: true,
            iters: 8,
        };

        let dist_est = estimate_distortion_from_homographies(&kmtx, &views, opts).unwrap();

        println!(
            "Ground truth: k1={}, k2={}, p1={}, p2={}",
            dist_gt.k1, dist_gt.k2, dist_gt.p1, dist_gt.p2
        );
        println!(
            "Estimated:    k1={}, k2={}, p1={}, p2={}",
            dist_est.k1, dist_est.k2, dist_est.p1, dist_est.p2
        );

        // Check sign and order of magnitude
        assert!(
            dist_est.k1.signum() == dist_gt.k1.signum(),
            "k1 sign mismatch"
        );
        assert!(dist_est.p1.abs() < 0.01, "p1 order of magnitude reasonable");
        assert!(dist_est.p2.abs() < 0.01, "p2 order of magnitude reasonable");
    }
}
