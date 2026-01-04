//! Coordinate transformation utilities for camera projection.
//!
//! This module provides functions for converting between pixel coordinates,
//! normalized camera coordinates, and for applying/reversing distortion models.

use crate::{DistortionModel, Mat3, Pt2, Real, Vec2, Vec3};

/// Convert pixel coordinates to normalized coordinates using intrinsics.
///
/// Applies K^{-1} to transform pixel coordinates to the normalized image plane
/// (Z=1 in camera frame).
///
/// # Arguments
/// * `pixel` - Pixel coordinates (u, v)
/// * `intrinsics` - Camera intrinsics matrix K
///
/// # Returns
/// Normalized coordinates (x, y) on the Z=1 plane.
///
/// # Panics
/// Panics if the intrinsics matrix is singular (non-invertible).
///
/// # Example
/// ```no_run
/// use calib_core::{Mat3, Pt2};
/// use calib_core::math::pixel_to_normalized;
///
/// let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
/// let pixel = Pt2::new(640.0, 480.0);
/// let normalized = pixel_to_normalized(pixel, &k);
/// // normalized ≈ (0.0, 0.0) for a pixel at the principal point
/// ```
pub fn pixel_to_normalized(pixel: Pt2, intrinsics: &Mat3) -> Vec2 {
    let k_inv = intrinsics
        .try_inverse()
        .expect("intrinsics matrix should be invertible");
    let v = k_inv * Vec3::new(pixel.x, pixel.y, 1.0);
    Vec2::new(v.x / v.z, v.y / v.z)
}

/// Convert normalized coordinates to pixel coordinates using intrinsics.
///
/// Applies K to transform normalized coordinates (Z=1 plane) to pixel coordinates.
///
/// # Arguments
/// * `normalized` - Normalized coordinates (x, y)
/// * `intrinsics` - Camera intrinsics matrix K
///
/// # Returns
/// Pixel coordinates computed as K * [x, y, 1]^T with homogeneous division.
///
/// # Example
/// ```no_run
/// use calib_core::{Mat3, Vec2};
/// use calib_core::math::normalized_to_pixel;
///
/// let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
/// let normalized = Vec2::new(0.0, 0.0);
/// let pixel = normalized_to_pixel(normalized, &k);
/// // pixel ≈ (640.0, 480.0) at the principal point
/// ```
pub fn normalized_to_pixel(normalized: Vec2, intrinsics: &Mat3) -> Pt2 {
    let v = intrinsics * Vec3::new(normalized.x, normalized.y, 1.0);
    Pt2::new(v.x / v.z, v.y / v.z)
}

/// Undistort pixel coordinates to normalized coordinates.
///
/// Combines pixel-to-normalized conversion with distortion model undistortion.
/// This is a common operation when preprocessing image points for algorithms
/// that assume ideal pinhole projection.
///
/// # Arguments
/// * `pixel` - Distorted pixel coordinates
/// * `intrinsics` - Camera intrinsics matrix K
/// * `distortion` - Distortion model implementing [`DistortionModel`]
///
/// # Returns
/// Undistorted normalized coordinates on the Z=1 plane.
///
/// # Algorithm
/// 1. Convert pixel to normalized coordinates: `n = K^{-1} * [x, y, 1]^T`
/// 2. Apply iterative undistortion: `n_undist = distortion.undistort(n)`
///
/// # Example
/// ```no_run
/// use calib_core::{BrownConrady5, Mat3, Pt2};
/// use calib_core::math::undistort_pixel;
///
/// let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
/// let distortion = BrownConrady5 { k1: -0.3, k2: 0.1, k3: 0.0, p1: 0.0, p2: 0.0, iters: 5 };
/// let pixel = Pt2::new(500.0, 400.0);
/// let undist_norm = undistort_pixel(pixel, &k, &distortion);
/// ```
pub fn undistort_pixel<D: DistortionModel<Real>>(
    pixel: Pt2,
    intrinsics: &Mat3,
    distortion: &D,
) -> Vec2 {
    let normalized = pixel_to_normalized(pixel, intrinsics);
    distortion.undistort(&normalized)
}

/// Apply distortion and convert to pixel coordinates.
///
/// Combines distortion model application with normalized-to-pixel conversion.
/// This is the inverse operation of [`undistort_pixel`] (approximately, depending
/// on distortion model convergence).
///
/// # Arguments
/// * `normalized` - Undistorted normalized coordinates
/// * `intrinsics` - Camera intrinsics matrix K
/// * `distortion` - Distortion model implementing [`DistortionModel`]
///
/// # Returns
/// Distorted pixel coordinates.
///
/// # Algorithm
/// 1. Apply forward distortion: `n_dist = distortion.distort(n)`
/// 2. Convert to pixels: `pixel = K * [n_dist.x, n_dist.y, 1]^T`
///
/// # Example
/// ```no_run
/// use calib_core::{BrownConrady5, Mat3, Vec2};
/// use calib_core::math::distort_to_pixel;
///
/// let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
/// let distortion = BrownConrady5 { k1: -0.3, k2: 0.1, k3: 0.0, p1: 0.0, p2: 0.0, iters: 5 };
/// let normalized = Vec2::new(-0.1, 0.05);
/// let pixel = distort_to_pixel(normalized, &k, &distortion);
/// ```
pub fn distort_to_pixel<D: DistortionModel<Real>>(
    normalized: Vec2,
    intrinsics: &Mat3,
    distortion: &D,
) -> Pt2 {
    let distorted = distortion.distort(&normalized);
    normalized_to_pixel(distorted, intrinsics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BrownConrady5;

    #[test]
    fn pixel_normalized_roundtrip() {
        let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 780.0, 360.0, 0.0, 0.0, 1.0);
        let pixel_orig = Pt2::new(700.0, 400.0);

        let normalized = pixel_to_normalized(pixel_orig, &k);
        let pixel_back = normalized_to_pixel(normalized, &k);

        assert!((pixel_back.x - pixel_orig.x).abs() < 1e-10);
        assert!((pixel_back.y - pixel_orig.y).abs() < 1e-10);
    }

    #[test]
    fn principal_point_maps_to_origin() {
        let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
        let pixel = Pt2::new(640.0, 480.0);

        let normalized = pixel_to_normalized(pixel, &k);

        assert!(normalized.x.abs() < 1e-10);
        assert!(normalized.y.abs() < 1e-10);
    }

    #[test]
    fn undistort_pixel_no_distortion() {
        let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
        let distortion = BrownConrady5 {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            iters: 5,
        };
        let pixel = Pt2::new(700.0, 500.0);

        let undist = undistort_pixel(pixel, &k, &distortion);
        let expected = pixel_to_normalized(pixel, &k);

        assert!((undist.x - expected.x).abs() < 1e-10);
        assert!((undist.y - expected.y).abs() < 1e-10);
    }

    #[test]
    fn distort_undistort_approximate_roundtrip() {
        let k = Mat3::new(800.0, 0.0, 640.0, 0.0, 800.0, 480.0, 0.0, 0.0, 1.0);
        let distortion = BrownConrady5 {
            k1: -0.3,
            k2: 0.1,
            k3: 0.0,
            p1: 0.001,
            p2: -0.001,
            iters: 5,
        };

        let normalized_orig = Vec2::new(-0.1, 0.05);
        let pixel_dist = distort_to_pixel(normalized_orig, &k, &distortion);
        let normalized_undist = undistort_pixel(pixel_dist, &k, &distortion);

        // Should be close but not exact due to iterative undistortion
        let diff = (normalized_undist - normalized_orig).norm();
        assert!(diff < 1e-6, "roundtrip error too large: {}", diff);
    }
}
