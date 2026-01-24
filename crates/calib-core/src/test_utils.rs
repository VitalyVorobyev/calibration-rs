//! Utilities and common types for testing calibration algorithms.
//!
//! This module is public to allow use across workspace test suites,
//! but is not intended for production use. It provides shared data structures
//! and helper functions for working with calibration test data.

use crate::{BrownConrady5, Mat3, Pt2, Real};
use serde::Deserialize;

/// A calibration board view with detections for multiple cameras.
///
/// Used for stereo calibration test data where each view contains
/// corner detections from both left and right cameras.
#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationView {
    /// Sequential view index.
    pub view_index: usize,
    /// Left camera detections.
    pub left: ViewDetections,
    /// Right camera detections.
    pub right: ViewDetections,
}

/// Corner detections for a single camera view.
///
/// Each corner is represented as a 4-element array: `[i, j, x, y]`
/// where `(i, j)` is the board grid index and `(x, y)` is the pixel coordinate.
#[derive(Debug, Clone, Deserialize)]
pub struct ViewDetections {
    /// Corner detections, each encoded as `[i, j, x, y]`.
    pub corners: Vec<[Real; 4]>,
}

/// Preprocessed corner information including undistorted coordinates.
///
/// This structure is used in tests that require ground truth undistortion
/// to validate linear algorithms that assume distortion-free inputs.
#[derive(Debug, Clone, Copy)]
pub struct CornerInfo {
    /// Board column index.
    pub i: usize,
    /// Board row index.
    pub j: usize,
    /// Undistorted normalized coordinates (after K^-1 and undistortion).
    pub undist_norm: Pt2,
    /// Undistorted pixel coordinates.
    pub undist_pixel: Pt2,
}

/// Undistort a pixel coordinate using intrinsics and distortion model.
///
/// This is a wrapper around [`crate::math::undistort_pixel`] that accepts
/// the Brown-Conrady distortion model specifically. Use this function in tests
/// for backward compatibility, or use the generic version from the math module
/// for production code.
///
/// # Arguments
/// * `pixel` - Distorted pixel coordinate
/// * `intrinsics` - Camera intrinsics matrix K
/// * `distortion` - Brown-Conrady distortion model
///
/// # Returns
/// Undistorted normalized coordinates (on the Z=1 plane in camera frame).
pub fn undistort_pixel_normalized(
    pixel: Pt2,
    intrinsics: &Mat3,
    distortion: &BrownConrady5<Real>,
) -> Pt2 {
    crate::math::undistort_pixel(pixel, intrinsics, distortion)
}

/// Project normalized coordinates to pixel coordinates using intrinsics.
///
/// This is a wrapper around [`crate::math::normalized_to_pixel`]. Use this
/// function in tests for backward compatibility, or use the version from
/// the math module for production code.
///
/// # Arguments
/// * `normalized` - Normalized coordinates (on Z=1 plane)
/// * `intrinsics` - Camera intrinsics matrix K
///
/// # Returns
/// Pixel coordinates computed as `K * [x, y, 1]^T` (homogeneous division applied).
pub fn pixel_from_normalized(normalized: Pt2, intrinsics: &Mat3) -> Pt2 {
    crate::math::normalized_to_pixel(normalized, intrinsics)
}

/// Build corner info list with ground truth undistortion applied.
///
/// This function processes raw corner detections by:
/// 1. Undistorting each pixel using ground truth K and distortion
/// 2. Computing both undistorted normalized and pixel coordinates
/// 3. Sorting corners by (i, j) grid index
///
/// # Arguments
/// * `detections` - Raw corner detections from calibration pattern
/// * `intrinsics` - Ground truth camera intrinsics
/// * `distortion` - Ground truth distortion model
///
/// # Returns
/// Vector of `CornerInfo` structs sorted by grid indices.
///
/// # Note
/// This function is intended for testing algorithms that assume undistorted
/// inputs. Real-world calibration should not require ground truth distortion.
pub fn build_corner_info(
    detections: &ViewDetections,
    intrinsics: &Mat3,
    distortion: &BrownConrady5<Real>,
) -> Vec<CornerInfo> {
    let mut corners = Vec::with_capacity(detections.corners.len());
    for c in &detections.corners {
        let i = c[0] as usize;
        let j = c[1] as usize;
        let pixel = Pt2::new(c[2], c[3]);
        let undist_norm = undistort_pixel_normalized(pixel, intrinsics, distortion);
        let undist_pixel = pixel_from_normalized(undist_norm, intrinsics);
        corners.push(CornerInfo {
            i,
            j,
            undist_norm,
            undist_pixel,
        });
    }
    corners.sort_by_key(|c| (c.i, c.j));
    corners
}
