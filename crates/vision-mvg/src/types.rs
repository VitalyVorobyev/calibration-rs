//! Core types for multiple-view geometry.
//!
//! Provides type aliases for geometric matrices and the input types of MVG
//! estimation: [`Correspondence2D`] for the uncalibrated estimators and
//! [`PixelMatch`] for the calibrated two-view functions.

use serde::{Deserialize, Serialize};
use vision_calibration_core::{Mat3, PointCovariance, Pt2, Pt3, Real};

/// Essential matrix (3×3, rank-2, encodes calibrated epipolar geometry).
pub type EssentialMatrix = Mat3;

/// Fundamental matrix (3×3, rank-2, encodes uncalibrated epipolar geometry).
pub type FundamentalMatrix = Mat3;

/// Homography matrix (3×3, invertible projective transform between planes).
pub type HomographyMatrix = Mat3;

/// A pair of corresponding points observed in two views, as plain image
/// coordinates, for the uncalibrated estimators (fundamental matrix,
/// homography). The calibrated functions take [`PixelMatch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correspondence2D {
    /// Point observed in the first view.
    pub pt1: Pt2,
    /// Point observed in the second view.
    pub pt2: Pt2,
}

impl Correspondence2D {
    /// Create a new correspondence from two 2D points.
    pub fn new(pt1: Pt2, pt2: Pt2) -> Self {
        Self { pt1, pt2 }
    }

    /// Split a slice of correspondences into two separate point vectors.
    ///
    /// This is useful for calling low-level solvers that accept `(&[Pt2], &[Pt2])`.
    pub fn split(corrs: &[Self]) -> (Vec<Pt2>, Vec<Pt2>) {
        let mut pts1 = Vec::with_capacity(corrs.len());
        let mut pts2 = Vec::with_capacity(corrs.len());
        for c in corrs {
            pts1.push(c.pt1);
            pts2.push(c.pt2);
        }
        (pts1, pts2)
    }
}

/// A pair of matched pixels with their stated noise, for the calibrated
/// two-view functions.
///
/// `pt1` is a pixel of the first camera and `pt2` of the second, the two
/// cameras the call names. Each pixel's covariance states its measurement
/// noise, and the functions weigh the epipolar constraint by it. Build a
/// covariance with [`PointCovariance::new`] (per observation),
/// [`PointCovariance::isotropic`] (one σ) or
/// [`CalibratedRig::reprojection_noise`](vision_calibration_core::CalibratedRig::reprojection_noise)
/// (the calibration's level).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelMatch {
    /// The pixel in the first camera.
    pub pt1: Pt2,
    /// Its covariance, px².
    pub noise1: PointCovariance,
    /// The pixel in the second camera.
    pub pt2: Pt2,
    /// Its covariance, px².
    pub noise2: PointCovariance,
}

impl PixelMatch {
    /// A match of `pt1` (with covariance `noise1`) to `pt2` (with `noise2`).
    pub fn new(pt1: Pt2, noise1: PointCovariance, pt2: Pt2, noise2: PointCovariance) -> Self {
        Self {
            pt1,
            noise1,
            pt2,
            noise2,
        }
    }
}

/// A triangulated 3D point with quality diagnostics.
#[derive(Debug, Clone, Serialize)]
pub struct TriangulatedPoint {
    /// Reconstructed 3D position, in the frame the triangulation's poses
    /// define (the world frame, or the first camera's frame for a relative
    /// pose).
    pub point: Pt3,
    /// Root-mean-square distance, in pixels, between each observation and the
    /// point projected through that observation's camera model. `None` when
    /// the point does not project in one of the cameras.
    pub reprojection_error: Option<Real>,
    /// Largest angle, in degrees, between two observing rays (the directions
    /// from each camera centre to the point).
    ///
    /// Small angles indicate poor triangulation geometry.
    pub parallax_deg: Real,
    /// Whether the point lies ahead along every observing bearing
    /// (`bearing · p_camera > 0`). For a pinhole this is positive depth; a
    /// wide-angle camera can see a point at negative `z` in front of it.
    pub in_front: bool,
}
