//! Perspective-n-Point (PnP) solvers for camera pose estimation.
//!
//! Includes:
//! - DLT (linear) pose estimation with normalization.
//! - P3P minimal solver (3 points, multiple solutions).
//! - EPnP (control-point formulation) for 4+ points.
//! - DLT wrapped in RANSAC for outlier rejection.
//!
//! All methods estimate a pose `T_C_W`: transform from world coordinates into
//! the camera frame.

use crate::Error;
use vision_calibration_core::{FxFyCxCySkew, Iso3, Pt2, Pt3, RansacOptions, Real};

mod dlt;
mod epnp;
mod p3p;
mod pose_utils;
mod ransac;

// Re-export public API
pub use dlt::dlt;
pub use epnp::epnp;
pub use p3p::p3p;
pub use ransac::dlt_ransac;

/// Linear PnP solver (DLT) for camera pose estimation.
///
/// This solves for a pose `T_C_W` (world to camera) from 3D points and their
/// 2D projections, using a Direct Linear Transform and optionally wrapping it
/// in a RANSAC loop.
#[derive(Debug, Clone, Copy)]
pub struct PnpSolver;

impl PnpSolver {
    /// Direct linear PnP on all inliers.
    ///
    /// `world` are 3D points in world coordinates, `image` are their pixel
    /// positions, and `k` are the camera intrinsics. Uses a normalized DLT
    /// solve and projects the rotation onto SO(3).
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 6 correspondences are given,
    /// or [`Error::Singular`] if normalization or SVD fails.
    pub fn dlt(world: &[Pt3], image: &[Pt2], k: &FxFyCxCySkew<Real>) -> Result<Iso3, Error> {
        dlt::dlt(world, image, k)
    }

    /// P3P minimal solver: returns up to four pose candidates.
    ///
    /// Requires exactly three non-collinear points and intrinsics `k` to
    /// convert pixels into rays. The resulting poses are in `T_C_W` form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the correspondence count is not exactly 3,
    /// or [`Error::Singular`] if the intrinsics are not invertible.
    pub fn p3p(world: &[Pt3], image: &[Pt2], k: &FxFyCxCySkew<Real>) -> Result<Vec<Iso3>, Error> {
        p3p::p3p(world, image, k)
    }

    /// EPnP pose estimation for 4+ points.
    ///
    /// Uses a control-point formulation derived from the covariance of the
    /// 3D points. Returns a single pose estimate in `T_C_W` form.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 4 correspondences are given,
    /// or [`Error::Singular`] if SVD fails.
    pub fn epnp(world: &[Pt3], image: &[Pt2], k: &FxFyCxCySkew<Real>) -> Result<Iso3, Error> {
        epnp::epnp(world, image, k)
    }

    /// Robust PnP using DLT inside a RANSAC loop.
    ///
    /// Returns the best pose and inlier indices. The residual is pixel
    /// reprojection error using the provided intrinsics.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] if fewer than 6 correspondences are given,
    /// or [`Error::Numerical`] if RANSAC fails to find a consensus.
    pub fn dlt_ransac(
        world: &[Pt3],
        image: &[Pt2],
        k: &FxFyCxCySkew<Real>,
        opts: &RansacOptions,
    ) -> Result<(Iso3, Vec<usize>), Error> {
        ransac::dlt_ransac(world, image, k, opts)
    }
}
