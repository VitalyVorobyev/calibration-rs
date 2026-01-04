//! Epipolar geometry solvers for fundamental and essential matrices.
//!
//! Includes normalized 8-point, 7-point, and 5-point minimal solvers, plus
//! decomposition of the essential matrix into candidate poses.
//!
//! - Fundamental matrix `F` expects **pixel coordinates** in both images.
//! - Essential matrix `E` expects **normalized coordinates** (after applying
//!   `K^{-1}`), or equivalently calibrated rays on the normalized image plane.

use calib_core::{Mat3, Pt2, RansacOptions, Vec3};
use thiserror::Error;

mod decomposition;
mod essential;
mod fundamental;
mod polynomial;

// Re-export public API
pub use decomposition::decompose_essential;
pub use essential::essential_5point;
pub use fundamental::{fundamental_7point, fundamental_8point, fundamental_8point_ransac};

/// Errors that can occur during fundamental / essential matrix estimation.
#[derive(Debug, Error)]
pub enum EpipolarError {
    /// Not enough point correspondences were provided.
    #[error("need at least 8 point correspondences, got {0}")]
    NotEnoughPoints(usize),
    /// Incorrect number of correspondences for a minimal solver.
    #[error("invalid number of correspondences: expected {expected}, got {got}")]
    InvalidPointCount { expected: usize, got: usize },
    /// Linear solve (SVD) failed.
    #[error("svd failed in epipolar estimation")]
    SvdFailed,
    /// Polynomial solve failed or produced no valid roots.
    #[error("failed to solve the epipolar polynomial system")]
    PolynomialSolveFailed,
    /// RANSAC failed to find a consensus model.
    #[error("ransac failed to find a consensus epipolar model")]
    RansacFailed,
}

/// Linear epipolar geometry solvers (fundamental / essential matrices).
///
/// All solvers are deterministic and use SVD-based nullspace extraction.
#[derive(Debug, Clone, Copy)]
pub struct EpipolarSolver;

impl EpipolarSolver {
    /// Normalized 8-point algorithm for the fundamental matrix.
    ///
    /// `pts1` and `pts2` are corresponding pixel points in two images. The
    /// returned matrix is forced to rank-2 and satisfies `x'^T F x = 0`
    /// (up to numerical error).
    pub fn fundamental_8point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3, EpipolarError> {
        fundamental::fundamental_8point(pts1, pts2)
    }

    /// 7-point algorithm for the fundamental matrix (minimal solver).
    ///
    /// Returns up to three candidate fundamental matrices. Inputs are pixel
    /// coordinates; internal normalization is applied before solving.
    pub fn fundamental_7point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>, EpipolarError> {
        fundamental::fundamental_7point(pts1, pts2)
    }

    /// 5-point algorithm for the essential matrix in normalized coordinates.
    ///
    /// The inputs must be **calibrated** (e.g. apply `K^{-1}` to pixel points).
    /// Returns up to ten candidate essential matrices that satisfy the cubic
    /// constraints; choose the physically valid one by cheirality or by
    /// reprojection error against additional correspondences.
    pub fn essential_5point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>, EpipolarError> {
        essential::essential_5point(pts1, pts2)
    }

    /// Decompose an essential matrix into candidate rotation and translation pairs.
    ///
    /// Returns four possible `(R, t)` pairs; the correct one can be selected by
    /// cheirality checks on triangulated points. The translation is unit-length
    /// (direction only).
    pub fn decompose_essential(e: &Mat3) -> Result<Vec<(Mat3, Vec3)>, EpipolarError> {
        decomposition::decompose_essential(e)
    }

    /// Robust fundamental matrix estimation using the 8-point algorithm inside RANSAC.
    ///
    /// Returns the best model and the indices of inliers. The residual uses an
    /// approximate symmetric epipolar distance in pixels.
    pub fn fundamental_8point_ransac(
        pts1: &[Pt2],
        pts2: &[Pt2],
        opts: &RansacOptions,
    ) -> Result<(Mat3, Vec<usize>), EpipolarError> {
        fundamental::fundamental_8point_ransac(pts1, pts2, opts)
    }
}
