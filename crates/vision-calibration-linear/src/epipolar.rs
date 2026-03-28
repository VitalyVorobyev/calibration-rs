//! Epipolar geometry solvers — re-exported from `vision-geometry`.
//!
//! This module provides backward-compatible access to epipolar solvers via
//! both free functions and the [`EpipolarSolver`] namespace struct.

use anyhow::Result;
use vision_calibration_core::{Mat3, Pt2, RansacOptions, Vec3};

// Re-export all free functions from vision-geometry.
pub use vision_geometry::epipolar::*;

/// Linear epipolar geometry solvers (fundamental / essential matrices).
///
/// All solvers are deterministic and use SVD-based nullspace extraction.
#[derive(Debug, Clone, Copy)]
pub struct EpipolarSolver;

impl EpipolarSolver {
    /// Normalized 8-point algorithm for the fundamental matrix.
    pub fn fundamental_8point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Mat3> {
        vision_geometry::epipolar::fundamental_8point(pts1, pts2)
    }

    /// 7-point algorithm for the fundamental matrix (minimal solver).
    pub fn fundamental_7point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>> {
        vision_geometry::epipolar::fundamental_7point(pts1, pts2)
    }

    /// 5-point algorithm for the essential matrix in normalized coordinates.
    pub fn essential_5point(pts1: &[Pt2], pts2: &[Pt2]) -> Result<Vec<Mat3>> {
        vision_geometry::epipolar::essential_5point(pts1, pts2)
    }

    /// Decompose an essential matrix into candidate rotation and translation pairs.
    pub fn decompose_essential(e: &Mat3) -> Result<Vec<(Mat3, Vec3)>> {
        vision_geometry::epipolar::decompose_essential(e)
    }

    /// Robust fundamental matrix estimation using the 8-point algorithm inside RANSAC.
    pub fn fundamental_8point_ransac(
        pts1: &[Pt2],
        pts2: &[Pt2],
        opts: &RansacOptions,
    ) -> Result<(Mat3, Vec<usize>)> {
        vision_geometry::epipolar::fundamental_8point_ransac(pts1, pts2, opts)
    }
}
