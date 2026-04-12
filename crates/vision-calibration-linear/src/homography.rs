//! Homography estimation — re-exported from `vision-geometry`.
//!
//! This module provides backward-compatible access to homography solvers via
//! both free functions and the [`HomographySolver`] namespace struct.

use anyhow::Result;
use vision_calibration_core::{Mat3, Pt2, RansacOptions};

// Re-export free functions from vision-geometry.
pub use vision_geometry::homography::*;

/// High-level entry point for homography estimation.
///
/// This is a thin wrapper around the DLT and DLT+RANSAC helpers and is
/// provided mainly for API consistency with other solvers.
#[derive(Debug, Clone, Copy)]
pub struct HomographySolver;

impl HomographySolver {
    /// Estimate a homography `H` such that `x' ~ H x` using the normalized DLT.
    pub fn dlt(src: &[Pt2], dst: &[Pt2]) -> Result<Mat3> {
        vision_geometry::homography::dlt_homography(src, dst)
    }

    /// Estimate a homography using DLT inside a RANSAC loop.
    pub fn dlt_ransac(
        src: &[Pt2],
        dst: &[Pt2],
        opts: &RansacOptions,
    ) -> Result<(Mat3, Vec<usize>)> {
        vision_geometry::homography::dlt_homography_ransac(src, dst, opts)
    }
}
