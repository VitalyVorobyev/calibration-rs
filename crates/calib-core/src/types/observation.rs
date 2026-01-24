//! Observation types for calibration data.
//!
//! This module provides canonical data structures for storing 2D-3D point
//! correspondences used throughout the calibration pipeline.

use crate::{Pt2, Pt3};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

/// A single view containing 2D-3D point correspondences.
///
/// This is the canonical observation type used across planar intrinsics,
/// rig extrinsics, hand-eye calibration, and other calibration problems.
///
/// # Fields
///
/// - `points_3d`: 3D points in world/target coordinates
/// - `points_2d`: Corresponding 2D pixel observations
/// - `weights`: Optional per-point weights for robust estimation
///
/// # Example
///
/// ```
/// use calib_core::{CorrespondenceView, Pt3, Pt2};
///
/// // Create from vectors
/// let points_3d = vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.1, 0.0, 0.0)];
/// let points_2d = vec![Pt2::new(320.0, 240.0), Pt2::new(400.0, 240.0)];
/// let view = CorrespondenceView::new(points_3d, points_2d).unwrap();
///
/// assert_eq!(view.len(), 2);
/// assert_eq!(view.weight(0), 1.0); // default weight
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrespondenceView {
    /// 3D points in world/target frame.
    pub points_3d: Vec<Pt3>,
    /// Corresponding 2D pixel observations.
    pub points_2d: Vec<Pt2>,
    /// Optional per-point weights (default: 1.0 for all points).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
}

impl CorrespondenceView {
    /// Construct observations without per-point weights.
    ///
    /// # Errors
    ///
    /// Returns an error if the 3D and 2D point counts don't match.
    pub fn new(points_3d: Vec<Pt3>, points_2d: Vec<Pt2>) -> Result<Self> {
        ensure!(
            points_3d.len() == points_2d.len(),
            "3D / 2D point counts must match: {} vs {}",
            points_3d.len(),
            points_2d.len()
        );
        Ok(Self {
            points_3d,
            points_2d,
            weights: None,
        })
    }

    pub fn planar_points(&self) -> Vec<Pt2> {
        self.points_3d
            .iter()
            .map(|p3| Pt2::new(p3.x, p3.y))
            .collect()
    }

    /// Construct observations with per-point weights.
    ///
    /// # Errors
    ///
    /// Returns an error if counts don't match or weights are negative.
    pub fn new_with_weights(
        points_3d: Vec<Pt3>,
        points_2d: Vec<Pt2>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        ensure!(
            points_3d.len() == points_2d.len(),
            "3D / 2D point counts must match: {} vs {}",
            points_3d.len(),
            points_2d.len()
        );
        ensure!(
            weights.len() == points_3d.len(),
            "weight count must match point count: {} vs {}",
            weights.len(),
            points_3d.len()
        );
        ensure!(
            weights.iter().all(|w| *w >= 0.0),
            "weights must be non-negative"
        );
        Ok(Self {
            points_3d,
            points_2d,
            weights: Some(weights),
        })
    }

    /// Number of point correspondences in this view.
    #[inline]
    pub fn len(&self) -> usize {
        self.points_3d.len()
    }

    /// Returns true if this view has no correspondences.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points_3d.is_empty()
    }

    /// Get the weight for a specific point index.
    ///
    /// Returns 1.0 if no weights were provided.
    #[inline]
    pub fn weight(&self, idx: usize) -> f64 {
        self.weights
            .as_ref()
            .and_then(|w| w.get(idx))
            .copied()
            .unwrap_or(1.0)
    }

    /// Iterate over (3D point, 2D point) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Pt3, &Pt2)> {
        self.points_3d.iter().zip(self.points_2d.iter())
    }

    /// Iterate over (3D point, 2D point, weight) tuples.
    pub fn iter_weighted(&self) -> impl Iterator<Item = (&Pt3, &Pt2, f64)> + '_ {
        self.points_3d
            .iter()
            .zip(self.points_2d.iter())
            .enumerate()
            .map(|(i, (p3, p2))| (p3, p2, self.weight(i)))
    }
}

/// Summary statistics for reprojection errors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReprojectionStats {
    /// Mean reprojection error in pixels.
    pub mean: f64,
    /// Root mean square error in pixels.
    pub rms: f64,
    /// Maximum reprojection error in pixels.
    pub max: f64,
    /// Number of points evaluated.
    pub count: usize,
}

impl ReprojectionStats {
    /// Compute statistics from a collection of errors.
    pub fn from_errors(errors: &[f64]) -> Self {
        if errors.is_empty() {
            return Self {
                mean: 0.0,
                rms: 0.0,
                max: 0.0,
                count: 0,
            };
        }

        let sum: f64 = errors.iter().sum();
        let sum_sq: f64 = errors.iter().map(|e| e * e).sum();
        let max = errors.iter().cloned().fold(0.0_f64, f64::max);
        let n = errors.len() as f64;

        Self {
            mean: sum / n,
            rms: (sum_sq / n).sqrt(),
            max,
            count: errors.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correspondence_view_creation() {
        let p3 = vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(1.0, 0.0, 0.0)];
        let p2 = vec![Pt2::new(320.0, 240.0), Pt2::new(400.0, 240.0)];

        let view = CorrespondenceView::new(p3.clone(), p2.clone()).unwrap();
        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());
        assert_eq!(view.weight(0), 1.0);
        assert_eq!(view.weight(1), 1.0);
    }

    #[test]
    fn correspondence_view_with_weights() {
        let p3 = vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(1.0, 0.0, 0.0)];
        let p2 = vec![Pt2::new(320.0, 240.0), Pt2::new(400.0, 240.0)];
        let w = vec![0.5, 2.0];

        let view = CorrespondenceView::new_with_weights(p3, p2, w).unwrap();
        assert_eq!(view.weight(0), 0.5);
        assert_eq!(view.weight(1), 2.0);
    }

    #[test]
    fn correspondence_view_rejects_mismatch() {
        let p3 = vec![Pt3::new(0.0, 0.0, 0.0)];
        let p2 = vec![Pt2::new(320.0, 240.0), Pt2::new(400.0, 240.0)];

        assert!(CorrespondenceView::new(p3, p2).is_err());
    }

    #[test]
    fn correspondence_view_rejects_negative_weights() {
        let p3 = vec![Pt3::new(0.0, 0.0, 0.0)];
        let p2 = vec![Pt2::new(320.0, 240.0)];
        let w = vec![-1.0];

        assert!(CorrespondenceView::new_with_weights(p3, p2, w).is_err());
    }

    #[test]
    fn reprojection_stats_empty() {
        let stats = ReprojectionStats::from_errors(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn reprojection_stats_computation() {
        let errors = vec![1.0, 2.0, 3.0];
        let stats = ReprojectionStats::from_errors(&errors);

        assert_eq!(stats.count, 3);
        assert!((stats.mean - 2.0).abs() < 1e-10);
        assert!((stats.rms - (14.0_f64 / 3.0).sqrt()).abs() < 1e-10);
        assert!((stats.max - 3.0).abs() < 1e-10);
    }

    #[test]
    fn correspondence_view_serde_roundtrip() {
        let p3 = vec![Pt3::new(0.0, 0.0, 0.0)];
        let p2 = vec![Pt2::new(320.0, 240.0)];
        let view = CorrespondenceView::new(p3, p2).unwrap();

        let json = serde_json::to_string(&view).unwrap();
        let restored: CorrespondenceView = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.len(), view.len());
    }
}
