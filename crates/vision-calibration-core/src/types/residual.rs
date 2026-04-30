//! Per-feature reprojection residual records.
//!
//! Records and aggregate histograms produced by every calibration `*Export` so
//! downstream consumers (the planned Tauri/React diagnose UI; ad-hoc analyses;
//! custom viewers) can drill into per-(view, camera, feature) errors without
//! recomputing geometry.
//!
//! See [ADR 0012](https://github.com/VitalyVorobyev/calibration-rs/blob/main/docs/adrs/0012-per-feature-reprojection-residuals.md)
//! for the schema and the indexing convention. Helpers that produce these
//! records from a calibrated camera + dataset live alongside this module
//! (`vision_calibration_core::compute_planar_target_residuals`,
//! `compute_rig_target_residuals`, `build_feature_histogram`) and in
//! `vision-calibration-optim` for laser flavors.
//!
//! # Indexing
//!
//! `target` and `laser` Vecs are pose-major. Every record carries an explicit
//! `(pose, camera, feature)` triple. Iteration order: outer = view, inner =
//! camera (camera-`None` slots skipped), innermost = feature index in that
//! view's `points_3d`. For single-camera problem types `camera` is always 0.
//!
//! # `Option<T>` semantics
//!
//! - `projected_px = None` and `error_px = None`: projection diverged for this
//!   feature (point behind camera or distortion fixed-point failed).
//! - `residual_m = None` / `residual_px = None`: laser ray did not intersect
//!   the target plane.
//! - `projected_line_px = None`: the camera could not synthesize endpoints in
//!   image space.
//!
//! # Histogram bucket edges
//!
//! Reprojection histograms use fixed bucket edges `[1.0, 2.0, 5.0, 10.0]`
//! pixels — five buckets `[<=1, <=2, <=5, <=10, >10]`. This matches the
//! existing `RigHandeyeLaserlinePerCamStats::reproj_histogram_px` so per-camera
//! aggregates are comparable across the workspace.

use serde::{Deserialize, Serialize};

/// Bucket edges (in pixels) for [`FeatureResidualHistogram`]: `[1.0, 2.0, 5.0, 10.0]`.
pub const REPROJECTION_HISTOGRAM_EDGES_PX: [f64; 4] = [1.0, 2.0, 5.0, 10.0];

/// Reprojection record for a single target feature in a single view.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TargetFeatureResidual {
    /// Pose / view index in the input dataset.
    pub pose: usize,
    /// Camera index. `0` for single-camera problem types.
    pub camera: usize,
    /// Feature index within the view's `points_3d`.
    pub feature: usize,
    /// 3D point in target / world frame (meters).
    pub target_xyz_m: [f64; 3],
    /// Observed pixel coordinate.
    pub observed_px: [f64; 2],
    /// Projected pixel using the calibrated camera + recovered pose.
    /// `None` when projection diverges (point behind camera, distortion failure).
    pub projected_px: Option<[f64; 2]>,
    /// Euclidean pixel distance `|projected - observed|`.
    /// `None` iff `projected_px` is `None`.
    pub error_px: Option<f64>,
}

/// Reprojection record for a single laser pixel in a single view.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LaserFeatureResidual {
    /// Pose / view index in the input dataset.
    pub pose: usize,
    /// Camera index. `0` for single-camera problem types.
    pub camera: usize,
    /// Pixel index within the view's laser pixel list.
    pub feature: usize,
    /// Observed laser pixel coordinate.
    pub observed_px: [f64; 2],
    /// Point-to-plane distance in meters between the back-projected ray and
    /// the calibrated laser plane. `None` if the ray does not intersect.
    pub residual_m: Option<f64>,
    /// Pixel-domain residual: distance from `observed_px` to the projected
    /// laser line in undistorted pixel space. `None` if unavailable.
    pub residual_px: Option<f64>,
    /// Two endpoints `[[x0, y0], [x1, y1]]` of the projected laser line in
    /// image space. Useful for 2D overlays. `None` if the line cannot be
    /// synthesized in this view.
    pub projected_line_px: Option<[[f64; 2]; 2]>,
}

/// Aggregate residual histogram for a (set of) reprojection error samples.
///
/// Buckets are fixed at `[<=1, <=2, <=5, <=10, >10]` pixels — see
/// [`REPROJECTION_HISTOGRAM_EDGES_PX`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureResidualHistogram {
    /// Bucket edges in pixels: `[1.0, 2.0, 5.0, 10.0]`.
    pub bucket_edges_px: [f64; 4],
    /// Counts in each bucket: `[<=1, <=2, <=5, <=10, >10]`.
    /// `counts.iter().sum() == count`.
    pub counts: [usize; 5],
    /// Total number of error samples included in the histogram.
    pub count: usize,
    /// Mean error (pixels). `0.0` when `count == 0`.
    pub mean: f64,
    /// Maximum error (pixels). `0.0` when `count == 0`.
    pub max: f64,
}

impl Default for FeatureResidualHistogram {
    fn default() -> Self {
        Self {
            bucket_edges_px: REPROJECTION_HISTOGRAM_EDGES_PX,
            counts: [0; 5],
            count: 0,
            mean: 0.0,
            max: 0.0,
        }
    }
}

/// Container bundled into every `*Export` carrying the per-feature drill-down
/// data the diagnose UI consumes.
///
/// Empty `Vec`s mean "this problem type does not produce observations of that
/// flavor" (e.g., `PlanarIntrinsicsExport.per_feature_residuals.laser` is
/// always empty). The histogram fields are `Option`: `None` means the problem
/// type chose not to produce a per-camera aggregate; `Some(vec)` length must
/// match `num_cameras`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PerFeatureResiduals {
    /// Per-target-corner reprojection records, pose-major.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target: Vec<TargetFeatureResidual>,
    /// Per-laser-pixel residual records, pose-major.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub laser: Vec<LaserFeatureResidual>,
    /// Per-camera target reprojection histogram (length = `num_cameras`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_hist_per_camera: Option<Vec<FeatureResidualHistogram>>,
    /// Per-camera laser pixel-distance histogram (length = `num_cameras`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laser_hist_per_camera: Option<Vec<FeatureResidualHistogram>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_feature_residual_default() {
        let r = TargetFeatureResidual::default();
        assert_eq!(r.pose, 0);
        assert_eq!(r.camera, 0);
        assert_eq!(r.feature, 0);
        assert_eq!(r.target_xyz_m, [0.0; 3]);
        assert_eq!(r.observed_px, [0.0; 2]);
        assert!(r.projected_px.is_none());
        assert!(r.error_px.is_none());
    }

    #[test]
    fn laser_feature_residual_default() {
        let r = LaserFeatureResidual::default();
        assert_eq!(r.pose, 0);
        assert_eq!(r.camera, 0);
        assert!(r.residual_m.is_none());
        assert!(r.residual_px.is_none());
        assert!(r.projected_line_px.is_none());
    }

    #[test]
    fn histogram_default_uses_reprojection_edges() {
        let h = FeatureResidualHistogram::default();
        assert_eq!(h.bucket_edges_px, [1.0, 2.0, 5.0, 10.0]);
        assert_eq!(h.bucket_edges_px, REPROJECTION_HISTOGRAM_EDGES_PX);
        assert_eq!(h.counts, [0; 5]);
        assert_eq!(h.count, 0);
        assert_eq!(h.mean, 0.0);
        assert_eq!(h.max, 0.0);
    }

    #[test]
    fn per_feature_residuals_default_is_empty() {
        let p = PerFeatureResiduals::default();
        assert!(p.target.is_empty());
        assert!(p.laser.is_empty());
        assert!(p.target_hist_per_camera.is_none());
        assert!(p.laser_hist_per_camera.is_none());
    }

    #[test]
    fn empty_per_feature_residuals_serializes_to_empty_object() {
        // Empty Vecs and Option::None should be skipped — the JSON for a
        // freshly-defaulted container is `{}`. This keeps payloads tight.
        let p = PerFeatureResiduals::default();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "{}");

        let restored: PerFeatureResiduals = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, p);
    }

    #[test]
    fn populated_per_feature_residuals_roundtrip() {
        let p = PerFeatureResiduals {
            target: vec![TargetFeatureResidual {
                pose: 3,
                camera: 1,
                feature: 17,
                target_xyz_m: [0.1, 0.2, 0.0],
                observed_px: [310.5, 245.0],
                projected_px: Some([310.7, 244.6]),
                error_px: Some(0.4472135954999579),
            }],
            laser: vec![LaserFeatureResidual {
                pose: 5,
                camera: 0,
                feature: 9,
                observed_px: [400.0, 200.0],
                residual_m: Some(0.0001),
                residual_px: Some(0.5),
                projected_line_px: Some([[100.0, 200.0], [700.0, 200.5]]),
            }],
            target_hist_per_camera: Some(vec![FeatureResidualHistogram {
                bucket_edges_px: REPROJECTION_HISTOGRAM_EDGES_PX,
                counts: [120, 30, 5, 0, 0],
                count: 155,
                mean: 0.55,
                max: 3.2,
            }]),
            laser_hist_per_camera: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        let restored: PerFeatureResiduals = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, p);
    }

    #[test]
    fn target_residual_with_diverged_projection_roundtrip() {
        // Critical: None must roundtrip cleanly so the UI can distinguish
        // "projection diverged" from "error happens to be exactly zero".
        let r = TargetFeatureResidual {
            pose: 0,
            camera: 0,
            feature: 0,
            target_xyz_m: [0.0; 3],
            observed_px: [320.0, 240.0],
            projected_px: None,
            error_px: None,
        };
        let json = serde_json::to_string(&r).unwrap();
        assert!(json.contains("\"projected_px\":null"));
        assert!(json.contains("\"error_px\":null"));
        let restored: TargetFeatureResidual = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, r);
    }

    #[test]
    fn histogram_roundtrip_preserves_edges() {
        let h = FeatureResidualHistogram {
            bucket_edges_px: REPROJECTION_HISTOGRAM_EDGES_PX,
            counts: [10, 5, 2, 1, 0],
            count: 18,
            mean: 0.85,
            max: 6.4,
        };
        let json = serde_json::to_string(&h).unwrap();
        let restored: FeatureResidualHistogram = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, h);
    }

    #[test]
    fn missing_optional_fields_deserialize_to_none() {
        // Producers that omit the histograms entirely must round-trip.
        let json = r#"{"target":[],"laser":[]}"#;
        let p: PerFeatureResiduals = serde_json::from_str(json).unwrap();
        assert!(p.target.is_empty());
        assert!(p.laser.is_empty());
        assert!(p.target_hist_per_camera.is_none());
        assert!(p.laser_hist_per_camera.is_none());
    }
}
