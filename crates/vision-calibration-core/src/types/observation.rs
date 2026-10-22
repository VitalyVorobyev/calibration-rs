//! Observation types for calibration data.
//!
//! This module provides canonical data structures for storing 2D-3D point
//! correspondences used throughout the calibration pipeline.

use crate::{Error, Pt2, Pt3};
use serde::{Deserialize, Serialize};

/// The covariance of one observed image point, px²: the symmetric matrix
/// `[[xx, xy], [xy, yy]]`.
///
/// It states the point's measurement noise. A calibration whose views carry
/// covariances weighs each reprojection residual by its inverse (it
/// *whitens* it): an uncertain point counts less, and an elongated one is
/// trusted more across its long axis than along it. A uniform weight `w` on
/// every point corresponds to the covariance `I / w`.
///
/// Constructed only through [`PointCovariance::new`] (or deserialization),
/// which checks that it is finite and positive definite.
///
/// # Example
///
/// ```
/// use vision_calibration_core::PointCovariance;
///
/// // σ = 0.3 px along u, 0.1 px along v, uncorrelated.
/// let cov = PointCovariance::new(0.09, 0.0, 0.01).unwrap();
/// assert_eq!(cov.yy(), 0.01);
/// assert!(PointCovariance::new(1.0, 2.0, 1.0).is_err()); // not positive definite
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(try_from = "PointCovarianceRaw")]
pub struct PointCovariance {
    xx: f64,
    xy: f64,
    yy: f64,
}

/// The wire form of [`PointCovariance`], validated on conversion; it also
/// describes its JSON Schema.
#[derive(Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars", schemars(rename = "PointCovariance"))]
struct PointCovarianceRaw {
    /// Variance along the image `u` axis, px².
    xx: f64,
    /// Covariance of `u` and `v`, px².
    xy: f64,
    /// Variance along the image `v` axis, px².
    yy: f64,
}

impl TryFrom<PointCovarianceRaw> for PointCovariance {
    type Error = Error;

    fn try_from(raw: PointCovarianceRaw) -> Result<Self, Self::Error> {
        Self::new(raw.xx, raw.xy, raw.yy)
    }
}

impl PointCovariance {
    /// The covariance `[[xx, xy], [xy, yy]]`, px².
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] unless every entry is finite and the
    /// matrix is positive definite (`xx > 0`, `yy > 0`, `xy² < xx·yy`) with
    /// a finite inverse.
    pub fn new(xx: f64, xy: f64, yy: f64) -> Result<Self, Error> {
        let cov = Self { xx, xy, yy };
        let finite = [xx, xy, yy].iter().all(|v| v.is_finite());
        let det = xx * yy - xy * xy;
        let positive = xx > 0.0 && yy > 0.0 && det.is_finite() && det > 0.0;
        if !(finite && positive && cov.sqrt_information().iter().all(|v| v.is_finite())) {
            return Err(Error::invalid_input(format!(
                "a point covariance must be finite and positive definite, got \
                 [[{xx}, {xy}], [{xy}, {yy}]] px²"
            )));
        }
        Ok(cov)
    }

    /// Variance along the image `u` axis, px².
    pub fn xx(&self) -> f64 {
        self.xx
    }

    /// Covariance of the `u` and `v` axes, px².
    pub fn xy(&self) -> f64 {
        self.xy
    }

    /// Variance along the image `v` axis, px².
    pub fn yy(&self) -> f64 {
        self.yy
    }

    /// The square-root information `[l₀₀, l₀₁, l₁₁]`: the upper-triangular
    /// `L = [[l₀₀, l₀₁], [0, l₁₁]]` with `LᵀL = Σ⁻¹`, px⁻¹. `L e` is a pixel
    /// error `e` whitened to unit covariance.
    pub fn sqrt_information(&self) -> [f64; 3] {
        // Σ⁻¹ = [[yy, −xy], [−xy, xx]] / det; its upper Cholesky factor.
        let det = self.xx * self.yy - self.xy * self.xy;
        let sqrt_yy = self.yy.sqrt();
        let sqrt_det = det.sqrt();
        [
            sqrt_yy / sqrt_det,
            -self.xy / (sqrt_yy * sqrt_det),
            1.0 / sqrt_yy,
        ]
    }

    /// The whitened length `‖L e‖ = √(eᵀ Σ⁻¹ e)` of a pixel error `e`: its
    /// size in units of the point's standard deviation along it.
    pub fn whitened_norm(&self, e: [f64; 2]) -> f64 {
        let [l00, l01, l11] = self.sqrt_information();
        (l00 * e[0] + l01 * e[1]).hypot(l11 * e[1])
    }
}

/// A single view containing 2D-3D point correspondences.
///
/// This is the canonical observation type used across planar intrinsics,
/// rig extrinsics, hand-eye calibration, and other calibration problems.
///
/// # Fields
///
/// - `points_3d`: 3D points in world/target coordinates
/// - `points_2d`: Corresponding 2D pixel observations
/// - `cov_px`: optionally, the covariance of each image point
///   ([`PointCovariance`], px²)
///
/// Within one calibration input either every view carries covariances or
/// none does.
///
/// # Example
///
/// ```
/// use vision_calibration_core::{CorrespondenceView, PointCovariance, Pt2, Pt3};
///
/// let points_3d = vec![Pt3::new(0.0, 0.0, 0.0), Pt3::new(0.1, 0.0, 0.0)];
/// let points_2d = vec![Pt2::new(320.0, 240.0), Pt2::new(400.0, 240.0)];
/// let view = CorrespondenceView::new(points_3d.clone(), points_2d.clone()).unwrap();
/// assert_eq!(view.len(), 2);
/// assert!(view.covariances().is_none());
///
/// // With a stated covariance per point.
/// let cov = PointCovariance::new(0.04, 0.0, 0.04).unwrap();
/// let view =
///     CorrespondenceView::new_with_covariances(points_3d, points_2d, vec![cov; 2]).unwrap();
/// assert_eq!(view.covariance(1), Some(cov));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(try_from = "CorrespondenceViewRaw")]
pub struct CorrespondenceView {
    /// 3D points in world/target frame.
    #[cfg_attr(feature = "schemars", schemars(with = "Vec<[f64; 3]>"))]
    pub points_3d: Vec<Pt3>,
    /// Corresponding 2D pixel observations.
    #[cfg_attr(feature = "schemars", schemars(with = "Vec<[f64; 2]>"))]
    pub points_2d: Vec<Pt2>,
    /// Per-point covariance, px², one per point; `None` when not stated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    cov_px: Option<Vec<PointCovariance>>,
}

/// The wire form of [`CorrespondenceView`], validated on conversion; it
/// also describes the view's JSON Schema.
#[derive(Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars", schemars(rename = "CorrespondenceView"))]
struct CorrespondenceViewRaw {
    /// 3D points `[x, y, z]` in the target frame, metres.
    #[cfg_attr(feature = "schemars", schemars(with = "Vec<[f64; 3]>"))]
    points_3d: Vec<Pt3>,
    /// The matching image points `[u, v]`, pixels.
    #[cfg_attr(feature = "schemars", schemars(with = "Vec<[f64; 2]>"))]
    points_2d: Vec<Pt2>,
    /// The covariance of each image point, px², one per point; absent or
    /// empty when not stated. Within one calibration input either every
    /// view states them or none does.
    #[serde(default)]
    cov_px: Vec<PointCovariance>,
}

impl TryFrom<CorrespondenceViewRaw> for CorrespondenceView {
    type Error = Error;

    fn try_from(raw: CorrespondenceViewRaw) -> Result<Self, Self::Error> {
        if raw.cov_px.is_empty() {
            Self::new(raw.points_3d, raw.points_2d)
        } else {
            Self::new_with_covariances(raw.points_3d, raw.points_2d, raw.cov_px)
        }
    }
}

impl CorrespondenceView {
    /// Construct observations without per-point covariances.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the 3D and 2D point counts don't match.
    pub fn new(points_3d: Vec<Pt3>, points_2d: Vec<Pt2>) -> Result<Self, Error> {
        check_counts(points_3d.len(), points_2d.len())?;
        Ok(Self {
            points_3d,
            points_2d,
            cov_px: None,
        })
    }

    /// Construct observations with a covariance per point, px².
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if the 3D point, 2D point and
    /// covariance counts don't all match.
    pub fn new_with_covariances(
        points_3d: Vec<Pt3>,
        points_2d: Vec<Pt2>,
        cov_px: Vec<PointCovariance>,
    ) -> Result<Self, Error> {
        check_counts(points_3d.len(), points_2d.len())?;
        if cov_px.len() != points_3d.len() {
            return Err(Error::invalid_input(format!(
                "covariance count must match point count: {} vs {}",
                cov_px.len(),
                points_3d.len()
            )));
        }
        Ok(Self {
            points_3d,
            points_2d,
            cov_px: Some(cov_px),
        })
    }

    /// Return planar target coordinates `(x, y)` extracted from `points_3d`.
    ///
    /// This is commonly used for planar homography/intrinsics initialization.
    pub fn planar_points(&self) -> Vec<Pt2> {
        self.points_3d
            .iter()
            .map(|p3| Pt2::new(p3.x, p3.y))
            .collect()
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

    /// The per-point covariances, px², one per point; `None` when the view
    /// states none.
    pub fn covariances(&self) -> Option<&[PointCovariance]> {
        self.cov_px.as_deref()
    }

    /// The covariance of point `idx`, px²; `None` when the view states none
    /// (or `idx` is out of range).
    #[inline]
    pub fn covariance(&self, idx: usize) -> Option<PointCovariance> {
        self.cov_px.as_ref().and_then(|c| c.get(idx).copied())
    }

    /// Iterate over (3D point, 2D point) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Pt3, &Pt2)> {
        self.points_3d.iter().zip(self.points_2d.iter())
    }
}

fn check_counts(n3: usize, n2: usize) -> Result<(), Error> {
    if n3 != n2 {
        return Err(Error::invalid_input(format!(
            "3D / 2D point counts must match: {n3} vs {n2}"
        )));
    }
    Ok(())
}

/// Whether a calibration input's views state per-point covariances: `true`
/// when every view does, `false` when none does. Views without points are
/// not counted.
///
/// Whitened and pixel residuals are not one noise population, so an input
/// that mixes them is rejected rather than guessed at. Every problem type
/// checks its input with this before any work.
///
/// # Errors
///
/// [`Error::MixedCovariance`] when some views state covariances and others
/// do not; [`Error::InvalidInput`] when a view's covariance count does not
/// match its points.
pub fn stated_covariance<'a>(
    views: impl IntoIterator<Item = &'a CorrespondenceView>,
) -> Result<bool, Error> {
    let (mut with, mut without) = (0usize, 0usize);
    for view in views {
        if view.is_empty() {
            continue;
        }
        match view.covariances() {
            Some(cov) if cov.len() == view.len() => with += 1,
            Some(cov) => {
                return Err(Error::invalid_input(format!(
                    "covariance count must match point count: {} vs {}",
                    cov.len(),
                    view.len()
                )));
            }
            None => without += 1,
        }
    }
    if with > 0 && without > 0 {
        return Err(Error::MixedCovariance { with, without });
    }
    Ok(with > 0)
}

/// Summary statistics for reprojection errors.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[non_exhaustive]
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
    /// Build statistics from already aggregated values.
    pub fn from_summary(mean: f64, rms: f64, max: f64, count: usize) -> Self {
        Self {
            mean,
            rms,
            max,
            count,
        }
    }

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

    fn points(n: usize) -> (Vec<Pt3>, Vec<Pt2>) {
        let p3 = (0..n).map(|i| Pt3::new(i as f64, 0.0, 0.0)).collect();
        let p2 = (0..n).map(|i| Pt2::new(320.0 + i as f64, 240.0)).collect();
        (p3, p2)
    }

    #[test]
    fn correspondence_view_creation() {
        let (p3, p2) = points(2);
        let view = CorrespondenceView::new(p3, p2).unwrap();
        assert_eq!(view.len(), 2);
        assert!(!view.is_empty());
        assert!(view.covariances().is_none());
        assert_eq!(view.covariance(0), None);
    }

    #[test]
    fn correspondence_view_with_covariances() {
        let (p3, p2) = points(2);
        let a = PointCovariance::new(0.25, 0.0, 4.0).unwrap();
        let b = PointCovariance::new(1.0, 0.5, 1.0).unwrap();
        let view = CorrespondenceView::new_with_covariances(p3, p2, vec![a, b]).unwrap();
        assert_eq!(view.covariance(0), Some(a));
        assert_eq!(view.covariance(1), Some(b));
        assert_eq!(view.covariance(2), None);
    }

    #[test]
    fn correspondence_view_rejects_mismatch() {
        let (p3, _) = points(1);
        let (_, p2) = points(2);
        assert!(CorrespondenceView::new(p3, p2).is_err());
        let (p3, p2) = points(2);
        let c = PointCovariance::new(1.0, 0.0, 1.0).unwrap();
        assert!(CorrespondenceView::new_with_covariances(p3, p2, vec![c]).is_err());
    }

    #[test]
    fn point_covariance_rejects_invalid() {
        assert!(PointCovariance::new(1.0, 0.0, 1.0).is_ok());
        assert!(PointCovariance::new(1.0, 0.99, 1.0).is_ok());
        for (xx, xy, yy) in [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
            (1.0, 1.0, 1.0), // singular
            (1.0, 2.0, 1.0), // indefinite
            (f64::NAN, 0.0, 1.0),
            (1.0, f64::INFINITY, 1.0),
            (f64::INFINITY, 0.0, 1.0),
        ] {
            assert!(
                PointCovariance::new(xx, xy, yy).is_err(),
                "[[{xx}, {xy}], [{xy}, {yy}]] accepted"
            );
        }
    }

    /// `LᵀL = Σ⁻¹` with `L` upper triangular, and the whitened norm is the
    /// Mahalanobis distance.
    #[test]
    fn sqrt_information_factors_the_inverse() {
        let (xx, xy, yy) = (0.81, -0.3, 0.2);
        let cov = PointCovariance::new(xx, xy, yy).unwrap();
        let [a, b, c] = cov.sqrt_information();
        let det = xx * yy - xy * xy;
        // LᵀL = [[a², ab], [ab, b² + c²]].
        let inv = [yy / det, -xy / det, xx / det];
        assert!((a * a - inv[0]).abs() < 1e-12);
        assert!((a * b - inv[1]).abs() < 1e-12);
        assert!((b * b + c * c - inv[2]).abs() < 1e-12);
        let e = [0.7, -1.3];
        let mahalanobis: f64 =
            (inv[0] * e[0] * e[0] + 2.0 * inv[1] * e[0] * e[1] + inv[2] * e[1] * e[1]).sqrt();
        assert!((cov.whitened_norm(e) - mahalanobis).abs() < 1e-12);
        // Isotropic: L = I / σ.
        let iso = PointCovariance::new(0.25, 0.0, 0.25).unwrap();
        let [a, b, c] = iso.sqrt_information();
        assert_eq!((a, b, c), (2.0, 0.0, 2.0));
    }

    #[test]
    fn stated_covariance_is_all_or_none() {
        let (p3, p2) = points(3);
        let c = PointCovariance::new(1.0, 0.0, 1.0).unwrap();
        let plain = CorrespondenceView::new(p3.clone(), p2.clone()).unwrap();
        let stated = CorrespondenceView::new_with_covariances(p3, p2, vec![c; 3]).unwrap();
        let empty = CorrespondenceView::new(Vec::new(), Vec::new()).unwrap();
        assert!(!stated_covariance([&plain, &plain]).unwrap());
        assert!(stated_covariance([&stated, &stated, &empty]).unwrap());
        assert!(!stated_covariance(std::iter::empty()).unwrap());
        let err = stated_covariance([&stated, &plain, &stated]).unwrap_err();
        assert!(matches!(
            err,
            Error::MixedCovariance {
                with: 2,
                without: 1
            }
        ));
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
        let (p3, p2) = points(1);
        let view = CorrespondenceView::new(p3.clone(), p2.clone()).unwrap();
        let json = serde_json::to_string(&view).unwrap();
        assert!(
            !json.contains("cov_px"),
            "no covariance key when none: {json}"
        );
        let restored: CorrespondenceView = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.len(), view.len());
        assert!(restored.covariances().is_none());

        let c = PointCovariance::new(0.09, -0.02, 0.04).unwrap();
        let view = CorrespondenceView::new_with_covariances(p3, p2, vec![c]).unwrap();
        let json = serde_json::to_string(&view).unwrap();
        assert!(
            json.contains(r#""cov_px":[{"xx":0.09,"xy":-0.02,"yy":0.04}]"#),
            "{json}"
        );
        let restored: CorrespondenceView = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.covariances(), Some(&[c][..]));
    }

    #[test]
    fn correspondence_view_deserialize_empty_covariance_is_none() {
        let json = r#"{"points_3d": [[0.0, 0.0, 0.0]], "points_2d": [[1.0, 2.0]], "cov_px": []}"#;
        let view: CorrespondenceView = serde_json::from_str(json).unwrap();
        assert!(view.covariances().is_none());
    }

    #[test]
    fn correspondence_view_deserialize_rejects_short_covariance() {
        let json = r#"{
            "points_3d": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "points_2d": [[320.0, 240.0], [400.0, 240.0]],
            "cov_px": [{"xx": 1.0, "xy": 0.0, "yy": 1.0}]
        }"#;
        let err = serde_json::from_str::<CorrespondenceView>(json).unwrap_err();
        assert!(err.to_string().contains("covariance count"), "{err}");
    }

    #[test]
    fn correspondence_view_deserialize_rejects_indefinite_covariance() {
        let json = r#"{
            "points_3d": [[0.0, 0.0, 0.0]],
            "points_2d": [[320.0, 240.0]],
            "cov_px": [{"xx": 1.0, "xy": 2.0, "yy": 1.0}]
        }"#;
        let err = serde_json::from_str::<CorrespondenceView>(json).unwrap_err();
        assert!(err.to_string().contains("positive definite"), "{err}");
    }

    #[test]
    fn correspondence_view_deserialize_rejects_point_mismatch() {
        let json = r#"{
            "points_3d": [[0.0, 0.0, 0.0]],
            "points_2d": [[320.0, 240.0], [400.0, 240.0]]
        }"#;
        let err = serde_json::from_str::<CorrespondenceView>(json).unwrap_err();
        assert!(err.to_string().contains("must match"));
    }
}
