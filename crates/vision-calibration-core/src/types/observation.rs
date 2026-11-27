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

    /// The isotropic covariance of a pixel with standard deviation `sigma`
    /// px on each axis, uncorrelated: `[[σ², 0], [0, σ²]]`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] unless `sigma` is finite and positive.
    ///
    /// # Example
    ///
    /// ```
    /// use vision_calibration_core::PointCovariance;
    ///
    /// let cov = PointCovariance::isotropic(0.5).unwrap();
    /// assert_eq!((cov.xx(), cov.xy(), cov.yy()), (0.25, 0.0, 0.25));
    /// assert!(PointCovariance::isotropic(0.0).is_err());
    /// ```
    pub fn isotropic(sigma: f64) -> Result<Self, Error> {
        if !(sigma.is_finite() && sigma > 0.0) {
            return Err(Error::invalid_input(format!(
                "a pixel noise σ must be finite and positive, got {sigma} px"
            )));
        }
        Self::new(sigma * sigma, 0.0, sigma * sigma)
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

/// The blur measured at one observed image point: the Gaussian point-spread
/// function (PSF) that best explains the blurred image around it, and the
/// covariance of that estimate.
///
/// [`psf_px`](Self::psf_px) is the PSF's covariance matrix `Σ = [[xx, xy],
/// [xy, yy]]`, px², stored as `[xx, xy, yy]` (the layout of
/// [`PointCovariance`]'s entries). Its principal axes are the directions of
/// least and most blur, and [`sigma_px`](Self::sigma_px) `= √(tr Σ / 2)` is
/// the PSF's width. `Σ` is the blur as the image shows it, **including the
/// pixel's area integration**: for a Gaussian optical PSF of covariance `Σ_o`
/// seen by square pixels, `Σ ≈ Σ_o + I/12`, so a model of the optics adds
/// `I/12` to its prediction before comparing it with `Σ`.
///
/// [`cov`](Self::cov) is the 3×3 covariance of `(xx, xy, yy)`, px⁴. What it
/// covers is the detector's statement: chessboard and PuzzleBoard corners
/// report one that allows for residuals correlated between neighbouring
/// pixels, and a combination of `Σ` the print around the corner constrains
/// weakly shows as a large variance.
///
/// Constructed only through [`PointBlur::new`] (or deserialization), which
/// checks that it is finite, that `Σ` is positive definite and that `cov` is
/// symmetric and positive semi-definite, so every quantity derived from the
/// PSF (its width, an axis) has a non-negative variance. A `cov` of zeros
/// states no precision.
///
/// # Example
///
/// ```
/// use vision_calibration_core::PointBlur;
///
/// // A PSF of 1.5 px along u and 1.2 px along v, uncorrelated.
/// let cov = [[1e-3, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-3]];
/// let blur = PointBlur::new([2.25, 0.0, 1.44], cov).unwrap();
/// assert!((blur.sigma_px() - 1.3583).abs() < 1e-4);
/// assert!(PointBlur::new([1.0, 2.0, 1.0], cov).is_err()); // Σ not positive definite
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[serde(try_from = "PointBlurRaw")]
pub struct PointBlur {
    psf_px: [f64; 3],
    cov: [[f64; 3]; 3],
}

/// The wire form of [`PointBlur`], validated on conversion; it also
/// describes its JSON Schema.
#[derive(Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schemars", schemars(rename = "PointBlur"))]
struct PointBlurRaw {
    /// The PSF covariance `[xx, xy, yy]`, px², including the pixel's area
    /// integration.
    psf_px: [f64; 3],
    /// The covariance of `psf_px`, px⁴, row-major.
    cov: [[f64; 3]; 3],
}

impl TryFrom<PointBlurRaw> for PointBlur {
    type Error = Error;

    fn try_from(raw: PointBlurRaw) -> Result<Self, Self::Error> {
        Self::new(raw.psf_px, raw.cov)
    }
}

/// The relative tolerance of [`PointBlur::new`]'s symmetry and
/// semi-definiteness checks, of the largest variance: rounding in a
/// propagated covariance stays far below it.
const BLUR_COV_TOLERANCE: f64 = 1e-9;

impl PointBlur {
    /// The blur of PSF covariance `psf_px` (`[xx, xy, yy]`, px²) with the
    /// covariance `cov` of those three elements (px⁴, row-major).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] unless every entry is finite, `Σ` is
    /// positive definite (`xx > 0`, `yy > 0`, `xy² < xx·yy`), and `cov` is
    /// symmetric and positive semi-definite (both to `1e-9` of its largest
    /// variance).
    pub fn new(psf_px: [f64; 3], cov: [[f64; 3]; 3]) -> Result<Self, Error> {
        let [xx, xy, yy] = psf_px;
        let finite = psf_px
            .iter()
            .chain(cov.iter().flatten())
            .all(|v| v.is_finite());
        if !finite || !(xx > 0.0 && yy > 0.0 && xx * yy - xy * xy > 0.0) {
            return Err(Error::invalid_input(format!(
                "a point blur's PSF covariance must be finite and positive definite, got \
                 [[{xx}, {xy}], [{xy}, {yy}]] px²"
            )));
        }
        let scale = (0..3).map(|i| cov[i][i]).fold(0.0, f64::max);
        let tolerance = BLUR_COV_TOLERANCE * scale;
        let symmetric = (0..3).all(|r| (0..r).all(|k| (cov[r][k] - cov[k][r]).abs() <= tolerance));
        let sym = nalgebra::Matrix3::from_fn(|r, k| 0.5 * (cov[r][k] + cov[k][r]));
        let semidefinite = sym.symmetric_eigenvalues().min() >= -tolerance;
        if !(symmetric && semidefinite) {
            return Err(Error::invalid_input(format!(
                "a point blur's covariance must be symmetric and positive semi-definite, \
                 got {cov:?} px⁴"
            )));
        }
        Ok(Self { psf_px, cov })
    }

    /// The PSF covariance `[xx, xy, yy]`, px², including the pixel's area
    /// integration.
    pub fn psf_px(&self) -> [f64; 3] {
        self.psf_px
    }

    /// The covariance of [`psf_px`](Self::psf_px), px⁴, row-major.
    pub fn cov(&self) -> [[f64; 3]; 3] {
        self.cov
    }

    /// The PSF's width `√(tr Σ / 2)`, px: the root mean of its two
    /// principal variances.
    pub fn sigma_px(&self) -> f64 {
        (0.5 * (self.psf_px[0] + self.psf_px[2])).sqrt()
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
/// - `blur`: optionally, the blur measured at each image point
///   ([`PointBlur`]), `None` for a point without one
///
/// Within one calibration input either every view carries covariances or
/// none does. The blur is a measurement carried with the points: any point
/// may have one or not, and no calibration reads it.
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
    /// Per-point measured blur, one per point (`None` for a point without
    /// one); empty when no point has one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    blur: Vec<Option<PointBlur>>,
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
    /// The blur measured at each image point, one per point, `null` for a
    /// point without one; absent or empty when no point has one.
    #[serde(default)]
    blur: Vec<Option<PointBlur>>,
}

impl TryFrom<CorrespondenceViewRaw> for CorrespondenceView {
    type Error = Error;

    fn try_from(raw: CorrespondenceViewRaw) -> Result<Self, Self::Error> {
        let view = if raw.cov_px.is_empty() {
            Self::new(raw.points_3d, raw.points_2d)
        } else {
            Self::new_with_covariances(raw.points_3d, raw.points_2d, raw.cov_px)
        }?;
        view.with_blur(raw.blur)
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
            blur: Vec::new(),
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
            blur: Vec::new(),
        })
    }

    /// The same view with the blur measured at each point, one per point
    /// (`None` for a point without one). An empty list, or one without any
    /// blur, leaves the view without blur.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidInput`] if a non-empty `blur` does not have
    /// one entry per point.
    pub fn with_blur(mut self, blur: Vec<Option<PointBlur>>) -> Result<Self, Error> {
        if !blur.is_empty() && blur.len() != self.len() {
            return Err(Error::invalid_input(format!(
                "blur count must match point count: {} vs {}",
                blur.len(),
                self.len()
            )));
        }
        self.blur = if blur.iter().any(Option::is_some) {
            blur
        } else {
            Vec::new()
        };
        Ok(self)
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

    /// The blur measured at each point, one per point (`None` for a point
    /// without one); `None` when no point has one.
    pub fn blurs(&self) -> Option<&[Option<PointBlur>]> {
        (!self.blur.is_empty()).then_some(self.blur.as_slice())
    }

    /// The blur measured at point `idx`; `None` when it has none (or `idx`
    /// is out of range).
    #[inline]
    pub fn blur(&self, idx: usize) -> Option<PointBlur> {
        self.blur.get(idx).copied().flatten()
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
///
/// The percentiles (`median`, `p95`) interpolate linearly between order
/// statistics of the ascending-sorted errors, the default of numpy and
/// pandas: the `q`-quantile of `n` errors lies at fractional rank
/// `q * (n - 1)`, so the median of an even count is the mean of the middle
/// two. They are present when the statistics were computed from the errors
/// ([`from_errors`](Self::from_errors)), and absent when built from
/// aggregates ([`from_summary`](Self::from_summary)).
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(schemars::JsonSchema))]
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
    /// Median reprojection error in pixels (the 50th percentile, linearly
    /// interpolated); `None` when built from aggregates. `0.0` for no
    /// errors, as the other statistics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median: Option<f64>,
    /// 95th percentile of the reprojection error in pixels, linearly
    /// interpolated; `None` when built from aggregates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p95: Option<f64>,
}

impl ReprojectionStats {
    /// Build statistics from already aggregated values; the percentiles are
    /// absent.
    pub fn from_summary(mean: f64, rms: f64, max: f64, count: usize) -> Self {
        Self {
            mean,
            rms,
            max,
            count,
            median: None,
            p95: None,
        }
    }

    /// Compute statistics from a collection of errors, the percentiles
    /// included. No errors give all-zero statistics.
    pub fn from_errors(errors: &[f64]) -> Self {
        if errors.is_empty() {
            return Self {
                mean: 0.0,
                rms: 0.0,
                max: 0.0,
                count: 0,
                median: Some(0.0),
                p95: Some(0.0),
            };
        }

        let sum: f64 = errors.iter().sum();
        let sum_sq: f64 = errors.iter().map(|e| e * e).sum();
        let max = errors.iter().cloned().fold(0.0_f64, f64::max);
        let n = errors.len() as f64;
        let mut sorted = errors.to_vec();
        sorted.sort_by(f64::total_cmp);

        Self {
            mean: sum / n,
            rms: (sum_sq / n).sqrt(),
            max,
            count: errors.len(),
            median: Some(percentile_linear(&sorted, 0.5)),
            p95: Some(percentile_linear(&sorted, 0.95)),
        }
    }
}

/// The `q`-quantile (`q` in `[0, 1]`) of an ascending-sorted, non-empty
/// slice, interpolated linearly between the order statistics around the
/// fractional rank `q * (n - 1)`.
fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let rank = q * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
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

    const BLUR_COV: [[f64; 3]; 3] = [[4e-3, 1e-4, -2e-4], [1e-4, 1e-3, 0.0], [-2e-4, 0.0, 3e-3]];

    #[test]
    fn point_blur_validates() {
        let blur = PointBlur::new([2.25, 0.3, 1.75], BLUR_COV).unwrap();
        assert_eq!(blur.psf_px(), [2.25, 0.3, 1.75]);
        assert_eq!(blur.cov(), BLUR_COV);
        assert!((blur.sigma_px() - 2.0_f64.sqrt()).abs() < 1e-15);
        for psf in [
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0], // singular
            [1.0, 0.0, -1.0],
            [f64::NAN, 0.0, 1.0],
        ] {
            assert!(PointBlur::new(psf, BLUR_COV).is_err(), "{psf:?} accepted");
        }
        let mut asymmetric = BLUR_COV;
        asymmetric[0][1] += 1e-6;
        let mut negative = BLUR_COV;
        negative[1][1] = -1e-6;
        let mut infinite = BLUR_COV;
        infinite[2][0] = f64::INFINITY;
        // Non-negative variances, but a variance of the width
        // `(xx + 2·xy + yy)/4` below zero: not a covariance.
        let indefinite = [[1e-3, 0.0, -1.5e-3], [0.0, 1e-4, 0.0], [-1.5e-3, 0.0, 1e-3]];
        for cov in [asymmetric, negative, infinite, indefinite] {
            assert!(
                PointBlur::new([1.0, 0.0, 1.0], cov).is_err(),
                "{cov:?} accepted"
            );
        }
        // Rounding-level asymmetry is accepted.
        let mut rounded = BLUR_COV;
        rounded[0][1] += 1e-18;
        assert!(PointBlur::new([1.0, 0.0, 1.0], rounded).is_ok());
        // A singular (rank-one) covariance and one of zeros are valid, and
        // so is rounding just below semi-definite.
        let v = [1.0, 0.2, 1.0];
        let rank_one: [[f64; 3]; 3] =
            std::array::from_fn(|r| std::array::from_fn(|k| 1e-3 * v[r] * v[k]));
        assert!(PointBlur::new([1.0, 0.0, 1.0], rank_one).is_ok());
        assert!(PointBlur::new([1.0, 0.0, 1.0], [[0.0; 3]; 3]).is_ok());
        let mut rounded_rank_one = rank_one;
        rounded_rank_one[1][1] -= 1e-18;
        assert!(PointBlur::new([1.0, 0.0, 1.0], rounded_rank_one).is_ok());
    }

    #[test]
    fn correspondence_view_carries_blur() {
        let (p3, p2) = points(3);
        let blur = PointBlur::new([2.25, 0.0, 2.25], BLUR_COV).unwrap();
        let view = CorrespondenceView::new(p3.clone(), p2.clone())
            .unwrap()
            .with_blur(vec![Some(blur), None, Some(blur)])
            .unwrap();
        assert_eq!(view.blur(0), Some(blur));
        assert_eq!(view.blur(1), None);
        assert_eq!(view.blur(3), None);
        assert_eq!(view.blurs().map(<[_]>::len), Some(3));

        // A list without any blur, or an empty one, is no blur.
        let none = CorrespondenceView::new(p3.clone(), p2.clone())
            .unwrap()
            .with_blur(vec![None; 3])
            .unwrap();
        assert!(none.blurs().is_none());
        assert!(
            CorrespondenceView::new(p3.clone(), p2.clone())
                .unwrap()
                .with_blur(Vec::new())
                .unwrap()
                .blurs()
                .is_none()
        );
        let err = CorrespondenceView::new(p3, p2)
            .unwrap()
            .with_blur(vec![Some(blur)])
            .unwrap_err();
        assert!(err.to_string().contains("blur count"), "{err}");
    }

    #[test]
    fn correspondence_view_blur_json_roundtrip() {
        let (p3, p2) = points(2);
        let blur = PointBlur::new([2.25, -0.1, 1.5], BLUR_COV).unwrap();
        let c = PointCovariance::new(0.09, 0.0, 0.04).unwrap();
        let view = CorrespondenceView::new_with_covariances(p3.clone(), p2.clone(), vec![c; 2])
            .unwrap()
            .with_blur(vec![None, Some(blur)])
            .unwrap();
        let json = serde_json::to_string(&view).unwrap();
        assert!(
            json.contains(r#""blur":[null,{"psf_px":[2.25,-0.1,1.5],"cov":"#),
            "{json}"
        );
        let restored: CorrespondenceView = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.blurs(), view.blurs());
        assert_eq!(restored.covariances(), view.covariances());

        // Without blur: no key, and a view written before the field reads.
        let plain = CorrespondenceView::new(p3, p2).unwrap();
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("blur"), "{json}");
        let old = r#"{"points_3d": [[0.0, 0.0, 0.0]], "points_2d": [[1.0, 2.0]]}"#;
        assert!(
            serde_json::from_str::<CorrespondenceView>(old)
                .unwrap()
                .blurs()
                .is_none()
        );

        // Validated on read: a count mismatch and an invalid blur fail.
        let short = r#"{"points_3d": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "points_2d": [[1.0, 2.0], [3.0, 2.0]],
            "blur": [{"psf_px": [1.0, 0.0, 1.0], "cov": [[0,0,0],[0,0,0],[0,0,0]]}]}"#;
        let err = serde_json::from_str::<CorrespondenceView>(short).unwrap_err();
        assert!(err.to_string().contains("blur count"), "{err}");
        let bad = r#"{"points_3d": [[0.0, 0.0, 0.0]], "points_2d": [[1.0, 2.0]],
            "blur": [{"psf_px": [1.0, 2.0, 1.0], "cov": [[0,0,0],[0,0,0],[0,0,0]]}]}"#;
        let err = serde_json::from_str::<CorrespondenceView>(bad).unwrap_err();
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

    #[test]
    fn percentiles_interpolate_between_order_statistics() {
        // Sorted: 1, 2, 3, 4, 10. Median at rank 2; p95 at rank 3.8,
        // 4 + 0.8 * (10 - 4).
        let stats = ReprojectionStats::from_errors(&[4.0, 1.0, 10.0, 3.0, 2.0]);
        assert_eq!(stats.median, Some(3.0));
        assert!((stats.p95.unwrap() - 8.8).abs() < 1e-12);
        assert_eq!((stats.max, stats.count), (10.0, 5));
        // An even count: the mean of the middle two.
        let even = ReprojectionStats::from_errors(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(even.median, Some(2.5));
        assert!((even.p95.unwrap() - 3.85).abs() < 1e-12);
        assert_eq!(ReprojectionStats::from_errors(&[0.7]).p95, Some(0.7));
        // From aggregates, no percentiles, and none written.
        let summary = ReprojectionStats::from_summary(1.0, 1.1, 3.0, 10);
        assert_eq!(summary.median, None);
        let json = serde_json::to_value(summary).unwrap();
        assert!(json.get("median").is_none() && json.get("p95").is_none());
        let back: ReprojectionStats = serde_json::from_value(
            serde_json::json!({"mean": 1.0, "rms": 1.1, "max": 3.0, "count": 10}),
        )
        .unwrap();
        assert_eq!(back.p95, None);
        let round: ReprojectionStats =
            serde_json::from_value(serde_json::to_value(stats).unwrap()).unwrap();
        assert_eq!((round.median, round.p95), (stats.median, stats.p95));
    }
}
