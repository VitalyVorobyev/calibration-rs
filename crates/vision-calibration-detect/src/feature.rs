//! Detector output: 2D image points paired with their 3D world points
//! in the target frame.

use serde::{Deserialize, Serialize};

#[cfg(feature = "schemars")]
use schemars::JsonSchema;

/// Single 2D-3D feature correspondence in canonical units.
///
/// `image_xy` is in source-image pixels (pre-ROI-crop if a ROI is
/// declared in the manifest, the converter adjusts as needed).
/// `world_xyz` is in metres, expressed in the calibration target's
/// own frame (target origin at `(0, 0, 0)`, target plane at `z = 0`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct Feature {
    /// Pixel coordinates `[x, y]` (column, row).
    pub image_xy: [f64; 2],
    /// World coordinates `[x, y, z]` in metres in the target frame.
    pub world_xyz: [f64; 3],
    /// Covariance of `image_xy` as `[xx, xy, yy]` in px², in the image
    /// frame of `image_xy`. Present only when the detector was asked for it
    /// (`ChessCornersConfig::covariance`) and measured one; `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cov_px: Option<[f64; 3]>,
    /// The blur the detector measured at this feature, with its covariance
    /// ([`FeatureBlur`]). Present when the detector was asked to measure one
    /// (`measure_blur` on the chessboard and PuzzleBoard configs) and the
    /// estimate at this corner was not declined; `None` otherwise.
    ///
    /// A measurement only: no calibration reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blur: Option<FeatureBlur>,
}

impl Feature {
    /// The correspondence of pixel `image_xy` to target point `world_xyz`,
    /// without a covariance or a blur.
    pub fn new(image_xy: [f64; 2], world_xyz: [f64; 3]) -> Self {
        Self {
            image_xy,
            world_xyz,
            cov_px: None,
            blur: None,
        }
    }

    /// The same feature with the pixel covariance `cov_px` (`[xx, xy, yy]`,
    /// px²), or none.
    pub fn with_cov_px(mut self, cov_px: Option<[f64; 3]>) -> Self {
        self.cov_px = cov_px;
        self
    }

    /// The same feature with the measured blur `blur`, or none.
    pub fn with_blur(mut self, blur: Option<FeatureBlur>) -> Self {
        self.blur = blur;
        self
    }
}

/// The blur a detector measured at one feature: the Gaussian point-spread
/// function (PSF) that best explains the blurred print around the feature,
/// and the covariance of that estimate.
///
/// **`psf_px`** is the PSF's covariance matrix `Σ = [[xx, xy], [xy, yy]]`
/// in px², stored as `[xx, xy, yy]` (the layout of
/// [`Feature::cov_px`]), in the image frame of `image_xy`. Its principal
/// axes are the directions of least and most blur, and `√(tr Σ / 2) =
/// √((xx + yy) / 2)` is the PSF's width σ in pixels.
///
/// `Σ` is the blur **as the image shows it, including the pixel's area
/// integration**: for a Gaussian optical PSF of covariance `Σ_o` seen by
/// square pixels, `Σ = Σ_o + I/12`, so a perfectly focused camera has
/// `Σ = I/12`. A model of the optics that predicts `Σ_o` adds `I/12` to its
/// prediction before comparing it with `psf_px`. The constant is not
/// subtracted here: it is what the detector measured, its covariance is
/// unchanged by a constant shift, and on a sharp feature `Σ − I/12` is
/// within the estimate's error of zero and need not be positive definite.
///
/// **`cov`** is the 3×3 covariance of `(xx, xy, yy)`, px⁴, row-major. How
/// far it can be trusted is the detector's statement. For chessboard and
/// PuzzleBoard corners it allows for residuals correlated between
/// neighbouring pixels, and a combination of `Σ` the print around the
/// corner constrains weakly (with one edge direction, the width along that
/// edge) shows as a large variance.
///
/// Under a non-Gaussian PSF (a defocus disc, a halo) `Σ` is the covariance
/// of the Gaussian that blurs the print most like the lens does, not the
/// PSF's second moment; it then depends on the cell size and the pattern,
/// so compare values measured on the same board.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
#[non_exhaustive]
pub struct FeatureBlur {
    /// The PSF covariance `[xx, xy, yy]`, px², including the pixel's area
    /// integration.
    pub psf_px: [f64; 3],
    /// The covariance of `psf_px`, px⁴, row-major.
    pub cov: [[f64; 3]; 3],
}

impl FeatureBlur {
    /// The blur of PSF covariance `psf_px` (`[xx, xy, yy]`, px²) with the
    /// covariance `cov` of those three elements (px⁴).
    pub fn new(psf_px: [f64; 3], cov: [[f64; 3]; 3]) -> Self {
        Self { psf_px, cov }
    }

    /// Whether the calibration accepts this blur as an observation
    /// (`vision_calibration_core::PointBlur::new`): every entry finite, `Σ`
    /// positive definite, and `cov` symmetric and positive semi-definite, to
    /// `1e-9` of its largest variance. A detector drops an estimate that
    /// fails this rather than emit one that fails its image downstream.
    ///
    /// The detectors do not depend on the calibration types, so this is the
    /// same check written without them, a little stricter on
    /// semi-definiteness: it requires `cov + (5e-10 · max variance)·I` to
    /// pass a Cholesky factorization, which holds only if the smallest
    /// eigenvalue is above `−5e-10 · max variance`, inside the calibration's
    /// `−1e-9 · max variance` by far more than either computation's
    /// rounding. What this accepts the calibration accepts; the two agree
    /// except on covariances already within rounding of indefinite.
    #[doc(hidden)]
    pub fn is_valid(&self) -> bool {
        let ([xx, xy, yy], cov) = (self.psf_px, self.cov);
        let finite = self
            .psf_px
            .iter()
            .chain(cov.iter().flatten())
            .all(|v| v.is_finite());
        if !finite || !(xx > 0.0 && yy > 0.0 && xx * yy - xy * xy > 0.0) {
            return false;
        }
        let scale = (0..3).map(|i| cov[i][i]).fold(0.0, f64::max);
        let tolerance = 1e-9 * scale;
        let symmetric = (0..3).all(|r| (0..r).all(|k| (cov[r][k] - cov[k][r]).abs() <= tolerance));
        if !symmetric {
            return false;
        }
        if scale == 0.0 {
            // No variance above zero: semi-definite only as all zeros.
            return cov.iter().flatten().all(|&v| v == 0.0);
        }
        // Cholesky of the symmetric part, shifted by half the tolerance.
        let shift = 0.5 * tolerance;
        let a =
            |r: usize, k: usize| 0.5 * (cov[r][k] + cov[k][r]) + if r == k { shift } else { 0.0 };
        let mut l = [[0.0_f64; 3]; 3];
        for r in 0..3 {
            for k in 0..=r {
                let dot: f64 = (0..k).map(|j| l[r][j] * l[k][j]).sum();
                if r == k {
                    let pivot = a(r, r) - dot;
                    if pivot.is_nan() || pivot <= 0.0 {
                        return false;
                    }
                    l[r][r] = pivot.sqrt();
                } else {
                    l[r][k] = (a(r, k) - dot) / l[k][k];
                }
            }
        }
        true
    }
}

/// Reject a detection whose image or target points are not distinct.
///
/// The targets this crate detects are planar, so the image↔target map is a
/// homography and therefore a bijection on valid correspondences: one pixel
/// maps to one board point and one board point maps to one pixel. A detection
/// that repeats either side has mislabelled its grid.
///
/// **The whole detection is rejected** (empty result — the same "no usable
/// board" signal every detector already uses for an empty frame), not just
/// the offending correspondences. A duplicated image or world point is not a
/// local blemish, it is evidence that the grid walk lost track of which corner
/// it was on; the labels that did *not* happen to collide come from the same
/// broken walk and are no more trustworthy. Dropping only the duplicates
/// leaves a handful of survivors that reproject in the thousands of pixels
/// and silently dominate the camera's mean.
///
/// The caller sees a skipped view, which is a normal and well-handled
/// outcome — vastly preferable to a view with no consistent pose.
///
/// The check costs one pass over the features. It is a cheap invariant, not
/// a workaround: it holds under any lens and any viewpoint, so it is always
/// safe to apply.
///
/// # What this does *not* catch
///
/// Duplicated points are the degenerate extreme of a broader failure: a
/// detector can also emit labels that are fully distinct yet still wrong,
/// with no homography fitting any subset. Testing for that requires the
/// lattice the grid builder just walked, so it belongs in the detector
/// backend rather than at this boundary — applying a projective test here
/// would need to assume distortion is small over the observed corners, and
/// would reject good views on wide-angle cameras.
pub fn reject_ambiguous_detection(features: Vec<Feature>) -> Vec<Feature> {
    use std::collections::HashSet;

    // Exact bit equality after normalizing `-0.0 -> +0.0` is the right
    // comparison: these duplicates are the *same* corner or grid point
    // emitted repeatedly, not two independently-refined points that happen
    // to land close together. A tolerance would risk rejecting views over
    // genuinely distinct neighbouring corners on a dense board.
    let mut seen_img: HashSet<[u64; 2]> = HashSet::with_capacity(features.len());
    let mut seen_world: HashSet<[u64; 3]> = HashSet::with_capacity(features.len());
    if features
        .iter()
        .all(|f| seen_img.insert(img_key(f)) && seen_world.insert(world_key(f)))
    {
        features
    } else {
        Vec::new()
    }
}

/// How far `features` repeat themselves: the number of features whose pixel
/// repeats an earlier feature's and the number whose target point does, or
/// `None` when both are zero. Repetition is judged exactly as
/// [`reject_ambiguous_detection`] judges it, which drops the detection
/// whenever this is `Some`.
pub(crate) fn repetitions(features: &[Feature]) -> Option<(usize, usize)> {
    use std::collections::HashSet;

    let mut seen_img: HashSet<[u64; 2]> = HashSet::with_capacity(features.len());
    let mut seen_world: HashSet<[u64; 3]> = HashSet::with_capacity(features.len());
    let (mut pixels, mut points) = (0, 0);
    for f in features {
        pixels += usize::from(!seen_img.insert(img_key(f)));
        points += usize::from(!seen_world.insert(world_key(f)));
    }
    (pixels + points > 0).then_some((pixels, points))
}

fn norm_bits(x: f64) -> u64 {
    if x == 0.0 {
        0.0_f64.to_bits()
    } else {
        x.to_bits()
    }
}

fn img_key(f: &Feature) -> [u64; 2] {
    [norm_bits(f.image_xy[0]), norm_bits(f.image_xy[1])]
}

fn world_key(f: &Feature) -> [u64; 3] {
    [
        norm_bits(f.world_xyz[0]),
        norm_bits(f.world_xyz[1]),
        norm_bits(f.world_xyz[2]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(x: f64, y: f64, wx: f64) -> Feature {
        Feature::new([x, y], [wx, 0.0, 0.0])
    }

    fn blur() -> FeatureBlur {
        FeatureBlur::new(
            [2.25, 0.1, 1.96],
            [[1e-3, 2e-5, -1e-4], [2e-5, 4e-4, 3e-6], [-1e-4, 3e-6, 9e-4]],
        )
    }

    #[test]
    fn blur_json_roundtrip_and_old_json_reads() {
        let with = feat(1.0, 2.0, 0.0).with_blur(Some(blur()));
        let json = serde_json::to_string(&with).unwrap();
        assert!(
            json.contains(r#""blur":{"psf_px":[2.25,0.1,1.96],"cov":"#),
            "{json}"
        );
        assert_eq!(serde_json::from_str::<Feature>(&json).unwrap(), with);

        let both = with.with_cov_px(Some([0.01, 0.0, 0.01]));
        let json = serde_json::to_string(&both).unwrap();
        assert_eq!(serde_json::from_str::<Feature>(&json).unwrap(), both);

        let without = feat(1.0, 2.0, 0.0);
        let json = serde_json::to_string(&without).unwrap();
        assert!(!json.contains("blur"), "{json}");
        assert_eq!(serde_json::from_str::<Feature>(&json).unwrap(), without);

        // A record written before the field existed reads as no blur.
        let old = r#"{"image_xy":[1.0,2.0],"world_xyz":[0.0,0.0,0.0],"cov_px":[0.01,0.0,0.01]}"#;
        let read = serde_json::from_str::<Feature>(old).unwrap();
        assert_eq!(read.blur, None);
        assert_eq!(read.cov_px, Some([0.01, 0.0, 0.01]));

        // The value and its covariance come together or not at all.
        let half =
            r#"{"image_xy":[1.0,2.0],"world_xyz":[0.0,0.0,0.0],"blur":{"psf_px":[1.0,0.0,1.0]}}"#;
        assert!(serde_json::from_str::<Feature>(half).is_err());
    }

    #[test]
    fn cov_px_json_roundtrip_and_old_json_reads() {
        let with = Feature {
            cov_px: Some([0.01, 0.002, 0.02]),
            ..feat(1.0, 2.0, 0.0)
        };
        let json = serde_json::to_string(&with).unwrap();
        assert_eq!(serde_json::from_str::<Feature>(&json).unwrap(), with);

        let without = feat(1.0, 2.0, 0.0);
        let json = serde_json::to_string(&without).unwrap();
        assert!(!json.contains("cov_px"), "{json}");
        assert_eq!(serde_json::from_str::<Feature>(&json).unwrap(), without);

        let old = r#"{"image_xy":[1.0,2.0],"world_xyz":[0.0,0.0,0.0]}"#;
        assert_eq!(serde_json::from_str::<Feature>(old).unwrap().cov_px, None);
    }

    #[test]
    fn repetitions_count_what_reject_ambiguous_detection_drops() {
        let distinct = vec![feat(1.0, 2.0, 0.0), feat(3.0, 4.0, 0.005)];
        assert_eq!(repetitions(&distinct), None);
        let both = vec![
            feat(5.0, 5.0, 0.0),
            feat(5.0, 5.0, 0.01),
            feat(-0.0, 5.0, 0.01),
            feat(0.0, 5.0, 0.02),
        ];
        assert_eq!(repetitions(&both), Some((2, 1)));
        assert!(reject_ambiguous_detection(both).is_empty());
    }

    #[test]
    fn distinct_points_pass_through_unchanged() {
        let fs = vec![feat(1.0, 2.0, 0.0), feat(3.0, 4.0, 0.005)];
        assert_eq!(reject_ambiguous_detection(fs.clone()), fs);
    }

    #[test]
    fn a_conflicted_pixel_rejects_the_whole_detection() {
        // The shape seen upstream: one pixel labelled with a run of
        // consecutive board cells. The unaffected corners go too — they came
        // from the same broken grid walk.
        let fs = vec![
            feat(10.0, 20.0, 0.0),
            feat(50.0, 60.0, 0.0052),
            feat(50.0, 60.0, 0.0104),
            feat(50.0, 60.0, 0.0156),
            feat(70.0, 80.0, 0.0208),
        ];
        assert!(reject_ambiguous_detection(fs).is_empty());
    }

    #[test]
    fn a_pixel_repeated_with_one_label_is_still_ambiguous() {
        // Same pixel twice is a duplicate correspondence even when both
        // carry the same target point: it double-weights that observation.
        let fs = vec![feat(5.0, 5.0, 0.0), feat(5.0, 5.0, 0.0)];
        assert!(reject_ambiguous_detection(fs).is_empty());
    }

    #[test]
    fn duplicate_world_xyz_and_signed_zero_reject_the_detection() {
        let dup_world = vec![feat(5.0, 5.0, 0.01), feat(15.0, 25.0, 0.01)];
        assert!(reject_ambiguous_detection(dup_world).is_empty());

        let neg_zero = vec![feat(-0.0, 5.0, 0.0), feat(0.0, 5.0, 0.01)];
        assert!(reject_ambiguous_detection(neg_zero).is_empty());

        let neg_zero_world = vec![feat(5.0, 5.0, -0.0), feat(15.0, 25.0, 0.0)];
        assert!(reject_ambiguous_detection(neg_zero_world).is_empty());
    }
}
