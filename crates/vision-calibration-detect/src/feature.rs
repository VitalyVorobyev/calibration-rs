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
pub struct Feature {
    /// Pixel coordinates `[x, y]` (column, row).
    pub image_xy: [f64; 2],
    /// World coordinates `[x, y, z]` in metres in the target frame.
    pub world_xyz: [f64; 3],
}

/// Reject a detection whose image points are not distinct.
///
/// The targets this crate detects are planar, so the image↔target map is a
/// homography and therefore injective: one pixel is one board point. A
/// detection that breaks this has mislabelled its grid.
///
/// **The whole detection is rejected** (empty result — the same "no usable
/// board" signal every detector already uses for an empty frame), not just
/// the offending correspondences. A duplicated image point is not a local
/// blemish, it is evidence that the grid walk lost track of which corner it
/// was on; the labels that did *not* happen to collide are produced by the
/// same broken walk and are no more trustworthy. Measured on a ChArUco rig
/// view that collapsed four consecutive board cells onto one corner: dropping
/// only the duplicates left six survivors that reprojected at ~1800 px, while
/// every healthy view in the same camera sat under 1.3 px.
///
/// The caller sees a skipped view, which is a normal and well-handled outcome
/// — vastly preferable to a view with no consistent pose, whose handful of
/// ~10³ px residuals silently dominates a camera's mean.
///
/// Observed against `calib-targets` 0.12 ChArUco detections; reported as
/// <https://github.com/VitalyVorobyev/calib-targets-rs/issues/86>. This guard
/// stays regardless of that fix: it costs one pass over the features and
/// converts a silent, hard-to-attribute accuracy loss into a skipped frame.
///
/// # What this does *not* catch
///
/// Duplicated points are the degenerate extreme of a broader upstream
/// failure: the detector can also emit a set of labels that is fully distinct
/// yet still wrong. The general test is projective — for a planar target the
/// grid-to-image map is a homography, so a detection can be checked against
/// itself by fitting one to its own labels. Measured over 20 views of one
/// camera, 18 healthy views fit at 0.56–0.99 px median while the two broken
/// ones sat at 7.7–7.8 px, with no subset of the bad labels admitting a fit.
///
/// That test is deliberately *not* applied here. It is only valid when lens
/// distortion over the observed corners is small — the map is really
/// `homography ∘ distortion` — so on a wide-angle camera it would reject
/// perfectly good views. It belongs upstream, inside the grid builder that
/// knows the lattice it just walked, which is what the issue above asks for.
pub fn reject_ambiguous_detection(features: Vec<Feature>) -> Vec<Feature> {
    use std::collections::HashSet;

    // Exact bit equality is the right comparison: these duplicates are the
    // *same* corner emitted repeatedly, not two independently-refined points
    // that happen to land close together. A tolerance would risk rejecting
    // views over genuinely distinct neighbouring corners on a dense board.
    let mut seen: HashSet<[u64; 2]> = HashSet::with_capacity(features.len());
    if features.iter().all(|f| seen.insert(key(f))) {
        features
    } else {
        Vec::new()
    }
}

fn key(f: &Feature) -> [u64; 2] {
    [f.image_xy[0].to_bits(), f.image_xy[1].to_bits()]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(x: f64, y: f64, wx: f64) -> Feature {
        Feature {
            image_xy: [x, y],
            world_xyz: [wx, 0.0, 0.0],
        }
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
}
