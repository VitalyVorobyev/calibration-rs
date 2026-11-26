//! What a conversion leaves out of the calibration input beyond what the
//! view filter rejects: [`ConvertOptions`], crate-private.
//!
//! Every public converter runs with the default options, which leave out
//! nothing, and then skips every branch here: its result is the one it gave
//! before the options existed, bit for bit. A project's exclusions resolve
//! to options (`project::exclude`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use vision_calibration_core::lexical_absolute;
use vision_calibration_detect::Feature;

/// Largest distance, per coordinate, at which a feature's target point is
/// the point an exclusion names, metres. Target pitches are at least a
/// millimetre, so this tells neighbours apart with room to spare while it
/// absorbs a point written out in decimal and read back.
pub(crate) const POINT_TOLERANCE_M: f64 = 1e-6;

/// The `(view, camera)` slots and the features a conversion leaves out.
///
/// A slot is its camera's index and its image, compared as
/// [`lexical_absolute`] states the path.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ConvertOptions {
    /// Slots that are not read: the detection is left out as
    /// [`RejectReason::Excluded`](super::RejectReason::Excluded).
    excluded: BTreeSet<(usize, PathBuf)>,
    /// Target points (metres, as the detector labels them) whose features
    /// a slot's detection drops.
    masks: BTreeMap<(usize, PathBuf), Vec<[f64; 3]>>,
}

impl ConvertOptions {
    /// Leave camera `camera`'s image `path` out unread.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the project's exclusions are its caller")
    )]
    pub(crate) fn exclude_slot(&mut self, camera: usize, path: &Path) {
        self.excluded.insert((camera, lexical_absolute(path)));
    }

    /// Drop the features at `points` from camera `camera`'s detection in
    /// image `path`.
    #[cfg_attr(
        not(test),
        allow(dead_code, reason = "the project's exclusions are its caller")
    )]
    pub(crate) fn mask_points(&mut self, camera: usize, path: &Path, points: &[[f64; 3]]) {
        self.masks
            .entry((camera, lexical_absolute(path)))
            .or_default()
            .extend_from_slice(points);
    }

    /// Whether camera `camera`'s image `path` is left out unread.
    pub(crate) fn excludes(&self, camera: usize, path: &Path) -> bool {
        !self.excluded.is_empty() && self.excluded.contains(&(camera, lexical_absolute(path)))
    }

    /// Which of `features`, camera `camera`'s detection in image `path`,
    /// the options keep; `None` when they drop none of the slot's.
    ///
    /// A feature is dropped when its target point (in the detector's
    /// labelling, before a rig turns it) lies within
    /// [`POINT_TOLERANCE_M`] of a masked point in every coordinate.
    pub(crate) fn kept_features(
        &self,
        camera: usize,
        path: &Path,
        features: &[Feature],
    ) -> Option<Vec<bool>> {
        if self.masks.is_empty() {
            return None;
        }
        let points = self.masks.get(&(camera, lexical_absolute(path)))?;
        let keep: Vec<bool> = features
            .iter()
            .map(|f| {
                !points.iter().any(|p| {
                    p.iter()
                        .zip(&f.world_xyz)
                        .all(|(a, b)| (a - b).abs() <= POINT_TOLERANCE_M)
                })
            })
            .collect();
        keep.contains(&false).then_some(keep)
    }
}

/// The features `keep` marks.
pub(crate) fn kept(features: &[Feature], keep: &[bool]) -> Vec<Feature> {
    features
        .iter()
        .zip(keep)
        .filter(|(_, keep)| **keep)
        .map(|(f, _)| *f)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feature(x: f64) -> Feature {
        Feature::new([0.0, 0.0], [x, 0.0, 0.0])
    }

    #[test]
    fn default_options_leave_out_nothing() {
        let options = ConvertOptions::default();
        assert!(!options.excludes(0, Path::new("a.png")));
        assert!(
            options
                .kept_features(0, Path::new("a.png"), &[feature(0.0)])
                .is_none()
        );
    }

    #[test]
    fn a_slot_is_its_camera_and_its_image_however_the_path_is_spelled() {
        let mut options = ConvertOptions::default();
        options.exclude_slot(1, Path::new("data/cam1/./a.png"));
        assert!(options.excludes(1, Path::new("data/cam0/../cam1/a.png")));
        assert!(!options.excludes(0, Path::new("data/cam1/a.png")));
        assert_ne!(options, ConvertOptions::default());
    }

    #[test]
    fn a_mask_drops_the_features_at_its_points() {
        let mut options = ConvertOptions::default();
        let path = Path::new("a.png");
        options.mask_points(0, path, &[[0.03 + 4e-7, 0.0, 0.0], [9.0, 9.0, 9.0]]);
        let features = [feature(0.0), feature(0.03), feature(0.06)];
        let keep = options.kept_features(0, path, &features).unwrap();
        assert_eq!(keep, [true, false, true]);
        assert_eq!(kept(&features, &keep), [feature(0.0), feature(0.06)]);
        // A mask that matches none of the slot's features keeps them all.
        assert!(
            options
                .kept_features(0, path, &[feature(0.0), feature(0.06)])
                .is_none()
        );
        // Another camera's detection of the same image is untouched.
        assert!(options.kept_features(1, path, &features).is_none());
    }
}
