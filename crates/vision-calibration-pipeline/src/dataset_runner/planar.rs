//! `DatasetSpec → PlanarDataset` converter, shared by the
//! `PlanarIntrinsics` and `ScheimpflugIntrinsics` topologies (both
//! problem types consume the same `PlanarDataset` IR).

use std::path::{Path, PathBuf};

use vision_calibration_core::{NoMeta, PlanarDataset, View};
use vision_calibration_dataset::{DatasetSpec, Topology, validate};
use vision_calibration_detect::DetectionCache;

use super::{
    RunError, augment_config_with_roi, detect_features, expand_camera_images, features_to_view,
    pattern_repr, pick_detector, target_to_detector_config,
};

/// Convert a single-camera planar manifest + cache into the existing
/// [`PlanarDataset`] IR. Accepts the `PlanarIntrinsics` and
/// `ScheimpflugIntrinsics` topologies — the Scheimpflug problem type
/// consumes the same input, only its config differs. Returns the
/// dataset plus a per-view list of the source image paths (for
/// downstream `image_manifest` population).
pub fn build_planar_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    cache: &dyn DetectionCache,
    force_redetect: bool,
) -> Result<PlanarRunResult, RunError> {
    // Order matters: gate on topology + target before touching the
    // filesystem so users with broken globs still see helpful
    // "wrong topology" or "wrong target" errors first.
    if !matches!(
        spec.topology,
        Topology::PlanarIntrinsics | Topology::ScheimpflugIntrinsics
    ) {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let (detector_name, detector_config) = target_to_detector_config(spec)?;
    let detector = pick_detector(detector_name)?;
    validate(spec)?;

    // Single-camera topologies have the validator-enforced invariant
    // of exactly one camera, so `cameras[0]` is the only camera — no
    // silent truncation.
    let camera = spec
        .cameras
        .first()
        .expect("validate guarantees exactly one camera for planar topologies");
    let images = expand_camera_images(camera, base_dir)?;
    if images.is_empty() {
        return Err(RunError::EmptyImageMatch {
            camera_id: camera.id.clone(),
            pattern: pattern_repr(&camera.images),
            base: base_dir.to_path_buf(),
        });
    }

    let mut views: Vec<View<NoMeta>> = Vec::new();
    let mut view_paths: Vec<PathBuf> = Vec::new();
    let mut usable = 0usize;
    let total = images.len();

    // ROI is part of what determines detection output, so it has to be
    // part of the cache key. We splice it into the key-side config and
    // leave the actual `detector_config` untouched (the detector itself
    // doesn't know about ROI — the runner crops the image first).
    let roi = camera.roi_xywh;
    let key_config = augment_config_with_roi(&detector_config, roi);

    for image_path in &images {
        let (features, _cache_hit) = detect_features(
            detector.as_ref(),
            detector_name,
            &detector_config,
            &key_config,
            roi,
            image_path,
            cache,
            force_redetect,
        )?;

        if features.len() < 4 {
            // Too few features for a homography — drop the view but
            // keep walking so we can surface a meaningful error if
            // _everything_ failed.
            continue;
        }
        usable += 1;
        views.push(features_to_view(&features)?);
        view_paths.push(image_path.clone());
    }

    if views.is_empty() {
        return Err(RunError::InsufficientUsableViews {
            camera_id: camera.id.clone(),
            usable,
            total,
        });
    }

    let dataset = PlanarDataset::new(views).expect(
        "PlanarDataset::new failed after we already filtered to >=4 \
         features per view; this indicates a regression in the core \
         crate's invariants",
    );

    Ok(PlanarRunResult {
        dataset,
        view_paths,
        usable_views: usable,
        total_views: total,
    })
}

/// Result of a planar dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct PlanarRunResult {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub dataset: PlanarDataset,
    /// One source image path per accepted view, in the same order.
    /// Used to populate the export's `image_manifest` after the solve.
    pub view_paths: Vec<PathBuf>,
    /// Number of views where detection produced ≥4 features.
    pub usable_views: usize,
    /// Total number of images attempted.
    pub total_views: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_calibration_dataset::{CameraSource, ImagePattern, TargetSpec};
    use vision_calibration_detect::FsDetectionCache;

    pub(crate) fn planar_spec_for_globless_test() -> DatasetSpec {
        DatasetSpec {
            version: 1,
            cameras: vec![CameraSource {
                id: "cam0".into(),
                images: ImagePattern::List { paths: vec![] },
                roi_xywh: None,
                laser_images: None,
            }],
            target: TargetSpec::Chessboard {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
            },
            detector: None,
            robot_poses: None,
            laser: None,
            upstream_calibration: None,
            topology: Topology::PlanarIntrinsics,
            pose_pairing: None,
            pose_convention: None,
            unresolved: vec![],
            description: None,
        }
    }

    #[test]
    fn rejects_rig_topology() {
        let mut spec = planar_spec_for_globless_test();
        spec.topology = Topology::RigExtrinsics;
        spec.cameras.push(CameraSource {
            id: "cam1".into(),
            images: ImagePattern::List { paths: vec![] },
            roi_xywh: None,
            laser_images: None,
        });
        let cache = FsDetectionCache::new(std::env::temp_dir().join("calib-test-cache"));
        let err = build_planar_input(&spec, Path::new("/tmp"), &cache, false).unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTopology { .. }));
    }

    #[test]
    fn accepts_scheimpflug_topology() {
        // Scheimpflug shares the planar input path; the topology gate
        // must not reject it. (The empty image list then fails
        // validation, which proves we got past the gate.)
        let mut spec = planar_spec_for_globless_test();
        spec.topology = Topology::ScheimpflugIntrinsics;
        let cache = FsDetectionCache::new(std::env::temp_dir().join("calib-test-cache"));
        let err = build_planar_input(&spec, Path::new("/tmp"), &cache, false).unwrap_err();
        assert!(
            !matches!(err, RunError::UnsupportedTopology { .. }),
            "scheimpflug must pass the topology gate, got {err:?}"
        );
    }

    #[test]
    fn empty_image_match_is_actionable() {
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "**/no_such_pattern_should_match_*.png".into(),
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let err = build_planar_input(&spec, tmp.path(), &cache, false).unwrap_err();
        match err {
            RunError::EmptyImageMatch { camera_id, .. } => assert_eq!(camera_id, "cam0"),
            other => panic!("expected EmptyImageMatch, got {other:?}"),
        }
    }
}
