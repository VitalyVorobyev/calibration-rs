//! `DatasetSpec → PlanarDataset` converter for the `PlanarIntrinsics`
//! topology (any camera model; the config names the sensor).

use std::path::{Path, PathBuf};

use vision_calibration_core::{NoMeta, PlanarDataset, View};
use vision_calibration_dataset::{DatasetSpec, Topology, validate};
use vision_calibration_detect::DetectionCache;

use super::{
    RejectedDetection, RunError, augment_config_with_roi, detect_features, expand_camera_images,
    features_to_view, pattern_repr, pick_detector, target_to_detector_config,
    view_filter::{ViewFilter, no_usable_views},
};

/// Convert a single-camera planar manifest + cache into the existing
/// [`PlanarDataset`] IR. Accepts the `PlanarIntrinsics` topology — a
/// tilted (Scheimpflug) camera consumes the same input, only its config
/// differs. Returns the
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
    if spec.topology != Topology::PlanarIntrinsics {
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
    let mut rejected: Vec<RejectedDetection> = Vec::new();
    let mut usable = 0usize;
    let total = images.len();

    let filter = ViewFilter::for_spec(spec);

    // ROI is part of what determines detection output, so it has to be
    // part of the cache key. We splice it into the key-side config and
    // leave the actual `detector_config` untouched (the detector itself
    // doesn't know about ROI — the runner crops the image first).
    let roi = camera.roi_xywh;
    let key_config = augment_config_with_roi(&detector_config, roi);

    for image_path in &images {
        let (features, _cache_hit) = detect_features(
            detector.as_ref(),
            &detector_config,
            &key_config,
            roi,
            image_path,
            cache,
            force_redetect,
        )?;

        if let Err(reason) = filter.check(&features) {
            // Drop the view but keep walking, so an all-rejected camera
            // surfaces a meaningful error.
            rejected.push(RejectedDetection {
                camera: 0,
                path: image_path.clone(),
                reason,
            });
            continue;
        }
        usable += 1;
        views.push(features_to_view(&features)?);
        view_paths.push(image_path.clone());
    }

    if views.is_empty() {
        return Err(no_usable_views(spec, 0, &rejected, total));
    }

    let dataset = PlanarDataset::new(views).expect(
        "PlanarDataset::new failed after we already filtered to >=4 \
         features per view; this indicates a regression in the core \
         crate's invariants",
    );

    Ok(PlanarRunResult {
        input: dataset,
        view_paths,
        usable_views: usable,
        total_views: total,
        rejected,
    })
}

/// Result of a planar dataset conversion.
#[derive(Debug)]
#[non_exhaustive]
pub struct PlanarRunResult {
    /// The IR ready to feed into `CalibrationSession::set_input`.
    pub input: PlanarDataset,
    /// One source image path per accepted view, in the same order.
    /// Used to populate the export's `image_manifest` after the solve.
    pub view_paths: Vec<PathBuf>,
    /// Number of accepted views.
    pub usable_views: usize,
    /// Total number of images attempted.
    pub total_views: usize,
    /// The detections left out, in image order, each with its reason.
    pub rejected: Vec<RejectedDetection>,
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
    fn planar_intrinsics_keeps_partial_chessboards() {
        // Each planar view has its own target pose, so a board labelled
        // from its own top-left corner is as good as a whole one.
        use vision_calibration_detect::{CacheKey, CachedFeatures, ChessboardDetector, Feature};
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let config = serde_json::json!({"rows": 9, "cols": 6, "square_size_m": 0.025});
        // Views showing 9, 5 and 3 of the board's 9 rows.
        let rel_paths = ["a.png", "b.png", "c.png"];
        for (rel, rows) in rel_paths.iter().zip([9u32, 5, 3]) {
            let features: Vec<Feature> = (0..rows)
                .flat_map(|v| (0..6u32).map(move |u| (u, v)))
                .map(|(u, v)| Feature {
                    image_xy: [40.0 * f64::from(u), 40.0 * f64::from(v)],
                    world_xyz: [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
                })
                .collect();
            std::fs::write(tmp.path().join(rel), rel.as_bytes()).unwrap();
            let key = CacheKey::for_detector(rel.as_bytes(), &ChessboardDetector, &config);
            cache.put(&key, &CachedFeatures { features }).unwrap();
        }
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::List {
            paths: rel_paths.iter().map(PathBuf::from).collect(),
        };
        let result = build_planar_input(&spec, tmp.path(), &cache, false).unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        assert!(result.rejected.is_empty());
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
