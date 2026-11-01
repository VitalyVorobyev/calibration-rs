//! `DatasetSpec → PlanarDataset` converter for the `PlanarIntrinsics`
//! topology (any camera model; the config names the sensor).

use std::path::{Path, PathBuf};

use vision_calibration_core::{NoMeta, PlanarDataset, View};
use vision_calibration_dataset::{DatasetSpec, Topology, validate};

use super::{
    Detection, RejectedDetection, RunError, detect_features, expand_camera_images,
    features_to_view, pattern_repr, target_to_detector_config,
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
    detection: Detection<'_>,
) -> Result<PlanarRunResult, RunError> {
    // Order matters: gate on topology + target before touching the
    // filesystem so users with broken globs still see helpful
    // "wrong topology" or "wrong target" errors first.
    if spec.topology != Topology::PlanarIntrinsics {
        return Err(RunError::UnsupportedTopology {
            topology: spec.topology,
        });
    }
    let detector_config = target_to_detector_config(spec)?;
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

    // The runner crops to the ROI before detecting; the crop is part of the
    // cache key.
    let roi = camera.roi_xywh;

    for image_path in &images {
        let (features, _cache_hit) = detect_features(&detector_config, roi, image_path, detection)?;

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
    use vision_calibration_detect::DetectionCache;
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
        let err =
            build_planar_input(&spec, Path::new("/tmp"), Detection::Cached(&cache)).unwrap_err();
        assert!(matches!(err, RunError::UnsupportedTopology { .. }));
    }

    #[test]
    fn planar_intrinsics_keeps_partial_chessboards() {
        // Each planar view has its own target pose, so a board labelled
        // from its own top-left corner is as good as a whole one.
        use vision_calibration_detect::{CacheKey, CachedFeatures, Feature};
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let config = vision_calibration_detect::DetectorConfig::Chessboard(
            vision_calibration_detect::ChessboardConfig {
                rows: 9,
                cols: 6,
                square_size_m: 0.025,
                chess_corners: None,
            },
        );
        // Views showing 9, 5 and 3 of the board's 9 rows.
        let rel_paths = ["a.png", "b.png", "c.png"];
        for (rel, rows) in rel_paths.iter().zip([9u32, 5, 3]) {
            let features: Vec<Feature> = (0..rows)
                .flat_map(|v| (0..6u32).map(move |u| (u, v)))
                .map(|(u, v)| Feature {
                    cov_px: None,
                    image_xy: [40.0 * f64::from(u), 40.0 * f64::from(v)],
                    world_xyz: [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
                })
                .collect();
            std::fs::write(tmp.path().join(rel), rel.as_bytes()).unwrap();
            let key = CacheKey::for_detection(rel.as_bytes(), &config, None);
            cache.put(&key, &CachedFeatures { features }).unwrap();
        }
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::List {
            paths: rel_paths.iter().map(PathBuf::from).collect(),
        };
        let result = build_planar_input(&spec, tmp.path(), Detection::Cached(&cache)).unwrap();
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
        let err = build_planar_input(&spec, tmp.path(), Detection::Cached(&cache)).unwrap_err();
        match err {
            RunError::EmptyImageMatch { camera_id, .. } => assert_eq!(camera_id, "cam0"),
            other => panic!("expected EmptyImageMatch, got {other:?}"),
        }
    }

    /// A planar manifest over a chessboard with covariance requested, plus a
    /// detection cache primed with `features` for one image.
    fn covariance_run(
        features: Vec<vision_calibration_detect::Feature>,
    ) -> Result<PlanarRunResult, RunError> {
        use vision_calibration_dataset::{
            ChessCornersDetectorSpec, CornerCovarianceSpec, DetectorSpec,
        };
        use vision_calibration_detect::{CacheKey, CachedFeatures};
        let tmp = tempfile::tempdir().unwrap();
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let mut spec = planar_spec_for_globless_test();
        spec.detector = Some(DetectorSpec {
            chess_corners: Some(ChessCornersDetectorSpec {
                covariance: Some(CornerCovarianceSpec::default()),
                ..ChessCornersDetectorSpec::default()
            }),
            min_features_per_view: None,
        });
        spec.cameras[0].images = ImagePattern::List {
            paths: vec![PathBuf::from("a.png")],
        };
        std::fs::write(tmp.path().join("a.png"), b"a.png").unwrap();
        let config = target_to_detector_config(&spec).unwrap();
        let key = CacheKey::for_detection(b"a.png", &config, None);
        cache.put(&key, &CachedFeatures { features }).unwrap();
        build_planar_input(&spec, tmp.path(), Detection::Cached(&cache))
    }

    fn feature(i: u32, cov: Option<[f64; 3]>) -> vision_calibration_detect::Feature {
        vision_calibration_detect::Feature {
            image_xy: [10.0 * f64::from(i), 7.0 * f64::from(i % 3)],
            world_xyz: [0.025 * f64::from(i), 0.025 * f64::from(i % 3), 0.0],
            cov_px: cov,
        }
    }

    #[test]
    fn requested_covariance_reaches_the_observations() {
        let features = (0..6)
            .map(|i| feature(i, Some([0.04, 0.0, 0.09])))
            .collect();
        let result = covariance_run(features).unwrap();
        let obs = &result.input.views[0].obs;
        let cov = obs.covariance(2).expect("covariances attached");
        assert_eq!((cov.xx(), cov.xy(), cov.yy()), (0.04, 0.0, 0.09));
    }

    #[test]
    fn requested_covariance_missing_on_a_corner_is_a_typed_error() {
        // Cached features are checked too, and no corner is dropped.
        let features = (0..6)
            .map(|i| feature(i, (i != 3).then_some([0.04, 0.0, 0.09])))
            .collect();
        match covariance_run(features).unwrap_err() {
            RunError::CornerCovariance {
                detector,
                path,
                fault,
            } => {
                assert_eq!(detector, "chessboard");
                assert!(path.ends_with("a.png"));
                assert_eq!(
                    fault,
                    crate::dataset_runner::CovarianceFault::Missing {
                        missing: 1,
                        total: 6
                    }
                );
                assert!(fault.to_string().contains("known"), "{fault}");
            }
            other => panic!("expected CornerCovariance, got {other:?}"),
        }
    }

    #[test]
    fn requested_covariance_that_is_not_positive_definite_is_a_typed_error() {
        let features = (0..6)
            .map(|i| {
                feature(
                    i,
                    Some(if i == 2 {
                        [1.0, 2.0, 1.0]
                    } else {
                        [0.04, 0.0, 0.09]
                    }),
                )
            })
            .collect();
        match covariance_run(features).unwrap_err() {
            RunError::CornerCovariance { fault, .. } => assert!(matches!(
                fault,
                crate::dataset_runner::CovarianceFault::NotPositiveDefinite { .. }
            )),
            other => panic!("expected CornerCovariance, got {other:?}"),
        }
    }
}
