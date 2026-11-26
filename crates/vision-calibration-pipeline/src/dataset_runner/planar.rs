//! `DatasetSpec → PlanarDataset` converter for the `PlanarIntrinsics`
//! topology (any camera model; the config names the sensor).

use std::path::{Path, PathBuf};

use vision_calibration_core::{NoMeta, PlanarDataset, RunControl, View};
use vision_calibration_dataset::{DatasetSpec, ProblemKind, validate};

use super::convert_options::{ConvertOptions, kept};
use super::{
    Detection, ImageProgress, RejectReason, RejectedDetection, RunError, detect_features,
    expand_camera_images, features_to_view, pattern_repr, target_to_detector_config,
    view_filter::{ViewFilter, no_usable_views},
};

/// Convert a single-camera planar manifest + cache into the existing
/// [`PlanarDataset`] IR. Accepts the `PlanarIntrinsics` topology — a
/// tilted (Scheimpflug) camera consumes the same input, only its config
/// differs. Returns the
/// dataset plus a per-view list of the source image paths (for
/// downstream `image_manifest` population).
///
/// `control` is checked before each image and told each one done
/// ([`Progress::Detection`](vision_calibration_core::Progress)); once its
/// token is cancelled the conversion stops with [`RunError::Cancelled`].
pub fn build_planar_input(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    control: &RunControl,
) -> Result<PlanarRunResult, RunError> {
    build_planar_input_with(
        spec,
        base_dir,
        detection,
        control,
        &ConvertOptions::default(),
    )
}

/// [`build_planar_input`], leaving out what `options` names.
pub(crate) fn build_planar_input_with(
    spec: &DatasetSpec,
    base_dir: &Path,
    detection: Detection<'_>,
    control: &RunControl,
    options: &ConvertOptions,
) -> Result<PlanarRunResult, RunError> {
    // Order matters: gate on topology + target before touching the
    // filesystem so users with broken globs still see helpful
    // "wrong topology" or "wrong target" errors first.
    if spec.topology != ProblemKind::PlanarIntrinsics {
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

    let mut progress = ImageProgress::targets(control, total);
    for image_path in &images {
        if options.excludes(0, image_path) {
            progress.skip();
            rejected.push(RejectedDetection {
                camera: 0,
                path: image_path.clone(),
                reason: RejectReason::Excluded,
            });
            continue;
        }
        let (mut features, _cache_hit) =
            progress.read(|| detect_features(&detector_config, roi, image_path, detection))?;

        let mut screened = filter.check(&features);
        if screened.is_ok()
            && let Some(keep) = options.kept_features(0, image_path, &features)
        {
            features = kept(&features, &keep);
            screened = filter.check_masked(&features);
        }
        if let Err(reason) = screened {
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
        DatasetSpec::new(
            vec![CameraSource::new(
                "cam0",
                ImagePattern::List { paths: vec![] },
            )],
            TargetSpec::chessboard(9, 6, 0.025),
            ProblemKind::PlanarIntrinsics,
        )
    }

    #[test]
    fn rejects_rig_topology() {
        let mut spec = planar_spec_for_globless_test();
        spec.topology = ProblemKind::RigExtrinsics;
        spec.cameras.push(CameraSource::new(
            "cam1",
            ImagePattern::List { paths: vec![] },
        ));
        let cache = FsDetectionCache::new(std::env::temp_dir().join("calib-test-cache"));
        let err = build_planar_input(
            &spec,
            Path::new("/tmp"),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
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
            vision_calibration_detect::ChessboardConfig::new(9, 6, 0.025),
        );
        // Views showing 9, 5 and 3 of the board's 9 rows.
        let rel_paths = ["a.png", "b.png", "c.png"];
        for (rel, rows) in rel_paths.iter().zip([9u32, 5, 3]) {
            let features: Vec<Feature> = (0..rows)
                .flat_map(|v| (0..6u32).map(move |u| (u, v)))
                .map(|(u, v)| {
                    Feature::new(
                        [40.0 * f64::from(u), 40.0 * f64::from(v)],
                        [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
                    )
                })
                .collect();
            std::fs::write(tmp.path().join(rel), rel.as_bytes()).unwrap();
            let key = CacheKey::for_detection(rel.as_bytes(), &config, None);
            cache.put(&key, &CachedFeatures::new(features)).unwrap();
        }
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::List {
            paths: rel_paths.iter().map(PathBuf::from).collect(),
        };
        let result = build_planar_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (3, 3));
        assert!(result.rejected.is_empty());
    }

    /// Three planar views of the whole 9 × 6 board, `a.png`, `b.png` and
    /// `c.png`, their detections in the cache.
    fn three_views(dir: &Path) -> (FsDetectionCache, DatasetSpec) {
        use vision_calibration_detect::{CacheKey, CachedFeatures};
        let cache = FsDetectionCache::new(dir.join("cache"));
        let config = vision_calibration_detect::DetectorConfig::Chessboard(
            vision_calibration_detect::ChessboardConfig::new(9, 6, 0.025),
        );
        let names = ["a.png", "b.png", "c.png"];
        for (i, rel) in names.iter().enumerate() {
            std::fs::write(dir.join(rel), rel.as_bytes()).unwrap();
            let key = CacheKey::for_detection(rel.as_bytes(), &config, None);
            cache
                .put(&key, &CachedFeatures::new(board(i as f64)))
                .unwrap();
        }
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::List {
            paths: names.iter().map(PathBuf::from).collect(),
        };
        (cache, spec)
    }

    /// The board's 54 corners, `(u, v)` at `0.025 · (u, v)` m, seen at an
    /// offset of `shift` pixels.
    fn board(shift: f64) -> Vec<vision_calibration_detect::Feature> {
        (0..9u32)
            .flat_map(|v| (0..6u32).map(move |u| (u, v)))
            .map(|(u, v)| {
                vision_calibration_detect::Feature::new(
                    [shift + 40.0 * f64::from(u), 40.0 * f64::from(v)],
                    [0.025 * f64::from(u), 0.025 * f64::from(v), 0.0],
                )
            })
            .collect()
    }

    /// An excluded image is not read, is listed as excluded, and still
    /// counts as done, so the progress reaches its total.
    #[test]
    fn an_excluded_image_is_left_out_unread() {
        use std::sync::{Arc, Mutex};
        use vision_calibration_core::Progress;
        let tmp = tempfile::tempdir().unwrap();
        let (cache, spec) = three_views(tmp.path());
        // b.png's bytes change, so its cached detection no longer applies:
        // reading it would decode them and fail.
        std::fs::write(tmp.path().join("b.png"), b"not an image").unwrap();
        let mut options = ConvertOptions::default();
        options.exclude_slot(0, &tmp.path().join("b.png"));
        let reports = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&reports);
        let control = RunControl::default().with_observer(move |p| {
            if let Progress::Detection { done, total, .. } = p {
                seen.lock().unwrap().push((done, total));
            }
        });
        let result = build_planar_input_with(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &control,
            &options,
        )
        .unwrap();
        assert_eq!((result.usable_views, result.total_views), (2, 3));
        assert_eq!(result.rejected.len(), 1);
        assert_eq!(result.rejected[0].reason, RejectReason::Excluded);
        assert!(result.rejected[0].path.ends_with("b.png"));
        assert!(result.view_paths[1].ends_with("c.png"));
        assert_eq!(reports.lock().unwrap().last(), Some(&(3, 3)));
    }

    /// A mask drops exactly the features at its points; a point it names
    /// that the detection lacks drops nothing; a view the mask leaves under
    /// the feature floor is left out for too few features.
    #[test]
    fn a_feature_mask_drops_the_features_at_its_points() {
        let tmp = tempfile::tempdir().unwrap();
        let (cache, spec) = three_views(tmp.path());
        let a = tmp.path().join("a.png");
        let mut options = ConvertOptions::default();
        options.mask_points(
            0,
            &a,
            &[[0.025, 0.05, 0.0], [0.0, 0.0, 0.0], [9.0, 9.0, 0.0]],
        );
        let masked = build_planar_input_with(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
            &options,
        )
        .unwrap();
        let unmasked = build_planar_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap();
        let view = &masked.input.views[0].obs;
        assert_eq!(view.points_3d.len(), 52);
        for dropped in [[0.025, 0.05], [0.0, 0.0]] {
            assert!(
                !view
                    .points_3d
                    .iter()
                    .any(|p| (p.x - dropped[0]).abs() < 1e-12 && (p.y - dropped[1]).abs() < 1e-12)
            );
        }
        // The other views are untouched.
        assert_eq!(
            format!("{:?}", &masked.input.views[1..]),
            format!("{:?}", &unmasked.input.views[1..])
        );

        let mut options = ConvertOptions::default();
        let all_but_three: Vec<[f64; 3]> = board(0.0).iter().skip(3).map(|f| f.world_xyz).collect();
        options.mask_points(0, &a, &all_but_three);
        let thinned = build_planar_input_with(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
            &options,
        )
        .unwrap();
        assert_eq!(thinned.usable_views, 2);
        assert_eq!(thinned.rejected.len(), 1);
        assert_eq!(thinned.rejected[0].reason, RejectReason::TooFewFeatures);
    }

    #[test]
    fn empty_image_match_is_actionable() {
        let mut spec = planar_spec_for_globless_test();
        spec.cameras[0].images = ImagePattern::Glob {
            pattern: "**/no_such_pattern_should_match_*.png".into(),
        };
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache = FsDetectionCache::new(tmp.path().join("cache"));
        let err = build_planar_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
        .unwrap_err();
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
        spec.detector = Some({
            let mut value = DetectorSpec::default();
            value.chess_corners = Some({
                let mut value = ChessCornersDetectorSpec::default();
                value.covariance = Some(CornerCovarianceSpec::default());
                value
            });
            value.min_features_per_view = None;
            value
        });
        spec.cameras[0].images = ImagePattern::List {
            paths: vec![PathBuf::from("a.png")],
        };
        std::fs::write(tmp.path().join("a.png"), b"a.png").unwrap();
        let config = target_to_detector_config(&spec).unwrap();
        let key = CacheKey::for_detection(b"a.png", &config, None);
        cache.put(&key, &CachedFeatures::new(features)).unwrap();
        build_planar_input(
            &spec,
            tmp.path(),
            Detection::Cached(&cache),
            &RunControl::default(),
        )
    }

    fn feature(i: u32, cov: Option<[f64; 3]>) -> vision_calibration_detect::Feature {
        vision_calibration_detect::Feature::new(
            [10.0 * f64::from(i), 7.0 * f64::from(i % 3)],
            [0.025 * f64::from(i), 0.025 * f64::from(i % 3), 0.0],
        )
        .with_cov_px(cov)
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

    /// A feature's measured blur reaches its observation, beside the
    /// covariance; a feature without one stays without.
    #[test]
    fn feature_blur_reaches_the_observations() {
        use vision_calibration_detect::FeatureBlur;
        let cov = [[1e-4, 0.0, 0.0], [0.0, 4e-5, 0.0], [0.0, 0.0, 9e-5]];
        let features = (0..6)
            .map(|i| {
                let blur = (i != 4).then(|| FeatureBlur::new([2.0, 0.1 * f64::from(i), 1.5], cov));
                feature(i, Some([0.04, 0.0, 0.09])).with_blur(blur)
            })
            .collect();
        let obs = &covariance_run(features).unwrap().input.views[0].obs;
        let blur = obs.blur(3).expect("blur attached");
        assert_eq!(
            (blur.psf_px(), blur.cov()),
            ([2.0, 0.30000000000000004, 1.5], cov)
        );
        assert_eq!(obs.blur(4), None);
        assert!(obs.covariance(4).is_some());
    }

    /// A cached blur that the observations cannot hold fails the image with
    /// a typed error rather than being dropped.
    #[test]
    fn an_invalid_cached_blur_is_a_typed_error() {
        use vision_calibration_detect::FeatureBlur;
        let features = (0..6)
            .map(|i| {
                let psf = if i == 2 {
                    [1.0, 2.0, 1.0]
                } else {
                    [1.0, 0.0, 1.0]
                };
                feature(i, Some([0.04, 0.0, 0.09]))
                    .with_blur(Some(FeatureBlur::new(psf, [[0.0; 3]; 3])))
            })
            .collect();
        match covariance_run(features).unwrap_err() {
            RunError::Detection {
                detector, source, ..
            } => {
                assert_eq!(detector, "chessboard");
                let message = source.to_string();
                assert!(message.contains("feature 2"), "{message}");
                assert!(message.contains("positive definite"), "{message}");
            }
            other => panic!("expected Detection, got {other:?}"),
        }
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
                assert!(fault.to_string().contains("stated"), "{fault}");
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
