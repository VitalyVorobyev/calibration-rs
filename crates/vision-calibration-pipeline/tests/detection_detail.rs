//! The detail of a run's detection of one image: read through the run's
//! cache key and region of interest, consistent with the features the run
//! used, and never read by the calibration.

#![allow(missing_docs)]

use std::path::{Path, PathBuf};

use serde_json::json;
use vision_calibration_core::RunControl;
use vision_calibration_dataset::DatasetSpec;
use vision_calibration_detect::{
    CacheKey, CachedFeatures, DetectionCache, DetectionOutcome, DetectorConfig, FsDetectionCache,
    GrayImageView, TargetDetail,
};
use vision_calibration_pipeline::dataset_runner::{
    Detection, RunError, build_planar_input, detect_target, detect_target_detail, detection_detail,
};

/// The rendered page of a frontal chessboard of 6 × 8 inner corners.
fn board_page() -> image::GrayImage {
    let doc = calib_targets::generate::chessboard_document(6, 8, 20.0);
    let bundle = calib_targets::printable::render_target_bundle(&doc).expect("render bundle");
    image::load_from_memory(&bundle.png_bytes)
        .expect("decode rendered PNG")
        .to_luma8()
}

const PAD: u32 = 40;

/// The page on a white image `PAD` pixels larger each way, at `offset`.
fn board_image(offset: u32) -> image::GrayImage {
    let page = board_page();
    let mut image =
        image::GrayImage::from_pixel(page.width() + PAD, page.height() + PAD, image::Luma([255]));
    image::imageops::replace(&mut image, &page, i64::from(offset), i64::from(offset));
    image
}

/// Two views of the board at different offsets in `dir`; a planar manifest
/// over them whose camera crops `roi`.
fn dataset(dir: &Path, roi: Option<[u32; 4]>) -> DatasetSpec {
    board_image(0).save(dir.join("a.png")).unwrap();
    board_image(PAD).save(dir.join("b.png")).unwrap();
    let mut camera =
        json!({ "id": "cam0", "images": { "kind": "list", "paths": ["a.png", "b.png"] } });
    if let Some(roi) = roi {
        camera["roi_xywh"] = json!(roi);
    }
    serde_json::from_value(json!({
        "version": 1,
        "topology": "planar_intrinsics",
        "cameras": [camera],
        "target": { "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 },
    }))
    .unwrap()
}

/// The features a run's converter cached for camera 0's image `name`.
fn cached_features(
    dir: &Path,
    cache: &FsDetectionCache,
    spec: &DatasetSpec,
    name: &str,
) -> CachedFeatures {
    let bytes = std::fs::read(dir.join(name)).unwrap();
    let key = CacheKey::for_detection(&bytes, &chessboard_config(), spec.cameras[0].roi_xywh);
    cache.get(&key).unwrap().expect("the run cached the image")
}

fn chessboard_config() -> DetectorConfig {
    DetectorConfig::Chessboard(vision_calibration_detect::ChessboardConfig::new(6, 8, 0.02))
}

fn detail_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".detail.json"))
        .collect()
}

#[test]
fn the_detail_of_a_run_image_matches_the_features_the_run_cached() {
    let data = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let cache = FsDetectionCache::new(cache_dir.path());
    // A region inside the white margin of both views: the ROI is part of
    // the key, and the features are in its pixels.
    let (w, h) = board_page().dimensions();
    let spec = dataset(data.path(), Some([PAD / 2, PAD / 2, w, h]));
    let run = build_planar_input(
        &spec,
        data.path(),
        Detection::Cached(&cache),
        &RunControl::default(),
    )
    .unwrap();
    assert_eq!(run.usable_views, 2);
    assert!(
        detail_files(cache_dir.path()).is_empty(),
        "a run writes no detail"
    );

    for name in ["a.png", "b.png"] {
        let cached = cached_features(data.path(), &cache, &spec, name);
        let detail = detection_detail(
            &spec,
            data.path(),
            0,
            Path::new(name),
            Detection::Cached(&cache),
        )
        .unwrap();
        assert_eq!(detail.features, cached.features, "{name}");
        assert_eq!(detail.outcome, DetectionOutcome::Detected);
        assert!(matches!(detail.target, TargetDetail::Chessboard(_)));
        // Read back from the cache the second time.
        let again = detection_detail(
            &spec,
            data.path(),
            0,
            &data.path().join(name),
            Detection::Cached(&cache),
        )
        .unwrap();
        assert_eq!(again, detail);
    }
    assert_eq!(detail_files(cache_dir.path()).len(), 2);

    // A run over the cache with the details in it reads the same features.
    let rerun = build_planar_input(
        &spec,
        data.path(),
        Detection::Cached(&cache),
        &RunControl::default(),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&rerun.input).unwrap(),
        serde_json::to_value(&run.input).unwrap()
    );
}

#[test]
fn cached_features_that_a_fresh_detection_contradicts_are_an_error() {
    let data = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();
    let cache = FsDetectionCache::new(cache_dir.path());
    let spec = dataset(data.path(), None);
    build_planar_input(
        &spec,
        data.path(),
        Detection::Cached(&cache),
        &RunControl::default(),
    )
    .unwrap();

    // Doctor the cached features of `a.png`: drop a corner.
    let bytes = std::fs::read(data.path().join("a.png")).unwrap();
    let key = CacheKey::for_detection(&bytes, &chessboard_config(), None);
    let mut features = cache.get(&key).unwrap().unwrap().features;
    features.pop();
    cache.put(&key, &CachedFeatures::new(features)).unwrap();

    let err = detection_detail(
        &spec,
        data.path(),
        0,
        Path::new("a.png"),
        Detection::Cached(&cache),
    )
    .unwrap_err();
    assert!(matches!(err, RunError::DetailMismatch { .. }), "{err}");
    assert!(
        detail_files(cache_dir.path()).is_empty(),
        "no detail stored"
    );

    // Refreshing detects again and overwrites both entries.
    let detail = detection_detail(
        &spec,
        data.path(),
        0,
        Path::new("a.png"),
        Detection::Refresh(&cache),
    )
    .unwrap();
    assert_eq!(cache.get(&key).unwrap().unwrap().features, detail.features);
    assert_eq!(cache.get_detail(&key).unwrap(), Some(detail));
}

#[test]
fn an_image_or_camera_outside_the_manifest_is_an_error() {
    let data = tempfile::tempdir().unwrap();
    let spec = dataset(data.path(), None);
    image::GrayImage::new(8, 8)
        .save(data.path().join("other.png"))
        .unwrap();
    for (camera, image) in [(0, "other.png"), (1, "a.png")] {
        let err = detection_detail(
            &spec,
            data.path(),
            camera,
            Path::new(image),
            Detection::Uncached,
        )
        .unwrap_err();
        assert!(matches!(err, RunError::ImageNotInDataset { .. }), "{err}");
    }
}

#[test]
fn the_in_memory_detail_matches_detect_target() {
    let image = board_image(0);
    let view = GrayImageView::new(image.width(), image.height(), image.as_raw()).unwrap();
    let target = serde_json::from_value(
        json!({ "kind": "chessboard", "rows": 6, "cols": 8, "square_size_m": 0.02 }),
    )
    .unwrap();
    let detail = detect_target_detail(view, &target, None).unwrap();
    assert_eq!(detail.features, detect_target(view, &target, None).unwrap());
    assert_eq!(detail.features.len(), 48);
}
