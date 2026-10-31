//! Target detection for Tier-B runs.
//!
//! Detection and input building are the pipeline's: an entry's manifest
//! ([`crate::manifest::dataset_spec`]) goes through
//! `vision_calibration_pipeline::dataset_runner`, which runs the
//! `vision-calibration-detect` detectors through the detection cache. This
//! module adds only what the bench records about that step: per-camera
//! detection statistics, each camera's image size, and the wall-clock time.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use vision_calibration_core::{CorrespondenceView, NoMeta, PlanarDataset, RigDataset};
use vision_calibration_dataset::{DatasetSpec, TargetSpec, Topology};
use vision_calibration_detect::{
    CacheError, CacheKey, CachedFeatures, DetectionCache, FsDetectionCache,
};
use vision_calibration_optim::RobotPoseMeta;
use vision_calibration_pipeline::dataset_runner::{
    build_planar_input, build_rig_extrinsics_input, build_rig_handeye_input,
    build_single_cam_handeye_input,
};
use vision_calibration_pipeline::single_cam_handeye::SingleCamHandeyeInput;

use crate::record::{Detection, DetectionStat};
use crate::registry::BoardGeometry;

/// The detection cache of bench runs: `target/calib-bench-detection-cache`
/// in the workspace, shared by every entry (entries are keyed by image
/// content).
///
/// The key covers the image, the detector config and the
/// `vision-calibration-detect` release, not the versions of the detector
/// crates it wraps: after updating one of those between releases, delete
/// this directory so the next run detects afresh.
pub fn detection_cache() -> FsDetectionCache {
    FsDetectionCache::new(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/calib-bench-detection-cache"),
    )
}

/// A dataset after detection: the problem input the dataset runner built, and
/// what the bench records about the detection step.
#[derive(Debug, Clone)]
pub struct Detected<I> {
    /// The problem input.
    pub input: I,
    /// Per-camera detection statistics.
    pub detection: Detection,
    /// Each camera's image size `[width, height]` (its region of interest,
    /// when the manifest crops one), in manifest camera order.
    pub image_sizes: Vec<[u32; 2]>,
    /// Wall-clock detection time in milliseconds (cache reads included).
    pub detection_ms: u64,
    /// Each camera's source images behind its kept observations, in
    /// manifest camera order.
    pub image_paths: Vec<Vec<PathBuf>>,
}

/// Source image paths of a rig dataset's views, `[view][camera]`, `None` where
/// the camera contributed no observation.
pub type RigViewPaths = Vec<Vec<Option<PathBuf>>>;

/// Detect a single-camera planar dataset.
pub fn detect_planar(
    spec: &DatasetSpec,
    base_dir: &Path,
    board: Option<&BoardGeometry>,
    cache: &dyn DetectionCache,
) -> Result<Detected<PlanarDataset>> {
    let start = Instant::now();
    let built = build_planar_input(spec, base_dir, cache, false)?;
    let detection_ms = ms(start.elapsed());
    let counts = [built
        .input
        .views
        .iter()
        .map(|v| v.obs.len())
        .collect::<Vec<_>>()];
    Ok(Detected {
        detection: detection_stats(spec, board, &counts, built.total_views, detection_ms),
        image_sizes: image_sizes(spec, &[built.view_paths.iter().collect()])?,
        input: built.input,
        detection_ms,
        image_paths: vec![built.view_paths],
    })
}

/// Detect a single-camera hand-eye dataset.
pub fn detect_single_cam_handeye(
    spec: &DatasetSpec,
    base_dir: &Path,
    board: Option<&BoardGeometry>,
    cache: &dyn DetectionCache,
) -> Result<Detected<SingleCamHandeyeInput>> {
    let start = Instant::now();
    let built = build_single_cam_handeye_input(spec, base_dir, cache, false)?;
    let detection_ms = ms(start.elapsed());
    let counts = [built
        .input
        .views
        .iter()
        .map(|v| v.obs.len())
        .collect::<Vec<_>>()];
    Ok(Detected {
        detection: detection_stats(spec, board, &counts, built.total_views, detection_ms),
        image_sizes: image_sizes(spec, &[built.view_paths.iter().collect()])?,
        input: built.input,
        detection_ms,
        image_paths: vec![built.view_paths],
    })
}

/// Detect a rig-extrinsics dataset.
pub fn detect_rig_extrinsics(
    spec: &DatasetSpec,
    base_dir: &Path,
    board: Option<&BoardGeometry>,
    cache: &dyn DetectionCache,
) -> Result<Detected<RigDataset<NoMeta>>> {
    let start = Instant::now();
    let built = build_rig_extrinsics_input(spec, base_dir, cache, false)?;
    let detection_ms = ms(start.elapsed());
    rig_detected(
        spec,
        board,
        built.input,
        &built.view_paths,
        built.total_views,
        detection_ms,
    )
}

/// Detect a rig hand-eye dataset; returns the source paths of its views too.
pub fn detect_rig_handeye(
    spec: &DatasetSpec,
    base_dir: &Path,
    board: Option<&BoardGeometry>,
    cache: &dyn DetectionCache,
) -> Result<(Detected<RigDataset<RobotPoseMeta>>, RigViewPaths)> {
    let start = Instant::now();
    let built = build_rig_handeye_input(spec, base_dir, cache, false)?;
    let detection_ms = ms(start.elapsed());
    let detected = rig_detected(
        spec,
        board,
        built.input,
        &built.view_paths,
        built.total_views,
        detection_ms,
    )?;
    Ok((detected, built.view_paths))
}

fn rig_detected<M>(
    spec: &DatasetSpec,
    board: Option<&BoardGeometry>,
    input: RigDataset<M>,
    view_paths: &RigViewPaths,
    total_views: usize,
    detection_ms: u64,
) -> Result<Detected<RigDataset<M>>> {
    let n_cam = input.num_cameras;
    let counts: Vec<Vec<usize>> = (0..n_cam)
        .map(|camera| {
            input
                .views
                .iter()
                .filter_map(|v| v.obs.cameras[camera].as_ref().map(CorrespondenceView::len))
                .collect()
        })
        .collect();
    let paths: Vec<Vec<&PathBuf>> = (0..n_cam)
        .map(|camera| {
            view_paths
                .iter()
                .filter_map(|v| v[camera].as_ref())
                .collect()
        })
        .collect();
    Ok(Detected {
        detection: detection_stats(spec, board, &counts, total_views, detection_ms),
        image_sizes: image_sizes(spec, &paths)?,
        image_paths: paths
            .iter()
            .map(|p| p.iter().map(|&p| p.clone()).collect())
            .collect(),
        input,
        detection_ms,
    })
}

/// The bench's detection record from the feature count of every kept
/// observation, per camera.
///
/// A registry `board`, else a chessboard or ChArUco manifest target, sets the
/// expected count per view: its full-board count, or more when a detection
/// found more. Other targets have no fixed count, so their expected count and
/// coverage are zero.
fn detection_stats(
    spec: &DatasetSpec,
    board: Option<&BoardGeometry>,
    counts: &[Vec<usize>],
    images_total: usize,
    detection_ms: u64,
) -> Detection {
    let max_per_view = counts.iter().flatten().copied().max().unwrap_or(0);
    let per_view = board
        .map(board_feature_count)
        .or_else(|| target_feature_count(&spec.target))
        .map(|n| n.max(max_per_view));
    let per_camera: Vec<DetectionStat> = spec
        .cameras
        .iter()
        .zip(counts)
        .map(|(camera, counts)| {
            let images_used = counts.len();
            let features_detected: usize = counts.iter().sum();
            let features_expected = per_view.map_or(0, |n| images_used * n);
            DetectionStat {
                camera_id: camera.id.clone(),
                images_total,
                images_used,
                features_detected,
                features_expected,
                coverage_pct: if features_expected > 0 {
                    100.0 * features_detected as f64 / features_expected as f64
                } else {
                    0.0
                },
                detect_ms: detection_ms,
            }
        })
        .collect();
    Detection {
        total_detected: per_camera.iter().map(|s| s.features_detected).sum(),
        total_expected: per_camera.iter().map(|s| s.features_expected).sum(),
        per_camera,
    }
}

/// Full-board corner count of a chessboard (interior corners) or ChArUco
/// (inner corners of its squares) manifest target.
fn target_feature_count(target: &TargetSpec) -> Option<usize> {
    match *target {
        TargetSpec::Chessboard { rows, cols, .. } => Some(rows as usize * cols as usize),
        TargetSpec::Charuco { rows, cols, .. } => {
            Some(rows.saturating_sub(1) as usize * cols.saturating_sub(1) as usize)
        }
        _ => None,
    }
}

/// Full-board feature count of a registry board: `rows × cols`, or the
/// `(rows − 1) × (cols − 1)` inner corners of a PuzzleBoard, whose registry
/// dimensions count squares.
pub fn board_feature_count(board: &BoardGeometry) -> usize {
    match board.layout.as_deref() {
        Some(layout) if layout.eq_ignore_ascii_case("puzzleboard") => {
            board.rows.saturating_sub(1) * board.cols.saturating_sub(1)
        }
        _ => board.rows * board.cols,
    }
}

/// Each camera's image size: its region of interest when the manifest crops
/// one, otherwise the size of the images its observations came from, which
/// must agree.
fn image_sizes(spec: &DatasetSpec, paths: &[Vec<&PathBuf>]) -> Result<Vec<[u32; 2]>> {
    spec.cameras
        .iter()
        .zip(paths)
        .map(|(camera, paths)| {
            if let Some([_, _, w, h]) = camera.roi_xywh {
                return Ok([w, h]);
            }
            let mut size = None;
            for path in paths {
                let (w, h) = image::image_dimensions(path)
                    .with_context(|| format!("read the size of {}", path.display()))?;
                match size {
                    None => size = Some([w, h]),
                    Some(previous) => anyhow::ensure!(
                        previous == [w, h],
                        "camera {}: images differ in size ({}×{} and {w}×{h})",
                        camera.id,
                        previous[0],
                        previous[1]
                    ),
                }
            }
            size.with_context(|| format!("camera {}: no image to size", camera.id))
        })
        .collect()
}

/// The single-camera planar manifest of `spec`'s camera `camera`: the same
/// images, region and target, without the other cameras or robot poses.
pub fn camera_spec(spec: &DatasetSpec, camera: usize) -> DatasetSpec {
    DatasetSpec {
        cameras: vec![spec.cameras[camera].clone()],
        topology: Topology::PlanarIntrinsics,
        robot_poses: None,
        pose_pairing: None,
        pose_convention: None,
        laser: None,
        upstream_calibration: None,
        ..spec.clone()
    }
}

/// A detection cache that never hits and records how long each detection
/// takes: the time from the lookup that missed to the store of its result,
/// which covers decoding, cropping and detecting one image.
///
/// After `limit` detections it answers every further lookup with an empty
/// detection, so the runner skips the remaining images (they count as views
/// without a board). Profiling runs the dataset runner through it, one camera
/// at a time.
#[derive(Debug, Default)]
pub struct TimingCache {
    limit: Option<usize>,
    state: Mutex<TimingState>,
}

#[derive(Debug, Default)]
struct TimingState {
    pending: Option<Instant>,
    durations: Vec<Duration>,
}

impl TimingCache {
    /// A cache that detects at most `limit` images (all when `None`).
    pub fn new(limit: Option<usize>) -> Self {
        Self {
            limit,
            state: Mutex::default(),
        }
    }

    /// Per-image detection times, in detection order.
    pub fn durations(&self) -> Vec<Duration> {
        self.lock().durations.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TimingState> {
        // A poisoned lock only means a timing was lost; keep the rest.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl DetectionCache for TimingCache {
    fn get(&self, _key: &CacheKey) -> Result<Option<CachedFeatures>, CacheError> {
        let mut state = self.lock();
        if self
            .limit
            .is_some_and(|limit| state.durations.len() >= limit)
        {
            return Ok(Some(CachedFeatures {
                features: Vec::new(),
            }));
        }
        state.pending = Some(Instant::now());
        Ok(None)
    }

    fn put(&self, _key: &CacheKey, _value: &CachedFeatures) -> Result<(), CacheError> {
        let mut state = self.lock();
        if let Some(start) = state.pending.take() {
            state.durations.push(start.elapsed());
        }
        Ok(())
    }
}

/// Whole milliseconds of `duration`.
pub fn ms(duration: Duration) -> u64 {
    duration.as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(layout: &str) -> BoardGeometry {
        BoardGeometry {
            rows: 130,
            cols: 130,
            cell_size_m: 0.001,
            dictionary: None,
            layout: Some(layout.into()),
            marker_size_rel: None,
            strict_grid: false,
        }
    }

    #[test]
    fn puzzleboard_counts_inner_corners() {
        assert_eq!(board_feature_count(&board("puzzleboard")), 129 * 129);
        assert_eq!(board_feature_count(&board("charuco")), 130 * 130);
    }

    #[test]
    fn manifest_targets_with_a_fixed_corner_count() {
        let chessboard = TargetSpec::Chessboard {
            rows: 17,
            cols: 28,
            square_size_m: 0.02,
        };
        assert_eq!(target_feature_count(&chessboard), Some(17 * 28));
        let charuco = TargetSpec::Charuco {
            rows: 22,
            cols: 22,
            square_size_m: 0.0045,
            marker_size_m: 0.0034,
            dictionary: "DICT_4X4_1000".into(),
        };
        assert_eq!(target_feature_count(&charuco), Some(21 * 21));
    }

    #[test]
    fn timing_cache_times_each_miss_until_its_store() {
        let cache = TimingCache::new(Some(2));
        let key = CacheKey::from_inputs(b"img", "chessboard", &serde_json::json!({}));
        for _ in 0..2 {
            assert!(cache.get(&key).unwrap().is_none());
            cache
                .put(&key, &CachedFeatures { features: vec![] })
                .unwrap();
        }
        // Past the limit every lookup is an empty hit.
        let skipped = cache.get(&key).unwrap().expect("hit");
        assert!(skipped.features.is_empty());
        assert_eq!(cache.durations().len(), 2);
    }
}
