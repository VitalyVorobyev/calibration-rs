//! What a project's detection cache knows of its images: the overview the
//! Detect screen's matrix shows (per view and camera, the features found
//! and the detector's outcome), read through the library's
//! `dataset_runner::cached_detection`, which keys every image as a run
//! does: nothing is decoded, detected or written.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use schemars::JsonSchema;
use serde::Serialize;
use vision_calibration::dataset::{DatasetSpec, RinggridLatticeSpec, TargetSpec};
use vision_calibration::dataset_runner::{RejectReason, cached_detection};
use vision_calibration::detect::{DetectionDetail, DetectionOutcome, FsDetectionCache};
use vision_calibration::project::{Project, RunSource, RunState};

use crate::api::StudioError;
use crate::runs::slash;

/// What the cache holds of one camera's image.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ImageDetectionState {
    /// No detection of the image with the project's detector settings is
    /// cached: it was never detected, or the image or the settings changed
    /// since.
    NotDetected,
    /// Detected: the features found, and the detector's outcome. `outcome`
    /// is `detected` when features were found; when none were, the cached
    /// detail's account (`not_found` with the detector's reason, `ambiguous`
    /// or `filtered`: a calibration caches it for an image it found nothing
    /// in), or `null` when only the (empty) features are cached, by a cache
    /// written before calibrations kept the detail.
    Detected {
        features: usize,
        outcome: Option<DetectionOutcome>,
    },
    /// The image could not be read or detected.
    Failed { error: StudioError },
}

/// One camera's image of a view.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ImageDetection {
    /// The image, relative to the data folder (absolute outside it),
    /// `/`-separated: as `project_views` lists it, and as an exclusion
    /// names it.
    pub image: String,
    pub state: ImageDetectionState,
    /// Why the overview's run left the image out; `null` when it used it
    /// (or there is no run).
    pub left_out: Option<RejectReason>,
}

/// One view: each camera's image.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ViewDetections {
    /// The view's pairing token (as `project_views` lists it).
    pub token: String,
    /// One entry per camera; `null` where the camera has no image of the
    /// view.
    pub images: Vec<Option<ImageDetection>>,
}

/// The project's images as its detection cache knows them.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct DetectionOverview {
    /// The cameras' ids, in manifest order.
    pub cameras: Vec<String>,
    /// The features a complete detection of the board has (corners of a
    /// chessboard or ChArUco board, markers of a ring grid); `null` for a
    /// PuzzleBoard, which is made to be seen in part.
    pub expected: Option<usize>,
    /// The views, in pairing order.
    pub views: Vec<ViewDetections>,
    /// The run whose `left_out` reasons are given: the one asked for, else
    /// the project's latest calibrated run; `null` when there is none.
    pub run: Option<String>,
}

/// The feature count of a complete board, where it is known.
pub fn expected_features(target: &TargetSpec) -> Option<usize> {
    let n = |v: u32| v as usize;
    match target {
        TargetSpec::Chessboard { rows, cols, .. } => Some(n(*rows) * n(*cols)),
        TargetSpec::Charuco { rows, cols, .. } => {
            Some(n(rows.saturating_sub(1)) * n(cols.saturating_sub(1)))
        }
        TargetSpec::Ringgrid {
            rows,
            long_row_cols,
            lattice,
            ..
        } => match lattice {
            RinggridLatticeSpec::Rect => Some(n(*rows) * n(*long_row_cols)),
            // Rows alternate between `long_row_cols` markers and one fewer.
            RinggridLatticeSpec::Hex => {
                let long = n(*long_row_cols);
                let rows = n(*rows);
                Some(rows.div_ceil(2) * long + rows / 2 * long.saturating_sub(1))
            }
            _ => None,
        },
        _ => None,
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What the project's cache holds of camera `camera`'s image `path`.
fn cache_state(
    cache: &FsDetectionCache,
    dataset: &DatasetSpec,
    data_root: &Path,
    camera: usize,
    path: &Path,
) -> ImageDetectionState {
    match cached_detection(dataset, data_root, camera, path, cache) {
        Ok(None) => ImageDetectionState::NotDetected,
        Ok(Some(cached)) => ImageDetectionState::Detected {
            features: cached.feature_count,
            outcome: if cached.feature_count > 0 {
                Some(DetectionOutcome::Detected)
            } else {
                cached.detail_outcome
            },
        },
        Err(e) => ImageDetectionState::Failed { error: e.into() },
    }
}

/// The state a detection's detail gives.
pub fn state_of(detail: &DetectionDetail) -> ImageDetectionState {
    ImageDetectionState::Detected {
        features: detail.features.len(),
        outcome: Some(detail.outcome.clone()),
    }
}

/// One camera's image of a view, by position.
pub struct Cell {
    pub view: usize,
    pub camera: usize,
    pub path: PathBuf,
}

/// A project's images and what an overview of them needs.
pub struct Images {
    pub dataset: DatasetSpec,
    pub data_root: PathBuf,
    pub cameras: Vec<String>,
    pub tokens: Vec<String>,
    /// `paths[view][camera]`, absolute.
    pub paths: Vec<Vec<Option<PathBuf>>>,
}

impl Images {
    /// The project's dataset and its paired images (no image is read).
    pub fn of(project: &Project) -> Result<Self, StudioError> {
        let dataset = project.dataset()?;
        let data_root = project.data_root()?;
        let views = project.views()?;
        Ok(Self {
            cameras: dataset.cameras.iter().map(|c| c.id.clone()).collect(),
            dataset,
            data_root,
            tokens: views.tokens,
            paths: views.paths,
        })
    }

    /// Every image, view by view, camera by camera.
    pub fn cells(&self) -> Vec<Cell> {
        self.paths
            .iter()
            .enumerate()
            .flat_map(|(view, cameras)| {
                cameras
                    .iter()
                    .enumerate()
                    .filter_map(move |(camera, path)| {
                        path.as_ref().map(|path| Cell {
                            view,
                            camera,
                            path: path.clone(),
                        })
                    })
            })
            .collect()
    }

    /// `path` as an overview and an exclusion name it.
    pub fn relative(&self, path: &Path) -> String {
        slash(path.strip_prefix(&self.data_root).unwrap_or(path))
    }

    /// The overview of `states` (one per [`cells`](Self::cells) entry),
    /// with the left-out reasons of `run`.
    pub fn overview(
        &self,
        states: Vec<ImageDetectionState>,
        run: Option<LeftOut>,
    ) -> DetectionOverview {
        let mut views: Vec<ViewDetections> = self
            .tokens
            .iter()
            .zip(&self.paths)
            .map(|(token, cameras)| ViewDetections {
                token: token.clone(),
                images: vec![None; cameras.len()],
            })
            .collect();
        let (run, left_out) = match run {
            Some(run) => (Some(run.run), run.images),
            None => (None, Vec::new()),
        };
        for (cell, state) in self.cells().into_iter().zip(states) {
            let image = self.relative(&cell.path);
            let camera_id = self.cameras.get(cell.camera);
            let left_out = left_out
                .iter()
                .find(|(camera, path, _)| Some(camera) == camera_id && *path == image)
                .map(|(_, _, reason)| *reason);
            // `cells` walks `paths`, which `views` mirrors.
            if let Some(slot) = views[cell.view].images.get_mut(cell.camera) {
                *slot = Some(ImageDetection {
                    image,
                    state,
                    left_out,
                });
            }
        }
        DetectionOverview {
            cameras: self.cameras.clone(),
            expected: expected_features(&self.dataset.target),
            views,
            run,
        }
    }
}

/// The images a run left out.
pub struct LeftOut {
    /// The run's id.
    pub run: String,
    /// `(camera id, image relative to the data folder, reason)`.
    pub images: Vec<(String, String, RejectReason)>,
}

/// The images run `id` (or the latest calibrated run) left out.
pub fn left_out(project: &Project, id: Option<&str>) -> Result<Option<LeftOut>, StudioError> {
    let info = match id {
        Some(id) => Some(project.run(id)?),
        None => project.runs()?.into_iter().rev().find(|run| {
            matches!(&run.state, RunState::Recorded(record)
                if matches!(record.source, RunSource::Calibrated(_)))
        }),
    };
    let Some(info) = info else { return Ok(None) };
    let reasons = match &info.state {
        RunState::Recorded(record) => match &record.source {
            RunSource::Calibrated(run) => run
                .left_out
                .iter()
                .map(|l| (l.camera.clone(), slash(&l.image), l.reason))
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    Ok(Some(LeftOut {
        run: info.id,
        images: reasons,
    }))
}

/// What the project's cache holds of every image, read in parallel.
pub fn cache_states(images: &Images, cache: &FsDetectionCache) -> Vec<ImageDetectionState> {
    let cells = images.cells();
    for_each_parallel(&cells, |cell| {
        cache_state(
            cache,
            &images.dataset,
            &images.data_root,
            cell.camera,
            &cell.path,
        )
    })
}

/// `f` of every item on up to `available_parallelism` threads; results in
/// the items' order.
pub fn for_each_parallel<I: Sync, R: Send>(items: &[I], f: impl Fn(&I) -> R + Sync) -> Vec<R> {
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(items.len().max(1));
    let next = AtomicUsize::new(0);
    let results: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    let Some(item) = items.get(i) else { break };
                    let result = f(item);
                    lock(&results)[i] = Some(result);
                }
            });
        }
    });
    results
        .into_inner()
        .unwrap_or_else(|e| e.into_inner())
        .into_iter()
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_boards_have_a_known_count() {
        assert_eq!(
            expected_features(&TargetSpec::chessboard(5, 7, 0.02)),
            Some(35)
        );
        let charuco: TargetSpec = serde_json::from_value(serde_json::json!({
            "kind": "charuco", "rows": 5, "cols": 7, "square_size_m": 0.02,
            "marker_size_m": 0.015, "dictionary": "DICT_4X4_50"
        }))
        .unwrap();
        assert_eq!(expected_features(&charuco), Some(24));
        let puzzle: TargetSpec = serde_json::from_value(serde_json::json!({
            "kind": "puzzleboard", "layout": "puzzle_10x10", "cell_size_m": 0.01
        }))
        .unwrap();
        assert_eq!(expected_features(&puzzle), None);
    }

    #[test]
    fn parallel_results_keep_the_items_order() {
        let items: Vec<usize> = (0..37).collect();
        assert_eq!(
            for_each_parallel(&items, |i| i * 2),
            (0..37).map(|i| i * 2).collect::<Vec<_>>()
        );
        assert!(for_each_parallel(&Vec::<usize>::new(), |i| *i).is_empty());
    }
}
