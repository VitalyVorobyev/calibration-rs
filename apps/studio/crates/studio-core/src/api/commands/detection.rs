//! A project's detections: what its detection cache holds of every image
//! (`detection_overview`), one image's detection with what an overlay
//! needs (`detection_detail`), a preview of one image with edited detector
//! settings (`detect_preview`, writes nothing), and the job that detects
//! every image into the cache (`detect_all`). A project is named by its
//! folder; an image by its path relative to the data folder, as
//! `project_views` lists it. Every path the project names is judged
//! against the roots before an image is read.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use vision_calibration::dataset::DetectorSpec;
use vision_calibration::dataset_runner::{Detection, detection_detail as library_detail};
use vision_calibration::detect::{DetectionDetail, FsDetectionCache};
use vision_calibration::project::Project;

use crate::api::{Ctx, JobCx, StudioError};
use crate::detections::{
    DetectionOverview, ImageDetectionState, Images, cache_states, for_each_parallel, left_out,
    state_of,
};
use crate::runs::open_project;
use crate::scope::UserPath;

/// The project in `dir`, with every path it names judged.
fn checked_project(ctx: &Ctx, dir: &UserPath) -> Result<Project, StudioError> {
    let project = open_project(ctx, dir)?;
    ctx.scope.check_project(&project)?;
    Ok(project)
}

/// `image` (relative to `data_root`, or absolute) as the library takes it,
/// once the scope has found it inside a root.
fn image_path(ctx: &Ctx, data_root: &Path, image: &str) -> Result<PathBuf, StudioError> {
    let path = data_root.join(image);
    ctx.scope.check(&path)?;
    Ok(path)
}

/// What the project's detection cache holds of every image, view by view
/// and camera by camera (the views of `project_views`): the features found
/// and the detector's outcome, or that the image has no cached detection
/// with the project's detector settings. No image is decoded or detected;
/// each is read to find its cache entry. `left_out` gives why run `run`
/// (default: the latest calibrated run) left an image out.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetectionOverviewArgs {
    /// The project folder.
    pub dir: UserPath,
    #[serde(default)]
    pub run: Option<String>,
}

pub fn detection_overview(
    ctx: &Ctx,
    args: DetectionOverviewArgs,
) -> Result<DetectionOverview, StudioError> {
    let project = checked_project(ctx, &args.dir)?;
    let images = Images::of(&project)?;
    let cache = FsDetectionCache::new(project.detection_cache_dir());
    let states = cache_states(&images, &cache);
    Ok(images.overview(states, left_out(&project, args.run.as_deref())?))
}

/// Camera `camera`'s detection of `image` (relative to the data folder, as
/// `project_views` lists it) with what an overlay needs: the features (in
/// the camera's region of interest when it has one), the target's per-kind
/// detail aligned with them, the candidates the detector left out and the
/// image's outcome. Read from the project's detection cache, or detected
/// and stored there (features and detail) when it holds none: the
/// detection a calibration of the project uses. A feature is excluded by
/// its `world_xyz`. `invalid` for an image that is not one of the
/// camera's.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetectionDetailArgs {
    /// The project folder.
    pub dir: UserPath,
    pub camera: usize,
    pub image: String,
}

pub fn detection_detail(
    ctx: &Ctx,
    args: DetectionDetailArgs,
) -> Result<DetectionDetail, StudioError> {
    let project = checked_project(ctx, &args.dir)?;
    let dataset = project.dataset()?;
    let data_root = project.data_root()?;
    let path = image_path(ctx, &data_root, &args.image)?;
    let cache = FsDetectionCache::new(project.detection_cache_dir());
    Ok(library_detail(
        &dataset,
        &data_root,
        args.camera,
        &path,
        Detection::Cached(&cache),
    )?)
}

/// `detection_detail` of one image with `detector` in place of the
/// project's detector settings (the manifest's `detector` section; `{}` is
/// the detectors' defaults), to preview an edit: detected now, and nothing
/// is written (neither the project nor its detection cache).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetectPreviewArgs {
    /// The project folder.
    pub dir: UserPath,
    pub camera: usize,
    pub image: String,
    pub detector: DetectorSpec,
}

pub fn detect_preview(ctx: &Ctx, args: DetectPreviewArgs) -> Result<DetectionDetail, StudioError> {
    let project = checked_project(ctx, &args.dir)?;
    let mut dataset = project.dataset()?;
    let data_root = project.data_root()?;
    let path = image_path(ctx, &data_root, &args.image)?;
    // No section and an empty one detect alike; no section is what a
    // manifest without overrides says.
    dataset.detector = (args.detector != DetectorSpec::default()).then_some(args.detector);
    Ok(library_detail(
        &dataset,
        &data_root,
        args.camera,
        &path,
        Detection::Uncached,
    )?)
}

/// Detect every image of the project into its detection cache (images it
/// already holds are read from it, unless `redetect`), several at a time;
/// its events count the images done, its output is the
/// `detection_overview` of the result. An image that cannot be read or
/// detected is reported in the overview, not as the job's failure.
/// Cancelling stops between images; what was detected stays cached.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DetectAllArgs {
    /// The project folder.
    pub dir: UserPath,
    #[serde(default)]
    pub redetect: bool,
}

/// How far `detect_all` is.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DetectProgress {
    /// Images done.
    pub done: usize,
    /// Images in all.
    pub total: usize,
    /// The image last done: its camera and path (relative to the data
    /// folder).
    pub camera: usize,
    pub image: String,
}

/// Progress is reported at most this often (and for the last image).
const PROGRESS_PERIOD: Duration = Duration::from_millis(50);

pub fn detect_all(
    ctx: &Ctx,
    args: DetectAllArgs,
    job: &JobCx<DetectProgress>,
) -> Result<DetectionOverview, StudioError> {
    let project = checked_project(ctx, &args.dir)?;
    let images = Images::of(&project)?;
    let cache = FsDetectionCache::new(project.detection_cache_dir());
    let cells = images.cells();
    let total = cells.len();
    let token = job.token();
    let done = AtomicUsize::new(0);
    // The clock only paces the reports; nothing it reads enters a result.
    let last_report: Mutex<Option<Instant>> = Mutex::new(None);
    let reporter = Mutex::new(job.clone());
    let states = for_each_parallel(&cells, |cell| {
        if token.is_cancelled() {
            return None;
        }
        let detection = if args.redetect {
            Detection::Refresh(&cache)
        } else {
            Detection::Cached(&cache)
        };
        let state = match library_detail(
            &images.dataset,
            &images.data_root,
            cell.camera,
            &cell.path,
            detection,
        ) {
            Ok(detail) => state_of(&detail),
            Err(e) => ImageDetectionState::Failed { error: e.into() },
        };
        let n = done.fetch_add(1, Ordering::SeqCst) + 1;
        let now = Instant::now();
        let mut last = last_report.lock().unwrap_or_else(|e| e.into_inner());
        if n == total || last.is_none_or(|t| now - t >= PROGRESS_PERIOD) {
            *last = Some(now);
            let report = DetectProgress {
                done: n,
                total,
                camera: cell.camera,
                image: images.relative(&cell.path),
            };
            reporter
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .emit(&report);
        }
        Some(state)
    });
    if token.is_cancelled() {
        // What was detected is cached: the project's overview changed.
        ctx.events.project_changed(project.dir());
        return Err(StudioError::cancelled());
    }
    ctx.events.project_changed(project.dir());
    let states = states.into_iter().flatten().collect();
    Ok(images.overview(states, left_out(&project, None)?))
}
